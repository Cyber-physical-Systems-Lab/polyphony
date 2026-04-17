import json
import re
import subprocess
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request

import gymnasium as gym
import numpy as np
import tarware
from tarware.definitions import CollisionLayers

from fixed_prompt_builder import render_template
from state_translation_helper import (
    battery_need_label,
    candidate_ids_for_prompt,
    charging_action_ids,
    charging_station_occupancy,
    classify_action,
    describe_action_id_for_agent,
    empty_shelf_action_ids,
    get_requested_action_ids,
    get_requested_shelves,
    render_self_state,
)


# Scenario notes are intentionally kept in code comments and logs rather than
# injected into prompts. They help analysis without leaking experiment labels
# into the model input.
@dataclass(frozen=True)
class ScenarioSpec:
    env_id: str
    label: str
    focus: str


MODELS: List[Tuple[str, str]] = [
    ("llama3.2:1b", "SLM"),
    ("llama3.2:3b", "SLM"),
    ("mistral:7b", "MLM"),
    ("gemma3:12b", "MLM"),
    ("phi4", "LLM"),
    ("qwen2.5:14b", "LLM"),
]


SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        env_id="tarware-tiny-1agvs-1pickers-globalobs-chg-v1",
        label="tiny_balanced_1v1",
        focus="Balanced baseline: reveal whether the model understands the normal AGV-Picker collaboration loop.",
    ),
    ScenarioSpec(
        env_id="tarware-small-2agvs-2pickers-globalobs-chg-v1",
        label="small_balanced_2v2",
        focus="Balanced coordination at slightly higher concurrency.",
    ),
    ScenarioSpec(
        env_id="tarware-medium-4agvs-4pickers-globalobs-chg-v1",
        label="medium_balanced_4v4",
        focus="Balanced scaling: larger warehouse and longer-horizon coordination.",
    ),
    ScenarioSpec(
        env_id="tarware-small-4agvs-2pickers-globalobs-chg-v1",
        label="small_agv_heavy_4v2",
        focus="AGV-heavy: reveal picker-support scheduling weakness and AGV waiting behavior.",
    ),
    ScenarioSpec(
        env_id="tarware-medium-6agvs-2pickers-globalobs-chg-v1",
        label="medium_agv_heavy_6v2",
        focus="Strongly AGV-heavy: stress-test support prioritization, waiting, and charging decisions.",
    ),
    ScenarioSpec(
        env_id="tarware-small-2agvs-4pickers-globalobs-chg-v1",
        label="small_picker_heavy_2v4",
        focus="Picker-heavy: reveal underutilization, role confusion, and unnecessary movement.",
    ),
]


STATED_OBJECTIVE = "Optimize calls to LLM and increase the total shelf delivery."
MIN_AGV_UNLOAD_WAIT_STEPS = 5


SHARED_CONTEXT_TEXT_TEMPLATE = (
    "Main objective: {stated_objective}\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load first moves to a requested shelf.\n"
    "- Picker should move to that same shelf location to support load/unload collaboration.\n"
    "- After load, AGV carries the requested shelf to a GOAL location.\n"
    "- GOAL marks delivery only; AGV still carries the same shelf after delivery.\n"
    "- After GOAL delivery, AGV should return the carried shelf to an EMPTY shelf location.\n"
    "- Picker support happens at shelf interaction points, not at GOAL.\n"
    "- AGV and Picker repeat this collaboration loop for the next request.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <30, need_charging_soon if >=30 and <65, not_needed if >=65.\n"
    "\n"
    "Shared coordination context:\n"
    "Nearest requested shelves (top 5 by distance from current agent):\n{requested_shelves}\n"
    "\n"
    "Charging place occupancy:\n{charging_occupancy}\n"
    "Your output should balance throughput with fewer unnecessary LLM replans.\n"
)


BASIC_SUMMARY_LINES = [
    "Multi-agent warehouse with AGV and Picker collaboration.",
    "Avoid any out-of-charge scenario.",
    "Use charging strategically so throughput does not collapse.",
    "AGV-Picker coordinate at shelf interactions (load/unload at shelf location).",
]


TYPICAL_FLOW_LINES = [
    "AGV without load first moves to a requested shelf.",
    "Picker should move to that same shelf location to support load/unload collaboration.",
    "If AGV has reached the requested shelf and Picker support is still needed there, AGV should usually hold position and wait.",
    "After load, AGV carries the requested shelf to a GOAL location.",
    "GOAL marks delivery only; AGV still carries the same shelf after delivery.",
    "After GOAL delivery, AGV should return the carried shelf to an EMPTY shelf location.",
    "Picker support happens at shelf interaction points, not at GOAL.",
    "AGV and Picker repeat this collaboration loop for the next request.",
]


BATTERY_MODEL_LINES = [
    "Move step consumption = 1.",
    "Load/Unload consumption = 2.",
    "Charging need level: critical if <30, need_charging_soon if >=30 and <65, not_needed if >=65.",
]


AGV_LANGUAGE_TEMPLATE = (
    "{shared_context}\n"
    "You are planning only for one AGV. Output should be of format: Action: selected action ID, Steps: suggested integer steps, Reason: very short justification. Only this much\n"
    "Current agent:\n"
    "- Agent: agent_{agent_id} (AGV)\n"
    "- State of the current Agent: {self_state}\n"
    "Last decided action state:\n"
    "{last_decided_action_text}\n"
    "\n"
    "{movement_warning_text}"
    "{blocking_warning_text}"
    "Current control state:\n"
    "{control_state_text}\n"
    "\n"
    "AGV state and delivery context:\n"
    "{agv_role_context_text}\n"
    "\n"
    "{post_delivery_helper_text}"
    "{empty_shelf_return_task_text}"
    "AGV flow rules:\n"
    "{agv_flow_rules_text}\n"
    "\n"
    "Current-state hard constraints:\n"
    "{agv_hard_constraints_text}\n"
    "\n"
    "AGV decision rules:\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, charging is forbidden.\n"
    "- If picker_support_inbound_now=yes and picker_at_same_cell=no, Picker is on the way to this same shelf now. Do not leave this shelf now.\n"
    "- If battery is critical, prioritize charging.\n"
    "- Use suggested_steps as the default Steps choice.\n"
    "- For movement, Steps should usually be close to distance_steps.\n"
    "- For waiting at a support shelf, prefer a smaller Steps value.\n"
    "- For charging, Steps should reflect how much charge is still needed.\n"
    "- When multiple requested shelf actions are valid and no stronger constraint applies, prefer the closest candidate by distance_steps.\n"
    "\n"
    "Candidate actions:\n"
    "{candidate_actions}\n"
    "\n"
    "Single-step decision:\n"
    "- Your job is to choose the next macro action for this AGV now.\n"
    "- Do not propose a strategy for future behavior.\n"
    "- Follow next_required_flow_state first, then candidate tags, then suggested_steps.\n"
    "- If an execution warning marks an action id as unreachable, selecting that same action id is invalid for this turn; choose another valid candidate.\n"
    "- If next_required_flow_state=move_requested_shelf_to_goal, choose a tag=deliver_now candidate.\n"
    "- If next_required_flow_state=return_delivered_shelf_to_empty_shelf, choose a tag=return_empty_shelf candidate.\n"
    "- If next_required_flow_state=wait_for_picker_at_requested_shelf, prefer the current shelf or tag=wait_for_picker_here.\n"
    "- If next_required_flow_state=charge_now, choose a charging candidate.\n"
    "- If picker_support_inbound_now=yes and picker_at_same_cell=no, stay on this shelf until Picker arrives and the shelf interaction completes.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, do not choose any charging action.\n"
    "- If battery is 100, do not choose any charging action.\n"
    "\n"
    "Forbidden outputs:\n"
    "- Do not write recommendations.\n"
    "- Do not rewrite flow rules.\n"
    "- Do not give example output.\n"
    "- Do not explain the system.\n"
    "- Do not output headings or bullet lists.\n"
    "- Do not output markdown.\n"
    "- Do not output action descriptions like SHELF@(x,y).\n"
    "- Do not output policy, report, or revised-rule text.\n"
    "\n"
    "Decision contract:\n"
    "- Choose exactly one action id from Candidate actions.\n"
    "- Choose exactly one Steps value.\n"
    "- Action must be the integer action id only.\n"
    "- Steps must be an integer only.\n"
    "- OUTPUT Should be of format: Action: selected action ID, Steps: suggested integer steps, Reason: very short justification. Only this much\n"
    "Action: selected action id\n"
    "Steps: integer between 1 and {max_hold_steps} suggested steps\n"
    "Reason: just one line very short reason why this action and why this many steps\n"
    "If unsure, still choose one valid integer action id and one Steps value.\n"
)


PICKER_LANGUAGE_TEMPLATE = (
    "{shared_context}\n"
    "You are planning only for one Picker. Output should be of format: Action: selected action ID, Steps: suggested integer steps, Reason: very short justification. Only this much\n"
    "Current agent:\n"
    "- Agent: agent_{agent_id} (PICKER)\n"
    "- State of the current Agent: {self_state}\n"
    "Last decided action state:\n"
    "{last_decided_action_text}\n"
    "\n"
    "{movement_warning_text}"
    "{blocking_warning_text}"
    "Current control state:\n"
    "{control_state_text}\n"
    "\n"
    "Picker support context:\n"
    "{picker_role_context_text}\n"
    "\n"
    "{post_delivery_helper_text}"
    "Current-state hard constraints:\n"
    "{picker_hard_constraints_text}\n"
    "\n"
    "Picker support candidate shelf actions:\n"
    "{picker_support_candidate_actions}\n"
    "\n"
    "Picker decision rules:\n"
    "- Picker support should be at the same shelf action id as the AGV support target, not at GOAL.\n"
    "- Do not enter a shelf cell unless the AGV is already there for support; otherwise stay near the shelf and wait.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, charging is forbidden.\n"
    "- Use suggested_steps as the default Steps choice.\n"
    "- For movement, Steps should usually be close to distance_steps.\n"
    "- For waiting at a support shelf, prefer a smaller Steps value.\n"
    "- For charging, Steps should reflect how much charge is still needed.\n"
    "- When multiple support shelf actions are valid and no stronger constraint applies, prefer the closest candidate by distance_steps.\n"
    "\n"
    "Candidate actions:\n"
    "{candidate_actions}\n"
    "\n"
    "Single-step decision:\n"
    "- Your job is to choose the next macro action for this Picker now.\n"
    "- Do not propose a strategy for future behavior.\n"
    "- Follow next_required_flow_state first, then candidate tags, then suggested_steps.\n"
    "- If an execution warning marks an action id as unreachable, selecting that same action id is invalid for this turn; choose another valid candidate.\n"
    "- If next_required_flow_state=move_to_support_load or move_to_support_unload, choose a tag=support_now candidate.\n"
    "- If support_needed_now=yes and battery is not critical, choose the matching support shelf action.\n"
    "- If support_needed_now=yes and battery is not critical, charging is forbidden.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, do not choose any charging action.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, do not choose any charging action.\n"
    "\n"
    "Forbidden outputs:\n"
    "- Do not write recommendations.\n"
    "- Do not rewrite flow rules.\n"
    "- Do not give example output.\n"
    "- Do not explain the system.\n"
    "- Do not output headings or bullet lists.\n"
    "- Do not output markdown.\n"
    "- Do not output action descriptions like SHELF@(x,y).\n"
    "- Do not output policy, report, or revised-rule text.\n"
    "\n"
    "Decision contract:\n"
    "- Choose exactly one action id from Candidate actions.\n"
    "- Choose exactly one Steps value.\n"
    "- Action must be the integer action id only.\n"
    "- Steps must be an integer only.\n"
    "- OUTPUT Should be of format: Action: selected action ID, Steps: suggested integer steps, Reason: very short justification. Only this much\n"
    "Action: selected action id\n"
    "Steps: integer between 1 and {max_hold_steps} suggested steps\n"
    "Reason: just one line very short reason why this action and why this many steps\n"
    "If unsure, still choose one valid integer action id and one Steps value.\n"
)


AGV_JSON_TEMPLATE = (
    "Main objective: Optimize calls to LLM and increase the total shelf delivery.\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load moves to requested shelf.\n"
    "- Picker and AGV meet at that shelf for loading.\n"
    "- If AGV reaches the requested shelf first and Picker support is still needed there, AGV should usually hold position and wait.\n"
    "- AGV carries shelf to GOAL.\n"
    "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <30, need_charging_soon if >=30 and <65, not_needed if >=65.\n"
    "\n"
    "You are planning only for one AGV.\n"
    "Choose exactly one action for the current AGV using the JSON sections in order: current_control_state, role_context, hard_constraints, decision_rules, and candidate_actions.\n"
    "AGV decision rules:\n"
    "- Use current_control_state.next_required_flow_state to decide the action family.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, charging is forbidden.\n"
    "- If picker_support_inbound_now=yes and picker_at_same_cell=no, Picker is on the way to this same shelf now. Do not leave this shelf now.\n"
    "- If battery is critical, charging may override the normal flow.\n"
    "- Use suggested_steps as the default steps choice.\n"
    "- For movement, steps should usually be close to distance_steps.\n"
    "- For waiting at a support shelf, prefer a smaller steps value.\n"
    "- For charging, steps should reflect how much charge is still needed.\n"
    "- When multiple requested shelf actions are valid and no stronger constraint applies, prefer the closest candidate by distance_steps.\n"
    "\n"
    "Shared planning context in JSON:\n"
    "{prompt_json}\n"
    "\n"
    "The planning context ends above.\n"
    "Single-step decision:\n"
    "- Your job is to choose the next macro action for this AGV now.\n"
    "- Do not propose a strategy for future behavior.\n"
    "- Follow current_control_state.next_required_flow_state first, then candidate_actions, then decision_rules.\n"
    "- If next_required_flow_state=move_requested_shelf_to_goal, choose a tag=deliver_now candidate.\n"
    "- If next_required_flow_state=return_delivered_shelf_to_empty_shelf, choose a tag=return_empty_shelf candidate.\n"
    "- If next_required_flow_state=wait_for_picker_at_requested_shelf, prefer the current shelf or tag=wait_for_picker_here.\n"
    "- If next_required_flow_state=wait_for_picker_at_unload_shelf, prefer the current shelf or tag=wait_for_picker_unload_here.\n"
    "- If next_required_flow_state=charge_now, choose a charging candidate.\n"
    "- If picker_support_inbound_now=yes and picker_at_same_cell=no, stay on this shelf until Picker arrives and the shelf interaction completes.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, do not choose any charging action.\n"
    "- If battery is 100, do not choose any charging action.\n"
    "\n"
    "Forbidden outputs:\n"
    "- Do not write recommendations.\n"
    "- Do not rewrite flow rules.\n"
    "- Do not give example output.\n"
    "- Do not explain the system.\n"
    "- Do not output headings, markdown, or report text.\n"
    "- Do not return action names, coordinates, or labels.\n"
    "Return exactly one JSON object and nothing else.\n"
    "Use only the keys reason, action, and steps.\n"
    "Choose the action that matches current_control_state.next_required_flow_state.\n"
    "Action must be an integer action id only, not a string description.\n"
    "Steps must be an integer only.\n"
    "If unsure, still choose one valid integer action id and one steps value.\n"
    "Required response shape:\n"
    "{{\"reason\":\"very short reason why this action and why this many steps\",\"action\":<selected action id>,\"steps\":<suggested steps>}}\n"
)


PICKER_JSON_TEMPLATE = (
    "Main objective: Optimize calls to LLM and increase the total shelf delivery.\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load moves to requested shelf.\n"
    "- Picker and AGV meet at that shelf for loading.\n"
    "- If AGV reaches the requested shelf first and Picker support is still needed there, AGV should usually hold position and wait while Picker moves there.\n"
    "- AGV carries shelf to GOAL.\n"
    "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <30, need_charging_soon if >=30 and <65, not_needed if >=65.\n"
    "\n"
    "You are planning only for one Picker.\n"
    "Choose exactly one action for the current Picker using the JSON sections in order: current_control_state, role_context, hard_constraints, decision_rules, and candidate_actions.\n"
    "Picker decision rules:\n"
    "- Use current_control_state.next_required_flow_state to decide the action family.\n"
    "- Picker support should be at the same shelf action id as the AGV support target, not at GOAL.\n"
    "- If AGV support is needed now and battery is not critical, do not choose charging.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, charging is forbidden.\n"
    "- Use suggested_steps as the default steps choice.\n"
    "- For movement, steps should usually be close to distance_steps.\n"
    "- For waiting at a support shelf, prefer a smaller steps value.\n"
    "- For charging, steps should reflect how much charge is still needed.\n"
    "- When multiple support shelf actions are valid and no stronger constraint applies, prefer the closest candidate by distance_steps.\n"
    "\n"
    "Shared planning context in JSON:\n"
    "{prompt_json}\n"
    "\n"
    "The planning context ends above.\n"
    "Single-step decision:\n"
    "- Your job is to choose the next macro action for this Picker now.\n"
    "- Do not propose a strategy for future behavior.\n"
    "- Follow current_control_state.next_required_flow_state first, then candidate_actions, then decision_rules.\n"
    "- If next_required_flow_state=move_to_support_load or move_to_support_unload, choose a tag=support_now candidate.\n"
    "- If current_control_state.support_needed_now is true and battery is not critical, choose the matching support shelf action.\n"
    "- If battery_need=not_needed and next_required_flow_state is not charge_now, do not choose any charging action.\n"
    "- Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.\n"
    "- If battery is 100, do not choose any charging action.\n"
    "\n"
    "Forbidden outputs:\n"
    "- Do not write recommendations.\n"
    "- Do not rewrite flow rules.\n"
    "- Do not give example output.\n"
    "- Do not explain the system.\n"
    "- Do not output headings, markdown, or report text.\n"
    "- Do not return action names, coordinates, or labels.\n"
    "Return exactly one JSON object and nothing else.\n"
    "Use only the keys reason, action, and steps.\n"
    "Choose the action that matches current_control_state.next_required_flow_state.\n"
    "Action must be an integer action id only, not a string description.\n"
    "Steps must be an integer only.\n"
    "If unsure, still choose one valid integer action id and one steps value.\n"
    "Required response shape:\n"
    "{{\"reason\":\"very short reason why this action and why this many steps\",\"action\":selected action id,\"steps\":suggested steps}}\n"
)


parser = ArgumentParser(
    description="Run objective-2 shared-context LLM experiments across models and warehouse scenarios",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=0, help="Base episode seed")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum steps per episode; use 0 for no step limit")
parser.add_argument("--max_inactivity_steps", type=int, default=200, help="Maximum inactive steps before the environment terminates")
parser.add_argument("--prompt_format", choices=["language", "json"], default="language", help="Prompt/output format")
parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama generate endpoint")
parser.add_argument("--request_timeout_s", type=int, default=60, help="HTTP timeout for Ollama generation")
parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
parser.add_argument("--num_predict", type=int, default=700, help="Generation token budget")
parser.add_argument("--max_action_hold_steps", type=int, default=10, help="Max number of steps an agent may keep the same LLM-decided action")
parser.add_argument("--max_picker_hold_steps", type=int, default=3, help="Max hold steps for picker agents")
parser.add_argument("--max_busy_steps_for_replan", type=int, default=0, help="Replan a busy shelf/travel agent after this many consecutive busy steps; use 0 to disable")
parser.add_argument("--disable_support_needed_soon", action="store_true", help="Disable proactive picker support and keep only waiting_for_picker_support_now")
parser.add_argument("--find_path_agent_aware_always", action="store_true", help="Force find_path to always consider other agents as obstacles")
parser.add_argument("--results_dir", type=str, default="results/obj2_shared_context_llm_experiments", help="Directory for logs and summaries")
parser.add_argument("--session_dir", type=str, default=None, help="Optional existing session directory path to resume into")
parser.add_argument("--resume_from_model", type=str, default=None, help="Model name (exact string) to resume from")
parser.add_argument("--resume_from_scenario", type=str, default=None, help="Scenario label to resume from when resuming a model")
parser.add_argument("--clean_resumed_model", action="store_true", help="If set, remove existing run results for the resumed model/scenario before restart")
parser.add_argument("--only_model", type=str, default=None, help="Run only this model name")
parser.add_argument("--only_scenario", type=str, default=None, help="Run only this scenario label")
parser.add_argument("--render", action="store_true", help="Render the environment during rollouts")


def safe_valid_action_masks(env) -> np.ndarray:
    try:
        masks = np.asarray(env.compute_valid_action_masks(), dtype=np.float64)
        if masks.shape != (env.num_agents, env.action_size):
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        return masks
    except Exception:
        return np.ones((env.num_agents, env.action_size), dtype=np.float64)


def support_needed_soon_disabled() -> bool:
    return bool(getattr(parser, "_parsed_disable_support_needed_soon", False))


def query_ollama_text(
    model: str,
    ollama_url: str,
    prompt: str,
    timeout_s: int,
    temperature: float,
    num_predict: int,
    prompt_format: str,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if prompt_format == "json":
        payload["format"] = "json"
    req = request.Request(
        ollama_url,
        data=str.encode(json.dumps(payload)),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def pull_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def remove_model(model: str) -> None:
    subprocess.run(["ollama", "rm", model], check=True)
    pass


def log(message: str, log_path: Path) -> None:
    stamped = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] {message}"
    print(stamped)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


def format_bullets(lines: List[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else "- none"


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def find_model_index(model_name: str) -> int:
    for idx, (model, _) in enumerate(MODELS):
        if model == model_name:
            return idx
    raise ValueError(f"Unknown model name: {model_name}")


def find_scenario_index(label: str) -> int:
    for idx, scenario in enumerate(SCENARIOS):
        if scenario.label == label:
            return idx
    raise ValueError(f"Unknown scenario label: {label}")


def remove_prior_aggregate_entries(aggregate_path: Path, model: str, scenario: str) -> None:
    if not aggregate_path.exists():
        return
    cleaned: List[str] = []
    with aggregate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except Exception:
                cleaned.append(line)
                continue
            if entry.get("model") == model and entry.get("scenario") == scenario:
                continue
            cleaned.append(line)
    with aggregate_path.open("w", encoding="utf-8") as handle:
        handle.writelines(cleaned)


def agent_step_snapshot(env) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, agent in enumerate(env.agents):
        target_id = int(agent.target)
        target_coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        rows.append(
            {
                "agent_id": idx,
                "agent_type": agent.type.name,
                "position": {"x": int(agent.x), "y": int(agent.y)},
                "busy": bool(agent.busy),
                "battery": float(agent.battery),
                "carrying": int(agent.carrying_shelf.id) if agent.carrying_shelf else None,
                "target_id": target_id if target_id > 0 else None,
                "target_type": classify_action(env, target_id) if target_id > 0 else "NONE",
                "target_position": (
                    {"x": int(target_coords[1]), "y": int(target_coords[0])}
                    if target_coords is not None
                    else None
                ),
                "micro_action": str(getattr(getattr(agent, "req_action", None), "name", None)),
            }
        )
    return rows


def action_distance_steps_for_agent(env, agent, action_id: int) -> int:
    if int(action_id) <= 0:
        return 0
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return 0
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    return len(path) if path else 0


def nearest_goal_for_agent(env, agent) -> Tuple[int, int]:
    best_goal_id = 0
    best_steps = None
    for goal_id in range(1, len(env.goals) + 1):
        steps = action_distance_steps_for_agent(env, agent, goal_id)
        if best_steps is None or (steps > 0 and steps < best_steps):
            best_steps = steps
            best_goal_id = int(goal_id)
    if best_steps is None:
        return 0, 0
    return best_goal_id, int(best_steps)


def nearest_charging_station_for_agent(env, agent) -> Tuple[int, int]:
    best_action_id = 0
    best_steps = None
    for action_id in charging_action_ids(env):
        steps = action_distance_steps_for_agent(env, agent, action_id)
        if best_steps is None or (steps > 0 and steps < best_steps):
            best_steps = steps
            best_action_id = int(action_id)
    if best_steps is None:
        return 0, 0
    return best_action_id, int(best_steps)


def suggested_steps_guidance(env, agent_idx: int, max_hold_steps: int) -> Dict[str, Any]:
    agent = env.agents[agent_idx]
    target_id = int(agent.target)
    target_distance = action_distance_steps_for_agent(env, agent, target_id) if target_id > 0 else None
    battery_need = battery_need_label(float(agent.battery))
    phase = classify_agent_phase(env, agent_idx)
    charging_target_id, charging_distance = nearest_charging_station_for_agent(env, agent)
    if battery_need == "critical":
        charging_steps_hint = max(1, min(int(max_hold_steps), 3))
    elif battery_need == "need_charging_soon":
        charging_steps_hint = max(1, min(int(max_hold_steps), 2))
    else:
        charging_steps_hint = 1
    movement_steps_hint = (
        max(1, min(int(max_hold_steps), int(target_distance)))
        if target_distance is not None and target_distance > 0
        else 1
    )
    wait_steps_hint = max(1, min(int(max_hold_steps), 2 if "waiting" in phase else 1))
    return {
        "phase": phase,
        "target_distance_steps": target_distance,
        "battery_need": battery_need,
        "movement_steps_hint": movement_steps_hint,
        "wait_steps_hint": wait_steps_hint,
        "charging_steps_hint": charging_steps_hint,
        "nearest_charging_action_id": charging_target_id,
        "nearest_charging_distance_steps": charging_distance,
        "policy": (
            "If moving to a target, use steps close to distance_steps. "
            "If waiting at a shelf, use a small-to-moderate value to hold position. "
            "If charging, use steps based on how much charge is still needed."
        ),
    }


def suggested_steps_for_action(env, agent_idx: int, action_id: int, max_hold_steps: int) -> int:
    agent = env.agents[agent_idx]
    max_steps = max(1, int(max_hold_steps))
    if int(action_id) <= 0:
        return 1
    action_type = classify_action(env, int(action_id))
    distance_steps = action_distance_steps_for_agent(env, agent, int(action_id))
    phase = classify_agent_phase(env, agent_idx)
    battery_need = battery_need_label(float(agent.battery))

    if action_type == "CHARGING":
        if battery_need == "critical":
            return max(1, min(max_steps, 3))
        if battery_need == "need_charging_soon":
            return max(1, min(max_steps, 2))
        return 1

    if "waiting" in phase and int(action_id) == int(agent.target) and int(agent.target) > 0:
        return max(1, min(max_steps, 2))

    if action_type in {"SHELF", "GOAL"}:
        return max(1, min(max_steps, int(distance_steps)))

    return 1


def nearest_empty_shelf_slots_for_agent(env, agent, limit: int = 5) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for action_id in empty_shelf_action_ids(env):
        coords = env.action_id_to_coords_map.get(int(action_id))
        if coords is None:
            continue
        steps = action_distance_steps_for_agent(env, agent, int(action_id))
        rows.append(
            {
                "action_id": int(action_id),
                "x": int(coords[1]),
                "y": int(coords[0]),
                "distance_steps": int(steps),
            }
        )
    rows.sort(key=lambda row: (row["distance_steps"], row["action_id"]))
    return rows[: max(0, int(limit))]


def carrying_status_label(env, agent) -> str:
    if agent.carrying_shelf is None:
        return "not_carrying"
    requested_shelf_ids = {int(shelf.id) for shelf in env.request_queue}
    if int(agent.carrying_shelf.id) in requested_shelf_ids:
        return "carrying_requested_shelf"
    return "carrying_delivered_shelf"


def current_target_type(env, agent) -> str:
    target_id = int(agent.target)
    return classify_action(env, target_id) if target_id > 0 else "NOOP"


def classify_agent_phase(env, agent_idx: int) -> str:
    agent = env.agents[agent_idx]
    target_id = int(agent.target)
    target_type = current_target_type(env, agent)
    target_coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
    at_target = bool(target_coords is not None and (int(agent.x), int(agent.y)) == (int(target_coords[1]), int(target_coords[0])))
    charging_cells = {(int(station.x), int(station.y)) for station in env.charging_stations}

    if (int(agent.x), int(agent.y)) in charging_cells:
        return "charging_now"
    if target_type == "CHARGING":
        return "moving_to_charging"

    if agent.type.name == "AGV":
        carrying_status = carrying_status_label(env, agent)
        if target_id <= 0:
            return "idle_no_target"
        if carrying_status == "carrying_requested_shelf" and target_type == "GOAL":
            return "carrying_requested_shelf_to_goal"
        if carrying_status == "carrying_delivered_shelf" and target_type == "SHELF":
            if at_target:
                return "moving_to_empty_shelf_for_unload"
            return "carrying_delivered_shelf_to_empty_shelf"
        if target_type == "SHELF" and carrying_status == "not_carrying":
            if at_target:
                return "waiting_for_picker_at_requested_shelf"
            return "moving_to_requested_shelf"
        return "idle_no_target"

    if target_id <= 0:
        return "idle_no_support_task"
    if target_type == "SHELF":
        if at_target:
            return "waiting_at_support_shelf"
        support_candidates = picker_support_candidate_action_lines(env, agent_idx, np.ones((env.num_agents, env.action_size)))
        if any(f"{target_id}:SHELF" in line and "unload" in line for line in support_candidates):
            return "moving_to_support_unload_shelf"
        return "moving_to_support_requested_shelf"
    return "idle_no_support_task"


def picker_at_same_cell_for_agv(env, agv_idx: int) -> bool:
    agv = env.agents[agv_idx]
    return any(
        agent.type.name == "PICKER" and int(agent.x) == int(agv.x) and int(agent.y) == int(agv.y)
        for agent in env.agents
    )


def picker_at_action_cell(env, action_id: int) -> bool:
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return False
    target_x, target_y = int(coords[1]), int(coords[0])
    return any(
        agent.type.name == "PICKER" and int(agent.x) == target_x and int(agent.y) == target_y
        for agent in env.agents
    )


def charging_cells_for_env(env) -> set[tuple[int, int]]:
    return {(int(station.x), int(station.y)) for station in getattr(env, "charging_stations", [])}


def agent_is_interruptible_on_charger(env, agent) -> bool:
    return bool(agent.type.name == "PICKER" and (int(agent.x), int(agent.y)) in charging_cells_for_env(env))


def committed_unload_action_id_for_agv(
    env,
    agv_idx: int,
    persisted_actions: List[int] | None = None,
) -> int:
    agv = env.agents[agv_idx]
    if carrying_status_label(env, agv) != "carrying_delivered_shelf":
        return 0

    live_target_id = int(agv.target)
    if live_target_id > 0 and classify_action(env, live_target_id) == "SHELF":
        return int(live_target_id)

    persisted_target_id = 0
    if persisted_actions is not None and agv_idx < len(persisted_actions):
        persisted_target_id = int(persisted_actions[agv_idx])
    if persisted_target_id > 0 and classify_action(env, persisted_target_id) == "SHELF":
        return int(persisted_target_id)

    return 0


def distance_steps_from_position_to_action(env, agent, start_x: int, start_y: int, action_id: int) -> int:
    if int(action_id) <= 0:
        return 0
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return 9999
    path = env.find_path((int(start_y), int(start_x)), coords, agent, care_for_agents=False)
    if path is None:
        return 9999
    return max(0, len(path))


def nearest_charging_steps_from_action(env, agent, action_id: int) -> int:
    if int(action_id) <= 0:
        return 9999
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return 9999
    start_x, start_y = int(coords[1]), int(coords[0])
    best_steps = 9999
    for charge_action_id in charging_action_ids(env):
        steps = distance_steps_from_position_to_action(env, agent, start_x, start_y, int(charge_action_id))
        best_steps = min(best_steps, int(steps))
    return int(best_steps)


def picker_effectively_in_place_for_unload_target(
    env,
    unload_action_id: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> bool:
    if int(unload_action_id) <= 0:
        return False
    for picker_idx in range(env.num_agvs, env.num_agents):
        picker = env.agents[picker_idx]
        flags = picker_support_flags(env, picker_idx, persisted_actions, hold_steps_remaining)
        targeted_support_action = flags.get("support_target_action_id")
        if targeted_support_action is None or int(targeted_support_action) != int(unload_action_id):
            continue
        persisted_target = 0
        if persisted_actions is not None and picker_idx < len(persisted_actions):
            persisted_target = int(persisted_actions[picker_idx])
        has_same_target = int(picker.target) == int(unload_action_id) or persisted_target == int(unload_action_id)
        if bool(flags.get("at_support_target")):
            return True
        if has_same_target and action_distance_steps_for_agent(env, picker, int(unload_action_id)) <= 2:
            return True
    return False


def agv_task_ready_charge_policy(
    env,
    agv_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any]:
    agv = env.agents[agv_idx]
    carrying_status = carrying_status_label(env, agv)
    on_charger = (int(agv.x), int(agv.y)) in charging_cells_for_env(env) and current_target_type(env, agv) == "CHARGING"
    if not on_charger:
        return {
            "active": False,
            "required_battery": 0,
            "picker_support_actionable": True,
            "carrying_status": carrying_status,
            "release_reason": "not_on_charger",
        }

    required_battery = 80
    picker_support_actionable = True
    release_reason = "minimum_floor_not_met"
    if carrying_status == "carrying_delivered_shelf":
        unload_action_id = committed_unload_action_id_for_agv(env, agv_idx, persisted_actions)
        unload_distance = action_distance_steps_for_agent(env, agv, unload_action_id) if unload_action_id > 0 else 9999
        return_to_charge_distance = nearest_charging_steps_from_action(env, agv, unload_action_id)
        required_battery = max(80, int(unload_distance) + 2 + int(return_to_charge_distance) + 10)
        picker_support_actionable = picker_effectively_in_place_for_unload_target(
            env,
            unload_action_id,
            persisted_actions,
            hold_steps_remaining,
        )
        release_reason = "unload_support_not_actionable_yet"
    elif carrying_status == "carrying_requested_shelf":
        goal_action_id, goal_distance = nearest_goal_for_agent(env, agv)
        return_to_charge_distance = nearest_charging_steps_from_action(env, agv, goal_action_id)
        required_battery = max(80, int(goal_distance) + int(return_to_charge_distance) + 10)
        release_reason = "goal_trip_not_energy_safe_yet"
    else:
        release_reason = "minimum_floor_not_met"

    battery_now = float(agv.battery)
    active = battery_now < float(required_battery) or not picker_support_actionable
    return {
        "active": bool(active),
        "required_battery": int(required_battery),
        "picker_support_actionable": bool(picker_support_actionable),
        "carrying_status": carrying_status,
        "release_reason": str(release_reason),
    }


def picker_task_ready_charge_policy(
    env,
    picker_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any]:
    del persisted_actions, hold_steps_remaining
    picker = env.agents[picker_idx]
    on_charger = (int(picker.x), int(picker.y)) in charging_cells_for_env(env) and current_target_type(env, picker) == "CHARGING"
    required_battery = 80
    if not on_charger:
        return {
            "active": False,
            "required_battery": int(required_battery),
            "release_reason": "not_on_charger",
        }
    battery_now = float(picker.battery)
    return {
        "active": bool(battery_now < float(required_battery)),
        "required_battery": int(required_battery),
        "release_reason": "minimum_floor_not_met",
    }


def effective_support_row_for_agv(
    env,
    agv_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any] | None:
    for row in effective_agv_support_rows(env, persisted_actions, hold_steps_remaining):
        if int(row["agv_idx"]) == int(agv_idx):
            return row
    return None


def support_type_for_row(row: Dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    if bool(row.get("is_post_delivery_unload")) or str(row.get("state_source")) == "post_delivery_helper":
        return "unload"
    support_need = str(row.get("support_need", "none"))
    if support_need == "waiting_for_picker_support_now":
        carrying = row.get("carrying")
        if carrying is not None:
            return "unload"
    if support_need in {"waiting_for_picker_support_now", "moving_to_shelf_will_need_picker_for_load"}:
        return "load"
    if support_need == "moving_to_shelf_will_need_picker_for_unload":
        return "unload"
    return "none"


def effective_picker_support_rows(
    env,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[int, int, str]] = set()

    for row in effective_agv_support_rows(env, persisted_actions, hold_steps_remaining):
        normalized = {
            **row,
            "is_post_delivery_unload": False,
            "shared_unload_action_id": None,
        }
        key = (
            int(normalized["agv_idx"]),
            int(normalized["effective_target_action_id"]),
            str(normalized["support_need"]),
        )
        seen_keys.add(key)
        rows.append(normalized)

    for agv_idx, agv in enumerate(env.agents[: env.num_agvs]):
        if carrying_status_label(env, agv) != "carrying_delivered_shelf":
            continue
        action_id = committed_unload_action_id_for_agv(env, agv_idx, persisted_actions)
        if action_id <= 0:
            continue
        coords = env.action_id_to_coords_map.get(action_id)
        if coords is None:
            continue
        picker_here = picker_at_action_cell(env, action_id)
        if picker_here:
            continue
        support_need = "moving_to_shelf_will_need_picker_for_unload"
        if support_needed_soon_disabled():
            support_need = "waiting_for_picker_support_now"
        key = (int(agv_idx), action_id, support_need)
        if key in seen_keys:
            continue
        remaining_hold_steps = 0
        if hold_steps_remaining is not None and agv_idx < len(hold_steps_remaining):
            remaining_hold_steps = int(hold_steps_remaining[agv_idx])
        rows.append(
            {
                "agv_idx": int(agv_idx),
                "effective_target_action_id": action_id,
                "target_x": int(coords[1]),
                "target_y": int(coords[0]),
                "agv_x": int(agv.x),
                "agv_y": int(agv.y),
                "carrying": int(agv.carrying_shelf.id) if agv.carrying_shelf is not None else None,
                "support_need": support_need,
                "state_source": "post_delivery_helper",
                "remaining_hold_steps": int(remaining_hold_steps),
                "is_post_delivery_unload": True,
                "shared_unload_action_id": action_id,
            }
        )
        seen_keys.add(key)

    rows.sort(
        key=lambda row: (
            0 if str(row["support_need"]) == "waiting_for_picker_support_now" else 1,
            int(row["agv_idx"]),
            int(row["effective_target_action_id"]),
        )
    )
    return rows


def support_flags_for_agv(
    env,
    agv_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any]:
    row = effective_support_row_for_agv(env, agv_idx, persisted_actions, hold_steps_remaining)
    support_need = str(row["support_need"]) if row is not None else "none"
    support_type = support_type_for_row(row)
    support_target_action_id = int(row["effective_target_action_id"]) if row is not None else None
    picker_at_same_cell = picker_at_same_cell_for_agv(env, agv_idx)
    picker_support_inbound_now = False
    picker_support_distance_steps = None
    if support_target_action_id is not None and not picker_at_same_cell:
        best_distance_steps = None
        for picker_idx in range(env.num_agvs, env.num_agents):
            picker_flags = picker_support_flags(env, picker_idx, persisted_actions, hold_steps_remaining)
            if int(picker_flags.get("support_target_action_id") or 0) != int(support_target_action_id):
                continue
            distance_steps = action_distance_steps_for_agent(env, env.agents[picker_idx], support_target_action_id)
            if best_distance_steps is None or distance_steps < best_distance_steps:
                best_distance_steps = int(distance_steps)
        if best_distance_steps is not None:
            picker_support_inbound_now = True
            picker_support_distance_steps = int(best_distance_steps)
    return {
        "needs_picker_for_load": support_need in {
            "waiting_for_picker_support_now",
            "moving_to_shelf_will_need_picker_for_load",
        } and support_type == "load",
        "needs_picker_for_unload": support_type == "unload",
        "waiting_for_picker_support": support_need == "waiting_for_picker_support_now",
        "picker_at_same_cell": picker_at_same_cell,
        "picker_support_inbound_now": bool(picker_support_inbound_now),
        "picker_support_distance_steps": picker_support_distance_steps,
        "support_target_action_id": support_target_action_id,
        "support_target_position": (
            {"x": int(row["target_x"]), "y": int(row["target_y"])} if row is not None else None
        ),
        "support_type": support_type,
        "support_need": support_need,
    }


def picker_support_flags(
    env,
    picker_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any]:
    picker = env.agents[picker_idx]
    rows = effective_picker_support_rows(env, persisted_actions, hold_steps_remaining)
    chosen = None
    if rows:
        def sort_key(row: Dict[str, Any]) -> Tuple[int, int, int]:
            support_need = str(row["support_need"])
            pri = 0 if support_need == "waiting_for_picker_support_now" else 1
            dist = action_distance_steps_for_agent(env, picker, int(row["effective_target_action_id"]))
            return (pri, dist, int(row["effective_target_action_id"]))
        chosen = sorted(rows, key=sort_key)[0]
    support_target_action_id = int(chosen["effective_target_action_id"]) if chosen is not None else None
    at_support_target = (
        support_target_action_id is not None
        and action_distance_steps_for_agent(env, picker, support_target_action_id) == 0
    )
    support_need = str(chosen["support_need"]) if chosen is not None else "none"
    support_type = support_type_for_row(chosen)
    return {
        "agv_waiting_for_picker_now": any(str(row["support_need"]) == "waiting_for_picker_support_now" for row in rows),
        "support_target_action_id": support_target_action_id,
        "support_target_position": (
            {"x": int(chosen["target_x"]), "y": int(chosen["target_y"])} if chosen is not None else None
        ),
        "support_type": support_type,
        "at_support_target": at_support_target,
        "support_need": support_need,
    }


def next_required_flow_state(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> str:
    agent = env.agents[agent_idx]
    if agent.type.name == "AGV":
        carrying_status = carrying_status_label(env, agent)
        flags = support_flags_for_agv(env, agent_idx, persisted_actions, hold_steps_remaining)
        if carrying_status == "carrying_delivered_shelf":
            if bool(flags["waiting_for_picker_support"]) and str(flags["support_type"]) == "unload":
                return "wait_for_picker_at_unload_shelf"
        battery_need = battery_need_label(float(agent.battery))
        if battery_need == "critical":
            return "charge_now"
        if carrying_status == "carrying_requested_shelf":
            return "move_requested_shelf_to_goal"
        if carrying_status == "carrying_delivered_shelf":
            return "return_delivered_shelf_to_empty_shelf"
        if bool(flags["waiting_for_picker_support"]):
            return "wait_for_picker_at_requested_shelf"
        return "move_to_requested_shelf"

    battery_need = battery_need_label(float(agent.battery))
    flags = picker_support_flags(env, agent_idx, persisted_actions, hold_steps_remaining)
    if battery_need == "critical":
        return "charge_now"
    if flags["support_target_action_id"] is None:
        return "idle"
    if bool(flags["at_support_target"]) and bool(flags["agv_waiting_for_picker_now"]):
        return "hold_at_support_shelf"
    if str(flags["support_type"]) == "unload":
        return "move_to_support_unload"
    return "move_to_support_load"


def control_state_for_agent(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[agent_idx]
    target_id = int(agent.target)
    at_target = target_id > 0 and action_distance_steps_for_agent(env, agent, target_id) == 0
    battery_need = battery_need_label(float(agent.battery))
    next_state = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)

    if agent.type.name == "AGV":
        flags = support_flags_for_agv(env, agent_idx, persisted_actions, hold_steps_remaining)
        carrying_status = carrying_status_label(env, agent)
        nearest_goal_id, nearest_goal_steps = nearest_goal_for_agent(env, agent)
        original_load_shelf_action_id = None
        if target_id > 0 and classify_action(env, target_id) == "SHELF":
            original_load_shelf_action_id = int(target_id)
        elif persisted_actions is not None and agent_idx < len(persisted_actions):
            persisted_action_id = int(persisted_actions[agent_idx])
            if persisted_action_id > 0 and classify_action(env, persisted_action_id) == "SHELF":
                original_load_shelf_action_id = int(persisted_action_id)
        payload = {
            "carrying_status": carrying_status,
            "at_target": bool(at_target),
            "current_target_action": int(target_id) if target_id > 0 else None,
            "current_target_type": current_target_type(env, agent),
            "waiting_for_picker_support_now": bool(flags["waiting_for_picker_support"]),
            "picker_at_same_cell": bool(flags["picker_at_same_cell"]),
            "picker_support_inbound_now": bool(flags["picker_support_inbound_now"]),
            "picker_support_distance_steps": int(flags["picker_support_distance_steps"]) if flags["picker_support_distance_steps"] is not None else None,
            "needs_picker_for_load": bool(flags["needs_picker_for_load"]),
            "needs_picker_for_unload": bool(flags["needs_picker_for_unload"]),
            "next_required_flow_state": next_state,
            "load_completed_and_goal_required_now": carrying_status == "carrying_requested_shelf",
            "original_load_shelf_action_id": original_load_shelf_action_id,
            "recommended_goal_action_id": nearest_goal_id if carrying_status == "carrying_requested_shelf" else None,
            "recommended_goal_distance_steps": nearest_goal_steps if carrying_status == "carrying_requested_shelf" else None,
        }
        lines = [
            f"- carrying_status: {payload['carrying_status']}",
            f"- at_target: {'yes' if payload['at_target'] else 'no'}",
            f"- current_target_action: {payload['current_target_action'] if payload['current_target_action'] is not None else 'none'}",
            f"- current_target_type: {payload['current_target_type']}",
            f"- waiting_for_picker_support_now: {'yes' if payload['waiting_for_picker_support_now'] else 'no'}",
            f"- picker_at_same_cell: {'yes' if payload['picker_at_same_cell'] else 'no'}",
            f"- picker_support_inbound_now: {'yes' if payload['picker_support_inbound_now'] else 'no'}",
            f"- picker_support_distance_steps: {payload['picker_support_distance_steps'] if payload['picker_support_distance_steps'] is not None else 'none'}",
            f"- needs_picker_for_load: {'yes' if payload['needs_picker_for_load'] else 'no'}",
            f"- needs_picker_for_unload: {'yes' if payload['needs_picker_for_unload'] else 'no'}",
            f"- next_required_flow_state: {payload['next_required_flow_state']}",
        ]
        if payload["load_completed_and_goal_required_now"]:
            lines.extend(
                [
                    f"- load_completed_and_goal_required_now: yes",
                    f"- original_load_shelf_action_id: {payload['original_load_shelf_action_id'] if payload['original_load_shelf_action_id'] is not None else 'none'}",
                    f"- recommended_goal_action_id: {payload['recommended_goal_action_id']}",
                    f"- recommended_goal_distance_steps: {payload['recommended_goal_distance_steps']}",
                ]
            )
        return "\n".join(lines), payload

    flags = picker_support_flags(env, agent_idx, persisted_actions, hold_steps_remaining)
    payload = {
        "support_needed_now": bool(flags["agv_waiting_for_picker_now"]),
        "support_target_action": int(flags["support_target_action_id"]) if flags["support_target_action_id"] is not None else None,
        "support_type": str(flags["support_type"]),
        "at_support_target": bool(flags["at_support_target"]),
        "battery_need": battery_need,
        "agv_waiting_for_picker_now": bool(flags["agv_waiting_for_picker_now"]),
        "next_required_flow_state": next_state,
    }
    lines = [
        f"- support_needed_now: {'yes' if payload['support_needed_now'] else 'no'}",
        f"- support_target_action: {payload['support_target_action'] if payload['support_target_action'] is not None else 'none'}",
        f"- support_type: {payload['support_type']}",
        f"- at_support_target: {'yes' if payload['at_support_target'] else 'no'}",
        f"- battery_need: {payload['battery_need']}",
        f"- agv_waiting_for_picker_now: {'yes' if payload['agv_waiting_for_picker_now'] else 'no'}",
        f"- next_required_flow_state: {payload['next_required_flow_state']}",
    ]
    return "\n".join(lines), payload


def render_self_state_with_target_distance(env, agent) -> str:
    base = render_self_state(env, agent)
    target_id = int(agent.target)
    target_type = current_target_type(env, agent)
    phase_label = classify_agent_phase(env, env.agents.index(agent))
    if target_id <= 0:
        return base + f", target_type={target_type}, target_distance_steps=none, phase={phase_label}"
    return (
        base
        + f", target_type={target_type}, target_distance_steps={action_distance_steps_for_agent(env, agent, target_id)}"
        + f", phase={phase_label}"
    )


def requested_shelves_with_agent_distance(env, agent, limit: int = 5) -> List[str]:
    rows_with_distance: List[Tuple[int, str]] = []
    for line in get_requested_shelves(env):
        match = re.search(r"action id (\d+)", line)
        action_id = int(match.group(1)) if match else 0
        distance_steps = action_distance_steps_for_agent(env, agent, action_id) if action_id > 0 else 0
        rendered = f"{line[:-1] if line.endswith('.') else line} Distance from current agent = {distance_steps} steps."
        rows_with_distance.append((distance_steps, rendered))
    rows_with_distance.sort(key=lambda item: (item[0], item[1]))
    return [row for _, row in rows_with_distance[: max(0, int(limit))]]


def effective_agv_support_rows(
    env,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> List[Dict[str, Any]]:
    requested_action_ids = set(get_requested_action_ids(env))
    picker_positions = {(int(agent.x), int(agent.y)) for agent in env.agents if agent.type.name == "PICKER"}
    rows: List[Dict[str, Any]] = []

    for idx, agv in enumerate(env.agents[: env.num_agvs]):
        live_target_id = int(agv.target)
        live_target_is_shelf = live_target_id > 0 and classify_action(env, live_target_id) == "SHELF"

        persisted_target_id = 0
        if persisted_actions is not None and idx < len(persisted_actions):
            persisted_target_id = int(persisted_actions[idx])
        persisted_target_is_shelf = persisted_target_id > 0 and classify_action(env, persisted_target_id) == "SHELF"

        effective_target_action_id = 0
        state_source = "none"
        if live_target_is_shelf:
            effective_target_action_id = live_target_id
            state_source = "live_env"
        elif persisted_target_is_shelf:
            effective_target_action_id = persisted_target_id
            state_source = "persisted_action"

        if effective_target_action_id <= 0:
            continue

        coords = env.action_id_to_coords_map.get(effective_target_action_id)
        if coords is None:
            continue

        carrying = agv.carrying_shelf is not None
        at_target = (int(agv.x), int(agv.y)) == (int(coords[1]), int(coords[0]))
        picker_here = (int(coords[1]), int(coords[0])) in picker_positions

        support_need = None
        if at_target and not carrying and not picker_here and effective_target_action_id in requested_action_ids:
            support_need = "waiting_for_picker_support_now"
        elif not at_target and not carrying and effective_target_action_id in requested_action_ids:
            support_need = "moving_to_shelf_will_need_picker_for_load"
        elif at_target and carrying and not picker_here and effective_target_action_id not in requested_action_ids:
            support_need = "waiting_for_picker_support_now"
        elif not at_target and carrying and effective_target_action_id not in requested_action_ids:
            support_need = "moving_to_shelf_will_need_picker_for_unload"

        if support_need is None:
            continue
        if support_needed_soon_disabled() and str(support_need) == "moving_to_shelf_will_need_picker_for_load":
            continue

        remaining_hold_steps = 0
        if hold_steps_remaining is not None and idx < len(hold_steps_remaining):
            remaining_hold_steps = int(hold_steps_remaining[idx])

        rows.append(
            {
                "agv_idx": int(idx),
                "effective_target_action_id": int(effective_target_action_id),
                "target_x": int(coords[1]),
                "target_y": int(coords[0]),
                "agv_x": int(agv.x),
                "agv_y": int(agv.y),
                "carrying": int(agv.carrying_shelf.id) if carrying else None,
                "support_need": str(support_need),
                "state_source": str(state_source),
                "remaining_hold_steps": int(remaining_hold_steps),
            }
        )

    return rows


def picker_support_candidate_action_lines(
    env,
    picker_idx: int,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> List[str]:
    picker = env.agents[picker_idx]
    valid_action_ids = set(candidate_ids_for_prompt(env, picker_idx, valid_masks, 0))
    rows: List[str] = []
    for support_row in effective_picker_support_rows(env, persisted_actions, hold_steps_remaining):
        agv_idx = int(support_row["agv_idx"])
        action_id = int(support_row["effective_target_action_id"])
        target_x = int(support_row["target_x"])
        target_y = int(support_row["target_y"])
        support_need = str(support_row["support_need"])
        if action_id not in valid_action_ids:
            continue
        distance_steps = action_distance_steps_for_agent(env, picker, action_id)
        rows.append(
            f"{action_id}:SHELF@({target_x},{target_y}) requested_by=agent_{agv_idx} "
            f"support_need={support_need} distance_steps={distance_steps} "
            f"state_source={support_row['state_source']}"
        )
    return rows if rows else ["none"]


def agv_flow_rules_text_from_payload(role_context_payload: Dict[str, Any]) -> str:
    agv_state = role_context_payload.get("agv_state", {})
    flags = role_context_payload.get("support_flags", {})
    lines = [
        "- If at a requested shelf, not carrying, and picker not here: hold this shelf and wait for picker support.",
        "- If at a requested shelf and picker is here: load happens here; after load, do not choose the same shelf again; move to GOAL next.",
        "- If carrying a requested shelf: choose GOAL, not shelf.",
        "- If carrying a delivered shelf: GOAL is forbidden; choose an EMPTY shelf return action.",
        "- If battery is critical: charging may override the normal flow.",
    ]
    if bool(agv_state.get("load_completed_and_goal_required_now", False)):
        lines.append("- If load already happened and you are carrying the requested shelf, do not choose the load shelf again; choose GOAL.")
    if bool(agv_state.get("empty_shelf_return_required", False)):
        lines.append("- Empty shelf return is active now; choose one EMPTY shelf action from the dedicated return block.")
    if bool(flags.get("waiting_for_picker_support", False)):
        lines.append("- Picker support is already pending at this shelf; do not abandon it unless battery is critical.")
    return "\n".join(lines)


def empty_shelf_return_task_block(role_context_payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    empty_ctx = role_context_payload.get("empty_shelf_return_context", {})
    recommended = list(empty_ctx.get("recommended_empty_shelf_actions", []))
    required = bool(empty_ctx.get("empty_shelf_return_required", False))
    if not required:
        return "", {"visible": False}
    lines = [
        "Empty shelf return task:",
        "- Your carried shelf is already delivered.",
        "- GOAL is forbidden now.",
        "- Drop this shelf at one of these EMPTY shelf actions:",
    ]
    for row in recommended:
        lines.append(
            f"- action_id={int(row['action_id'])} SHELF@({int(row['x'])},{int(row['y'])}) "
            f"distance_steps={int(row['distance_steps'])} suggested_steps={int(row['suggested_steps'])}"
        )
    return "\n".join(lines) + "\n\n", {"visible": True, "recommended_empty_shelf_actions": recommended}


def post_load_helper_for_agent(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[agent_idx]
    if agent.type.name == "AGV":
        if carrying_status_label(env, agent) != "carrying_requested_shelf":
            return "", {"post_load_active": False}
        nearest_goal_id, nearest_goal_steps = nearest_goal_for_agent(env, agent)
        original_load_shelf_action_id = None
        target_id = int(agent.target)
        if target_id > 0 and classify_action(env, target_id) == "SHELF":
            original_load_shelf_action_id = int(target_id)
        elif persisted_actions is not None and agent_idx < len(persisted_actions):
            persisted_action_id = int(persisted_actions[agent_idx])
            if persisted_action_id > 0 and classify_action(env, persisted_action_id) == "SHELF":
                original_load_shelf_action_id = int(persisted_action_id)
        carried_shelf_id = int(agent.carrying_shelf.id) if agent.carrying_shelf is not None else None
        lines = [
            "Post-load helper:",
            "- Load already happened at the shelf.",
            f"- You are now carrying the requested shelf {carried_shelf_id}.",
            "- Do not choose the same shelf again.",
            "- The original shelf interaction is complete.",
            "- Your current task is to move this shelf to GOAL.",
            "- Choose a GOAL action next unless battery is critical.",
            "- The original load shelf is not the correct next action.",
            "- This phase is complete only when the carried requested shelf is delivered to GOAL.",
        ]
        if original_load_shelf_action_id is not None:
            lines.append(f"- Original load shelf action_id={original_load_shelf_action_id}.")
        if nearest_goal_id > 0:
            lines.append(f"- Recommended GOAL action now: {nearest_goal_id} distance_steps={nearest_goal_steps}.")
        payload = {
            "post_load_active": True,
            "goal_required_now": True,
            "carried_shelf_id": carried_shelf_id,
            "original_load_shelf_action_id": original_load_shelf_action_id,
            "current_post_load_goal_action_id": nearest_goal_id if nearest_goal_id > 0 else None,
            "recommended_goal_action_id": nearest_goal_id if nearest_goal_id > 0 else None,
            "recommended_goal_distance_steps": nearest_goal_steps if nearest_goal_id > 0 else None,
        }
        return "\n".join(lines) + "\n\n", payload

    candidate_agvs = [
        idx
        for idx, agv in enumerate(env.agents[: env.num_agvs])
        if carrying_status_label(env, agv) == "carrying_requested_shelf"
    ]
    if not candidate_agvs:
        return "", {"post_load_active": False}
    picker = env.agents[agent_idx]
    chosen_idx = min(
        candidate_agvs,
        key=lambda idx: (
            abs(int(env.agents[idx].x) - int(picker.x)) + abs(int(env.agents[idx].y) - int(picker.y)),
            idx,
        ),
    )
    agv = env.agents[chosen_idx]
    lines = [
        "Post-load helper:",
        f"- AGV agent_{chosen_idx} has already loaded the requested shelf.",
        "- The load support phase at the original shelf is complete.",
        "- Do not keep selecting the old load shelf unless another AGV needs support there.",
        "- The AGV should now move to GOAL.",
        "- Your support priority should shift away from that completed load shelf.",
    ]
    payload = {
        "post_load_active": True,
        "agv_id_loaded_requested_shelf": int(chosen_idx),
        "shift_support_away_from_completed_load_shelf": True,
    }
    return "\n".join(lines) + "\n\n", payload


def post_delivery_helper_for_agent(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[agent_idx]
    if agent.type.name == "AGV":
        if carrying_status_label(env, agent) != "carrying_delivered_shelf":
            return "", {"post_delivery_active": False}
        empty_slots = nearest_empty_shelf_slots_for_agent(env, agent, limit=1)
        recommended = empty_slots[0] if empty_slots else None
        carried_shelf_id = int(agent.carrying_shelf.id) if agent.carrying_shelf is not None else None
        original_request_action_id = int(persisted_actions[agent_idx]) if persisted_actions is not None and agent_idx < len(persisted_actions) else None
        lines = [
            "Post-delivery helper:",
            f"- You already delivered shelf {carried_shelf_id} to GOAL.",
            "- GOAL is finished for this shelf.",
            f"- You are still carrying that shelf.",
            "- GOAL is forbidden now.",
            "- This is now the post-delivery unload support phase.",
            "- Return the shelf to an EMPTY shelf location.",
            "- Do not return to the original request shelf just because the load started there.",
            "- The original request shelf is no longer the task after delivery.",
            (
                f"- Original request shelf: action_id={original_request_action_id} (used for the earlier load interaction)."
                if original_request_action_id is not None and original_request_action_id > 0
                else "- Original request shelf: earlier load interaction shelf."
            ),
        ]
        payload = {
            "post_delivery_active": True,
            "carried_shelf_id": carried_shelf_id,
            "agv_next_task": "return_to_empty_shelf",
            "goal_forbidden_now": True,
            "charging_override_only_if_battery_critical": True,
            "picker_should_support_same_action_id": None,
            "recommended_empty_shelf_action": None,
            "picker_support_required_for_unload": True,
            "current_post_delivery_unload_action_id": None,
            "current_post_delivery_unload_position": None,
        }
        if recommended is not None:
            lines.extend(
                [
                    f"- Current unload shelf: action_id={int(recommended['action_id'])}:SHELF@({int(recommended['x'])},{int(recommended['y'])}) for returning the delivered shelf.",
                    f"- Recommended EMPTY shelf action now: {int(recommended['action_id'])}:SHELF@({int(recommended['x'])},{int(recommended['y'])}) distance_steps={int(recommended['distance_steps'])}.",
                    "- Your current task is the EMPTY shelf return location shown below.",
                    "- Choose that EMPTY shelf action unless battery is critical.",
                    "- Picker support is needed at that same shelf for unload.",
                    "- This post-delivery phase is complete only after unload at that EMPTY shelf location.",
                    "- This phase is complete only when carrying=None after unload.",
                ]
            )
            payload["recommended_empty_shelf_action_id"] = int(recommended["action_id"])
            payload["recommended_empty_shelf_position"] = {
                "x": int(recommended["x"]),
                "y": int(recommended["y"]),
            }
            payload["recommended_empty_shelf_distance_steps"] = int(recommended["distance_steps"])
            payload["picker_should_support_same_action_id"] = int(recommended["action_id"])
            payload["recommended_empty_shelf_action"] = {
                "action_id": int(recommended["action_id"]),
                "position": {"x": int(recommended["x"]), "y": int(recommended["y"])},
                "distance_steps": int(recommended["distance_steps"]),
            }
            payload["current_post_delivery_unload_action_id"] = int(recommended["action_id"])
            payload["current_post_delivery_unload_position"] = {
                "x": int(recommended["x"]),
                "y": int(recommended["y"]),
            }
        return "\n".join(lines) + "\n\n", payload

    support_rows = effective_agv_support_rows(env, persisted_actions, hold_steps_remaining)
    unload_rows = [
        row
        for row in effective_picker_support_rows(env, persisted_actions, hold_steps_remaining)
        if str(row["support_need"]) == "moving_to_shelf_will_need_picker_for_unload"
    ]
    if not unload_rows:
        return "", {"post_delivery_active": False}
    picker = env.agents[agent_idx]
    unload_rows.sort(
        key=lambda row: (
            action_distance_steps_for_agent(env, picker, int(row["effective_target_action_id"])),
            int(row["agv_idx"]),
        )
    )
    row = unload_rows[0]
    action_id = int(row["effective_target_action_id"])
    target_x = int(row["target_x"])
    target_y = int(row["target_y"])
    distance_steps = action_distance_steps_for_agent(env, picker, action_id)
    lines = [
        "Post-delivery helper:",
        f"- AGV agent_{int(row['agv_idx'])} already delivered its shelf and is still carrying it.",
        "- This is now the post-delivery unload support phase.",
        "- That AGV must return the shelf to an EMPTY shelf location for unload.",
        "- Do not wait at the old request shelf after delivery.",
        "- The current support location is the EMPTY shelf return location, not the original request shelf.",
        f"- Support the AGV at the same unload shelf action id.",
        f"- Current unload shelf: action_id={action_id}:SHELF@({target_x},{target_y}) for returning the delivered shelf.",
        f"- Recommended unload support action now: {action_id}:SHELF@({target_x},{target_y}) distance_steps={distance_steps}.",
        "- Move to the same unload shelf action id as the AGV.",
        "- Do not choose charging unless battery is critical.",
        "- This post-delivery phase is complete only after the AGV unloads the shelf there.",
    ]
    payload = {
        "post_delivery_active": True,
        "agv_id_needing_unload_support": int(row["agv_idx"]),
        "support_action_id": action_id,
        "support_position": {"x": target_x, "y": target_y},
        "support_type": "unload",
        "charging_override_only_if_battery_critical": True,
        "current_post_delivery_unload_action_id": action_id,
        "current_post_delivery_unload_position": {"x": target_x, "y": target_y},
    }
    return "\n".join(lines) + "\n\n", payload


def agv_role_context(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[agent_idx]
    phase = classify_agent_phase(env, agent_idx)
    target_id = int(agent.target)
    target_type = current_target_type(env, agent)
    target_distance = action_distance_steps_for_agent(env, agent, target_id) if target_id > 0 else 0
    carrying_status = carrying_status_label(env, agent)
    nearest_goal_id, nearest_goal_steps = nearest_goal_for_agent(env, agent)
    nearest_charge_id, nearest_charge_steps = nearest_charging_station_for_agent(env, agent)
    empty_slots = nearest_empty_shelf_slots_for_agent(env, agent, limit=5)
    support_flags = support_flags_for_agv(env, agent_idx, persisted_actions, hold_steps_remaining)
    next_flow = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)
    battery_need = battery_need_label(float(agent.battery))
    charge_policy = agv_task_ready_charge_policy(env, agent_idx, persisted_actions, hold_steps_remaining)

    requested_shelf_ids = {int(shelf.id) for shelf in env.request_queue}
    carrying_shelf_id = int(agent.carrying_shelf.id) if agent.carrying_shelf is not None else None
    carrying_delivery_status = "not_carrying"
    if carrying_shelf_id is not None:
        carrying_delivery_status = (
            "still_requested_not_delivered_yet"
            if carrying_shelf_id in requested_shelf_ids
            else "already_delivered_return_to_empty_shelf"
        )
    goal_allowed = carrying_status == "carrying_requested_shelf"
    empty_shelf_return_required = carrying_status == "carrying_delivered_shelf"
    if carrying_status == "not_carrying":
        preferred_action_family = "requested_shelf" if battery_need == "not_needed" and next_flow != "charge_now" else "requested_shelf_or_charging"
        forbidden_action_family = "goal"
    elif carrying_status == "carrying_requested_shelf":
        preferred_action_family = "goal"
        forbidden_action_family = "none"
    else:
        if next_flow == "wait_for_picker_at_unload_shelf":
            preferred_action_family = "wait_at_unload_shelf"
        else:
            preferred_action_family = "empty_shelf_return" if battery_need == "not_needed" and next_flow != "charge_now" else "empty_shelf_return_or_charging"
        forbidden_action_family = "goal"

    lines = [
        f"- AGV phase: {phase}",
        f"- Carrying status: {carrying_status}",
        f"- Carrying delivery status: {carrying_delivery_status}",
        f"- Battery need: {battery_need}",
        f"- Current target type: {target_type}",
        f"- Current target distance: {target_distance if target_id > 0 else 'none'}",
        f"- GOAL allowed now: {goal_allowed}",
        f"- EMPTY shelf return required now: {empty_shelf_return_required}",
        f"- Preferred action family now: {preferred_action_family}",
        f"- Forbidden action family now: {forbidden_action_family}",
        f"- Next required flow state: {next_flow}",
        f"- waiting_for_picker_support: {bool(support_flags['waiting_for_picker_support'])}",
        f"- picker_at_same_cell: {bool(support_flags['picker_at_same_cell'])}",
        f"- picker_support_inbound_now: {bool(support_flags['picker_support_inbound_now'])}",
        f"- picker_support_distance_steps: {support_flags['picker_support_distance_steps'] if support_flags['picker_support_distance_steps'] is not None else 'none'}",
        f"- needs_picker_for_load: {bool(support_flags['needs_picker_for_load'])}",
        f"- needs_picker_for_unload: {bool(support_flags['needs_picker_for_unload'])}",
        f"- Nearest GOAL: action_id={nearest_goal_id} distance_steps={nearest_goal_steps}",
        f"- Nearest charging station: action_id={nearest_charge_id} distance_steps={nearest_charge_steps}",
    ]
    if phase == "waiting_for_picker_at_requested_shelf":
        lines.extend(
            [
                "- The AGV has already reached the requested shelf.",
                "- Picker support has been raised for this shelf and Picker should be moving here.",
                "- Hold position and wait unless battery is critical or another very important action is required.",
            ]
        )
        if bool(support_flags["picker_support_inbound_now"]):
            lines.extend(
                [
                    "- Picker is on the way to this same shelf now.",
                    "- Do not leave this shelf now.",
                ]
            )
    if next_flow == "wait_for_picker_at_unload_shelf":
        lines.extend(
            [
                "- The AGV has already reached the EMPTY shelf return location.",
                "- Picker support is required here for unloading.",
                "- Hold position and wait for the Picker on this shelf.",
            ]
        )
        if bool(support_flags["picker_support_inbound_now"]):
            lines.extend(
                [
                    "- Picker is on the way to this same shelf now.",
                    "- Do not leave this shelf now.",
                ]
            )
    if target_type == "CHARGING" and float(agent.battery) < float(charge_policy["required_battery"]):
        lines.extend(
            [
                f"- Once you are on this charging station, stay here until battery reaches at least {int(charge_policy['required_battery'])} so you have enough charge for long tasks.",
                "- Ignore other requests and prioritize charging while this threshold is not met.",
            ]
        )
        if carrying_status == "carrying_delivered_shelf":
            lines.append(
                f"- Unload support actionable now: {bool(charge_policy['picker_support_actionable'])}."
            )
            if not bool(charge_policy["picker_support_actionable"]):
                lines.append("- Picker is not yet effectively in place for unload support, so charging should continue.")
    if empty_slots:
        lines.append("- Candidate EMPTY shelf return locations:")
        lines.extend(
            f"  - action_id={row['action_id']} at ({row['x']},{row['y']}) distance_steps={row['distance_steps']} suggested_steps={min(10, max(1, int(row['distance_steps'])))}"
            for row in empty_slots
        )

    payload = {
        "agv_state": {
            "phase": phase,
            "carrying_status": carrying_status,
            "carrying_delivery_status": carrying_delivery_status,
            "battery_need": battery_need,
            "battery": float(agent.battery),
            "current_target_type": target_type,
            "current_target_distance_steps": target_distance if target_id > 0 else None,
            "goal_allowed": goal_allowed,
            "empty_shelf_return_required": empty_shelf_return_required,
            "preferred_action_family": preferred_action_family,
            "forbidden_action_family": forbidden_action_family,
            "hold_position_preferred": next_flow in {"wait_for_picker_at_requested_shelf", "wait_for_picker_at_unload_shelf"},
            "next_required_flow_state": next_flow,
            "load_completed_and_goal_required_now": carrying_status == "carrying_requested_shelf",
            "original_load_shelf_action_id": int(target_id) if target_id > 0 and target_type == "SHELF" else None,
            "recommended_goal_action_id": nearest_goal_id if carrying_status == "carrying_requested_shelf" else None,
            "recommended_goal_distance_steps": nearest_goal_steps if carrying_status == "carrying_requested_shelf" else None,
        },
        "support_flags": support_flags,
        "goal_delivery_context": {
            "nearest_goal_action_id": nearest_goal_id,
            "nearest_goal_distance_steps": nearest_goal_steps,
            "goal_is_required_when_carrying_requested_shelf": True,
        },
        "empty_shelf_return_context": {
            "empty_shelf_return_required": empty_shelf_return_required,
            "goal_allowed": goal_allowed,
            "goal_forbidden_when_carrying_delivered_shelf": True,
            "instruction": "choose one EMPTY shelf action, not GOAL",
            "recommended_empty_shelf_actions": [
                {**row, "suggested_steps": min(10, max(1, int(row["distance_steps"])))}
                for row in empty_slots[:5]
            ],
        },
        "charging_context": {
            "nearest_charging_action_id": nearest_charge_id,
            "nearest_charging_distance_steps": nearest_charge_steps,
            "charge_until_task_ready_active": bool(target_type == "CHARGING" and float(agent.battery) < float(charge_policy["required_battery"])),
            "required_battery_for_release": int(charge_policy["required_battery"]),
            "picker_support_actionable": bool(charge_policy["picker_support_actionable"]),
            "release_reason": str(charge_policy["release_reason"]),
        },
    }
    return "\n".join(lines), payload


def picker_role_context(
    env,
    picker_idx: int,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[picker_idx]
    phase = classify_agent_phase(env, picker_idx)
    support_rows = effective_picker_support_rows(env, persisted_actions, hold_steps_remaining)
    task_view = [
        f"agv agent_{row['agv_idx']}: target={row['effective_target_action_id']}:SHELF "
        f"pos=({row['agv_x']},{row['agv_y']}) carrying={row['carrying']} "
        f"support_need={row['support_need']} state_source={row['state_source']}"
        for row in support_rows
    ]
    support_candidates = picker_support_candidate_action_lines(
        env,
        picker_idx,
        valid_masks,
        persisted_actions,
        hold_steps_remaining,
    )
    battery_need = battery_need_label(float(agent.battery))
    support_needed_now = any("waiting_for_picker_support_now" in row for row in support_candidates)
    charging_allowed_now = battery_need == "critical" or not support_needed_now
    flags = picker_support_flags(env, picker_idx, persisted_actions, hold_steps_remaining)
    next_flow = next_required_flow_state(env, picker_idx, persisted_actions, hold_steps_remaining)
    charge_policy = picker_task_ready_charge_policy(env, picker_idx, persisted_actions, hold_steps_remaining)
    lines = [
        f"- Picker phase: {phase}",
        f"- Battery need: {battery_need}",
        f"- Support needed now: {support_needed_now}",
        f"- Charging allowed now: {charging_allowed_now}",
        f"- support_target_action_id: {flags['support_target_action_id'] if flags['support_target_action_id'] is not None else 'none'}",
        f"- support_type: {flags['support_type']}",
        f"- at_support_target: {bool(flags['at_support_target'])}",
        f"- agv_waiting_for_picker_now: {bool(flags['agv_waiting_for_picker_now'])}",
        f"- Next required flow state: {next_flow}",
        "- Picker support happens at shelf interaction points, not at GOAL.",
        "- Align to the same shelf action id as the AGV support target.",
        "- Highest priority is support needed now, then support needed soon for load/unload.",
        "- If support is needed now and battery is not critical, support overrides charging.",
        "- AGV support timing view:",
    ]
    if current_target_type(env, agent) == "CHARGING" and float(agent.battery) < float(charge_policy["required_battery"]):
        lines.extend(
            [
                f"- Once you are on this charging station, stay here until battery reaches at least {int(charge_policy['required_battery'])} so you have enough charge for long tasks.",
                "- Ignore other requests and prioritize charging while this threshold is not met.",
            ]
        )
    lines.extend(f"  - {row}" for row in (task_view if task_view else ["none"]))
    payload = {
        "picker_support_context": {
            "phase": phase,
            "battery_need": battery_need,
            "battery": float(agent.battery),
            "support_needed_now": support_needed_now,
            "charging_allowed_now": charging_allowed_now,
            "support_happens_at_shelf_not_goal": True,
            "agv_support_timing_view": task_view,
            "agv_waiting_for_picker_now": bool(flags["agv_waiting_for_picker_now"]),
            "support_target_action_id": flags["support_target_action_id"],
            "support_target_position": flags["support_target_position"],
            "support_type": flags["support_type"],
            "current_target_type": current_target_type(env, agent),
            "next_required_flow_state": next_flow,
            "charge_until_task_ready_active": bool(current_target_type(env, agent) == "CHARGING" and float(agent.battery) < float(charge_policy["required_battery"])),
            "required_battery_for_release": int(charge_policy["required_battery"]),
        },
        "picker_support_candidate_actions": support_candidates,
    }
    return "\n".join(lines), payload


def picker_hard_constraints_text_from_payload(role_context_payload: Dict[str, Any]) -> str:
    picker_context = role_context_payload.get("picker_support_context", {})
    support_needed_now = bool(picker_context.get("support_needed_now", False))
    battery_need = str(picker_context.get("battery_need", "not_needed"))
    next_required_flow_state = str(picker_context.get("next_required_flow_state", "idle"))
    current_target_type = str(picker_context.get("current_target_type", "NOOP"))
    battery_value = float(picker_context.get("battery", 0.0))
    if battery_value >= 100.0:
        lines = [
            "- Battery is 100 => charging is forbidden.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    elif battery_need == "not_needed" and next_required_flow_state != "charge_now":
        lines = [
            "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    elif support_needed_now and battery_need != "critical":
        lines = [
            "- AGV support is needed now => choose the matching support shelf action, not charging.",
            "- Battery is not critical => charging is forbidden right now.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    elif battery_need == "critical":
        lines = [
            "- Battery is critical => charging is allowed and preferred.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    else:
        lines = ["- Only choose an action id that appears in Candidate actions."]
    if current_target_type == "CHARGING" and battery_value < 80.0:
        lines.insert(0, "- Once you are on a charging station, stay there until battery reaches at least 80 so you have enough charge for long tasks.")
        lines.insert(1, "- Ignore other requests and prioritize charging while this threshold is not met.")
    return "\n".join(lines)


def agv_hard_constraints_text_from_payload(role_context_payload: Dict[str, Any]) -> str:
    agv_state = role_context_payload.get("agv_state", {})
    carrying_status = str(agv_state.get("carrying_status", "not_carrying"))
    goal_allowed = bool(agv_state.get("goal_allowed", False))
    empty_shelf_return_required = bool(agv_state.get("empty_shelf_return_required", False))
    battery_need = str(agv_state.get("battery_need", "not_needed"))
    next_required_flow_state = str(agv_state.get("next_required_flow_state", "idle"))
    charge_until_task_ready_active = bool(agv_state.get("current_target_type") == "CHARGING" and float(agv_state.get("battery", 0.0)) < float(agv_state.get("required_battery_for_release", 80)))
    if carrying_status == "not_carrying":
        lines = [
            "- carrying=None => GOAL action ids are forbidden.",
            "- carrying=None => choose a requested shelf action.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    elif goal_allowed:
        lines = [
            "- carrying=requested shelf => GOAL action ids are allowed and preferred.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    elif empty_shelf_return_required:
        lines = [
            "- carrying=delivered shelf => GOAL action ids are forbidden.",
            "- carrying=delivered shelf => choose an EMPTY shelf return action.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    else:
        lines = ["- Only choose an action id that appears in Candidate actions."]
    if charge_until_task_ready_active:
        lines = [
            f"- Once you are on a charging station, stay there until battery reaches at least {int(agv_state.get('required_battery_for_release', 80))} so you have enough charge for long tasks.",
            "- Ignore other requests and prioritize charging while this threshold is not met.",
            "- Only choose an action id that appears in Candidate actions.",
        ]
    if float(agv_state.get("battery", 0.0)) >= 100.0:
        lines.insert(0, "- Battery is 100 => charging is forbidden.")
    elif battery_need == "not_needed" and next_required_flow_state != "charge_now":
        lines.insert(0, "- If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.")
    if bool(agv_state.get("hold_position_preferred", False)):
        lines.extend(
            [
                "- AGV is already at the requested shelf and waiting for Picker support.",
                "- Hold position unless battery is critical or another very important action is required.",
            ]
        )
    return "\n".join(lines)


def build_shared_context(env, agent, valid_masks: np.ndarray, agent_idx: int) -> str:
    fields = {
        "stated_objective": STATED_OBJECTIVE,
        "requested_shelves": format_bullets(requested_shelves_with_agent_distance(env, agent)),
        "charging_occupancy": format_bullets(charging_station_occupancy(env)),
    }
    return render_template(SHARED_CONTEXT_TEXT_TEMPLATE, fields)


def build_shared_context_payload(
    env,
    agent,
    agent_idx: int,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Dict[str, Any]:
    if agent.type.name == "PICKER":
        role_context_text, role_context_payload = picker_role_context(
            env,
            agent_idx,
            valid_masks,
            persisted_actions,
            hold_steps_remaining,
        )
    else:
        role_context_text, role_context_payload = agv_role_context(
            env,
            agent_idx,
            persisted_actions,
            hold_steps_remaining,
        )
    payload = {
        "stated_objective": STATED_OBJECTIVE,
        "basic_summary": BASIC_SUMMARY_LINES,
        "typical_flow": TYPICAL_FLOW_LINES,
        "battery_model": BATTERY_MODEL_LINES,
        "shared_coordination_context": {
            "requested_shelves": requested_shelves_with_agent_distance(env, agent),
            "charging_place_occupancy": charging_station_occupancy(env),
        },
        "role_specific_context": role_context_payload,
    }
    if agent.type.name == "PICKER":
        payload["shared_coordination_context"]["agvs_requesting_picker_support"] = [
            (
                f"agent_{row['agv_idx']} target={row['effective_target_action_id']}:SHELF@({row['target_x']},{row['target_y']}) "
                f"pos=({row['agv_x']},{row['agv_y']}) carrying={row['carrying']} "
                f"support_need={row['support_need']} state_source={row['state_source']}"
            )
            for row in effective_picker_support_rows(env, persisted_actions, hold_steps_remaining)
        ]
    payload["_role_context_text"] = role_context_text
    return payload


def obj2_candidate_action_ids(env, agent_idx: int, valid_masks: np.ndarray) -> List[int]:
    agent = env.agents[agent_idx]
    valid_ids = {int(i) for i in np.where(valid_masks[agent_idx] > 0)[0].tolist()}
    charging_ids = {int(i) for i in charging_action_ids(env)}
    allowed = set(charging_ids)

    if agent.type.name == "PICKER":
        support_ids: set[int] = set()
        for line in picker_support_candidate_action_lines(env, agent_idx, valid_masks):
            match = re.match(r"(\d+):SHELF", line)
            if match is not None:
                support_ids.add(int(match.group(1)))
        allowed.update(support_ids)
        battery_need = battery_need_label(float(agent.battery))
        if not support_ids and battery_need == "not_needed":
            allowed.add(0)
        candidate_ids = [action_id for action_id in sorted(allowed) if action_id in valid_ids]
        if 0 in allowed:
            candidate_ids = [0] + candidate_ids
        return candidate_ids if candidate_ids else [0]

    carrying_status = carrying_status_label(env, agent)
    if carrying_status == "carrying_requested_shelf":
        allowed.update(range(1, len(env.goals) + 1))
    elif carrying_status == "carrying_delivered_shelf":
        allowed.update(int(row["action_id"]) for row in nearest_empty_shelf_slots_for_agent(env, agent, limit=6))
    else:
        allowed.update(get_requested_action_ids(env))

    candidate_ids = [action_id for action_id in sorted(allowed) if action_id in valid_ids]
    return candidate_ids if candidate_ids else [0]


def obj2_candidate_action_ids_with_support(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> List[int]:
    agent = env.agents[agent_idx]
    valid_ids = {int(i) for i in np.where(valid_masks[agent_idx] > 0)[0].tolist()}
    charging_ids = {int(i) for i in charging_action_ids(env)}
    allowed = set(charging_ids)

    if agent.type.name == "PICKER":
        support_ids = {
            int(row["effective_target_action_id"])
            for row in effective_picker_support_rows(env, persisted_actions, hold_steps_remaining)
        }
        allowed.update(support_ids)
        battery_need = battery_need_label(float(agent.battery))
        if not support_ids and battery_need == "not_needed":
            allowed.add(0)
        candidate_ids = [action_id for action_id in sorted(allowed) if action_id in valid_ids]
        if 0 in allowed:
            candidate_ids = [0] + candidate_ids
        return candidate_ids if candidate_ids else [0]

    return obj2_candidate_action_ids(env, agent_idx, valid_masks)


def candidate_action_tag(
    env,
    agent_idx: int,
    action_id: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> str:
    agent = env.agents[agent_idx]
    if (
        int(action_id) > 0
        and bool(getattr(agent, "resolution_failed_recently", False))
        and int(getattr(agent, "unreachable_target_action_id", 0)) == int(action_id)
        and int(getattr(agent, "unreachable_target_cooldown_steps", 0)) > 0
    ):
        return "temporarily_unreachable_do_not_choose_now"
    action_type = classify_action(env, int(action_id)) if int(action_id) > 0 else "NOOP"
    next_flow = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)
    if action_type == "CHARGING":
        battery_need = battery_need_label(float(agent.battery))
        if float(agent.battery) >= 100.0:
            return "charging_full_battery_forbidden_now"
        if current_target_type(env, agent) == "CHARGING" and float(agent.battery) < 80.0 and int(action_id) == int(agent.target):
            return "continue_charging_preferred"
        if battery_need == "not_needed" and next_flow != "charge_now":
            return "charging_forbidden_now"
        return "charge_only_if_charging_is_needed"
    if int(action_id) == 0:
        return "idle_fallback"
    if agent.type.name == "AGV":
        if carrying_status_label(env, agent) == "carrying_requested_shelf" and action_type == "SHELF":
            return "completed_load_shelf_do_not_reselect"
        if next_flow == "wait_for_picker_at_requested_shelf" and int(action_id) == int(agent.target):
            return "wait_for_picker_here"
        if next_flow == "wait_for_picker_at_unload_shelf" and int(action_id) == int(agent.target):
            return "wait_for_picker_unload_here"
        if next_flow == "move_requested_shelf_to_goal" and action_type == "GOAL":
            return "deliver_now"
        if next_flow == "return_delivered_shelf_to_empty_shelf" and action_type == "SHELF":
            return "return_empty_shelf"
        if next_flow == "move_to_requested_shelf" and action_type == "SHELF":
            return "requested_shelf"
    else:
        if next_flow in {"move_to_support_load", "move_to_support_unload", "hold_at_support_shelf"} and action_type == "SHELF":
            return "support_now"
    return "candidate"


def candidate_action_instruction(
    env,
    agent_idx: int,
    action_id: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> str:
    agent = env.agents[agent_idx]
    action_type = classify_action(env, int(action_id)) if int(action_id) > 0 else "NOOP"
    tag = candidate_action_tag(env, agent_idx, action_id, persisted_actions, hold_steps_remaining)
    if tag == "temporarily_unreachable_do_not_choose_now":
        return "invalid_now_choose_another_action"
    if action_type == "CHARGING":
        if tag == "charging_full_battery_forbidden_now":
            return "forbidden_now_battery_already_full"
        if tag == "continue_charging_preferred":
            return "prefer_this_to_continue_charging_until_80"
        if tag == "charging_forbidden_now":
            return "forbidden_now_do_not_choose_charging"
        return "choose_this_only_if_charging_is_needed"
    if agent.type.name == "AGV":
        if tag == "wait_for_picker_unload_here":
            return "hold_this_unload_shelf_for_picker_support"
        if tag == "deliver_now":
            return "choose_this_to_deliver_to_goal"
        if tag == "return_empty_shelf":
            return "choose_this_to_unload_delivered_shelf"
    else:
        if tag == "support_now":
            support_flags = picker_support_flags(env, agent_idx, persisted_actions, hold_steps_remaining)
            if str(support_flags.get("support_type", "none")) == "unload":
                return "choose_this_to_support_unloading_on_exact_same_shelf_cell"
            return "choose_this_to_support_loading"
    return "candidate"


def recommended_now_for_candidate(
    env,
    agent_idx: int,
    action_id: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> bool:
    agent = env.agents[agent_idx]
    if (
        int(action_id) > 0
        and bool(getattr(agent, "resolution_failed_recently", False))
        and int(getattr(agent, "unreachable_target_action_id", 0)) == int(action_id)
        and int(getattr(agent, "unreachable_target_cooldown_steps", 0)) > 0
    ):
        return False
    next_flow = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)
    action_type = classify_action(env, int(action_id)) if int(action_id) > 0 else "NOOP"
    if agent.type.name == "AGV":
        if action_type == "CHARGING" and current_target_type(env, agent) == "CHARGING" and float(agent.battery) < 80.0:
            return int(action_id) == int(agent.target)
        if next_flow == "wait_for_picker_at_unload_shelf" and int(action_id) == int(agent.target) and action_type == "SHELF":
            return True
        if next_flow == "move_requested_shelf_to_goal" and action_type == "GOAL":
            nearest_goal_id, _ = nearest_goal_for_agent(env, agent)
            return int(action_id) == int(nearest_goal_id)
        if next_flow == "return_delivered_shelf_to_empty_shelf" and action_type == "SHELF":
            empty_slots = nearest_empty_shelf_slots_for_agent(env, agent, limit=1)
            return bool(empty_slots) and int(action_id) == int(empty_slots[0]["action_id"])
        return False

    if action_type == "CHARGING" and current_target_type(env, agent) == "CHARGING" and float(agent.battery) < 80.0:
        return int(action_id) == int(agent.target)
    if next_flow in {"move_to_support_load", "move_to_support_unload", "hold_at_support_shelf"} and action_type == "SHELF":
        flags = picker_support_flags(env, agent_idx, persisted_actions, hold_steps_remaining)
        support_target_action_id = flags.get("support_target_action_id")
        return support_target_action_id is not None and int(action_id) == int(support_target_action_id)
    return False


def build_candidate_views(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    max_hold_steps: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> Tuple[str, str]:
    agent = env.agents[agent_idx]
    candidate_ids = obj2_candidate_action_ids_with_support(
        env,
        agent_idx,
        valid_masks,
        persisted_actions,
        hold_steps_remaining,
    )
    if float(agent.battery) >= 100.0:
        candidate_ids = [
            int(action_id)
            for action_id in candidate_ids
            if classify_action(env, int(action_id)) != "CHARGING"
        ]
    primary_candidate_rows: List[Tuple[int, int, str]] = []
    charging_candidate_rows: List[Tuple[int, int, str]] = []
    for action_id in candidate_ids:
        if valid_masks[agent_idx, action_id] <= 0:
            continue
        distance_steps = action_distance_steps_for_agent(env, agent, action_id)
        suggested_steps = suggested_steps_for_action(env, agent_idx, action_id, max_hold_steps)
        tag = candidate_action_tag(env, agent_idx, action_id, persisted_actions, hold_steps_remaining)
        instruction = candidate_action_instruction(env, agent_idx, action_id, persisted_actions, hold_steps_remaining)
        recommended_now = "yes" if recommended_now_for_candidate(env, agent_idx, action_id, persisted_actions, hold_steps_remaining) else "no"
        description = describe_action_id_for_agent(env, agent, action_id)
        if "distance_steps=" in description:
            description = f"{description} suggested_steps={suggested_steps} tag={tag} recommended_now={recommended_now} instruction={instruction}"
        else:
            description = f"{description} (suggested_steps={suggested_steps}) tag={tag} recommended_now={recommended_now} instruction={instruction}"
        row_tuple = (distance_steps, int(action_id), description)
        if classify_action(env, int(action_id)) == "CHARGING":
            charging_candidate_rows.append(row_tuple)
        else:
            primary_candidate_rows.append(row_tuple)
    primary_candidate_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    charging_candidate_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    primary_lines = [line for _, _, line in primary_candidate_rows]
    charging_lines = [line for _, _, line in charging_candidate_rows]
    if not primary_lines and not charging_lines:
        primary_lines = ["0:NOOP (distance_steps=0)"]
    next_flow = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)
    text_lines: List[str]
    text_lines = [f"- {line}" for line in primary_lines]
    if charging_lines:
        if text_lines:
            text_lines.append("- IF CHARGING is needed, select following actions:")
        else:
            text_lines = ["- IF CHARGING is needed, select following actions:"]
        text_lines.extend(f"- {line}" for line in charging_lines)
    text_view = "\n".join(text_lines)
    json_view = json.dumps(
        {
            "primary_candidate_actions": primary_lines,
            "charging_candidate_header": (
                "IF CHARGING is needed, select following actions:"
                if charging_lines
                else None
            ),
            "charging_candidate_actions": charging_lines,
        },
        indent=2,
        ensure_ascii=True,
    )
    return text_view, json_view


def build_structured_candidate_actions(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    max_hold_steps: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> List[Dict[str, Any]]:
    agent = env.agents[agent_idx]
    candidate_ids = obj2_candidate_action_ids_with_support(
        env,
        agent_idx,
        valid_masks,
        persisted_actions,
        hold_steps_remaining,
    )
    if float(agent.battery) >= 100.0:
        candidate_ids = [
            int(action_id)
            for action_id in candidate_ids
            if classify_action(env, int(action_id)) != "CHARGING"
        ]
    rows: List[Dict[str, Any]] = []
    for action_id in candidate_ids:
        if valid_masks[agent_idx, action_id] <= 0:
            continue
        coords = env.action_id_to_coords_map.get(action_id) if action_id > 0 else None
        distance_steps = 0
        if action_id > 0 and coords is not None:
            path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
            distance_steps = max(1, len(path)) if path else 9999
        suggested_steps = suggested_steps_for_action(env, agent_idx, int(action_id), max_hold_steps=max_hold_steps)
        tag = candidate_action_tag(env, agent_idx, int(action_id), persisted_actions, hold_steps_remaining)
        instruction = candidate_action_instruction(env, agent_idx, int(action_id), persisted_actions, hold_steps_remaining)
        recommended_now = recommended_now_for_candidate(env, agent_idx, int(action_id), persisted_actions, hold_steps_remaining)
        rows.append(
            {
                "action_id": int(action_id),
                "action_type": classify_action(env, int(action_id)) if action_id > 0 else "NOOP",
                "target_position": (
                    {"x": int(coords[1]), "y": int(coords[0])}
                    if coords is not None
                    else None
                ),
                "distance_steps": int(distance_steps),
                "suggested_steps": int(suggested_steps),
                "tag": str(tag),
                "instruction": str(instruction),
                "recommended_now": bool(recommended_now),
                "description": (
                    f"{describe_action_id_for_agent(env, agent, int(action_id))} suggested_steps={int(suggested_steps)} tag={str(tag)} recommended_now={'yes' if recommended_now else 'no'} instruction={str(instruction)}"
                ),
            }
        )
    if not rows:
        rows.append(
            {
                "action_id": 0,
                "action_type": "NOOP",
                "target_position": None,
                "distance_steps": 0,
                "suggested_steps": 1,
                "tag": "idle_fallback",
                "instruction": "candidate",
                "recommended_now": False,
                "description": "0:NOOP (distance_steps=0) suggested_steps=1 tag=idle_fallback recommended_now=no instruction=candidate",
            }
        )
    primary_rows = [row for row in rows if str(row.get("action_type", "")) != "CHARGING"]
    charging_rows = [row for row in rows if str(row.get("action_type", "")) == "CHARGING"]
    primary_rows.sort(key=lambda row: (int(row["distance_steps"]), int(row["action_id"])))
    charging_rows.sort(key=lambda row: (int(row["distance_steps"]), int(row["action_id"])))
    return primary_rows + charging_rows


def grouped_structured_candidate_actions(
    rows: List[Dict[str, Any]],
    next_flow: str = "",
) -> Dict[str, Any]:
    del next_flow
    primary_candidate_actions = [
        row for row in rows if str(row.get("action_type", "")) != "CHARGING"
    ]
    charging_candidate_actions = [
        row for row in rows if str(row.get("action_type", "")) == "CHARGING"
    ]
    return {
        "primary_candidate_actions": primary_candidate_actions,
        "charging_candidate_header": (
            "IF CHARGING is needed, select following actions:"
            if charging_candidate_actions
            else None
        ),
        "charging_candidate_actions": charging_candidate_actions,
    }


def compact_candidate_action_groups(
    env,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    compact_rows: List[Dict[str, Any]] = []
    for row in rows:
        target_position = row.get("target_position")
        target = None
        if isinstance(target_position, dict):
            target = f"({int(target_position.get('x', 0))},{int(target_position.get('y', 0))})"
        compact_rows.append(
            {
                "action": int(row.get("action_id", 0)),
                "kind": str(row.get("action_type", "NOOP")).lower(),
                "target": target,
                "distance_steps": int(row.get("distance_steps", 0)),
                "tag": str(row.get("tag", "candidate")),
                "recommended": bool(row.get("recommended_now", False)),
                "allowed_now": not str(row.get("instruction", "")).startswith("forbidden_now"),
                "instruction": str(row.get("instruction", "candidate")),
            }
        )
    primary_rows = [row for row in compact_rows if str(row.get("kind", "")) != "charging"]
    charging_rows = [row for row in compact_rows if str(row.get("kind", "")) == "charging"]
    return {
        "candidate_actions_compact": primary_rows + charging_rows,
        "primary_candidate_actions_compact": primary_rows,
        "charging_candidate_actions_compact": charging_rows,
    }


def split_bullet_text(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:]
        lines.append(line)
    return lines


def simple_json_candidate_actions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    simple_rows: List[Dict[str, Any]] = []
    for row in rows:
        target_position = row.get("target_position")
        position = None
        if isinstance(target_position, dict):
            position = {"x": int(target_position.get("x", 0)), "y": int(target_position.get("y", 0))}
        simple_rows.append(
            {
                "action": int(row.get("action_id", 0)),
                "type": str(row.get("action_type", "NOOP")),
                "position": position,
                "distance_steps": int(row.get("distance_steps", 0)),
                "suggested_steps": int(row.get("suggested_steps", 1)),
                "tag": str(row.get("tag", "candidate")),
                "recommended_now": bool(row.get("recommended_now", False)),
                "instruction": str(row.get("instruction", "candidate")),
            }
        )
    return simple_rows


def charging_allowed_now_from_candidates(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        if str(row.get("action_type", "")) != "CHARGING":
            continue
        if not str(row.get("instruction", "")).startswith("forbidden_now"):
            return True
    return False


def build_role_context_json(
    env,
    agent,
    role_specific_context: Dict[str, Any],
    control_state_payload: Dict[str, Any],
    empty_shelf_return_task_payload: Dict[str, Any],
    post_delivery_helper_payload: Dict[str, Any],
    post_load_helper_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if agent.type.name == "AGV":
        agv_state = role_specific_context.get("agv_state", {})
        support_flags = role_specific_context.get("support_flags", {})
        goal_ctx = role_specific_context.get("goal_delivery_context", {})
        charging_ctx = role_specific_context.get("charging_context", {})
        role_context: Dict[str, Any] = {
            "phase": agv_state.get("phase"),
            "carrying_status": agv_state.get("carrying_status"),
            "carrying_delivery_status": agv_state.get("carrying_delivery_status"),
            "battery_need": agv_state.get("battery_need"),
            "current_target_type": agv_state.get("current_target_type"),
            "current_target_distance_steps": agv_state.get("current_target_distance_steps"),
            "goal_allowed": bool(agv_state.get("goal_allowed", False)),
            "empty_shelf_return_required": bool(agv_state.get("empty_shelf_return_required", False)),
            "preferred_action_family": agv_state.get("preferred_action_family"),
            "forbidden_action_family": agv_state.get("forbidden_action_family"),
            "next_required_flow_state": control_state_payload.get("next_required_flow_state"),
            "waiting_for_picker_support": bool(support_flags.get("waiting_for_picker_support", False)),
            "picker_at_same_cell": bool(support_flags.get("picker_at_same_cell", False)),
            "picker_support_inbound_now": bool(support_flags.get("picker_support_inbound_now", False)),
            "picker_support_distance_steps": support_flags.get("picker_support_distance_steps"),
            "needs_picker_for_load": bool(support_flags.get("needs_picker_for_load", False)),
            "needs_picker_for_unload": bool(support_flags.get("needs_picker_for_unload", False)),
            "nearest_goal": {
                "action": goal_ctx.get("nearest_goal_action_id"),
                "distance_steps": goal_ctx.get("nearest_goal_distance_steps"),
            },
            "nearest_charging_station": {
                "action": charging_ctx.get("nearest_charging_action_id"),
                "distance_steps": charging_ctx.get("nearest_charging_distance_steps"),
            },
            "goal_delivery_context": goal_ctx,
            "empty_shelf_return_context": role_specific_context.get("empty_shelf_return_context", {}),
            "charging_context": charging_ctx,
        }
        if post_delivery_helper_payload.get("post_delivery_active"):
            role_context["post_delivery_helper"] = post_delivery_helper_payload
        if empty_shelf_return_task_payload.get("visible"):
            role_context["empty_shelf_return_task"] = empty_shelf_return_task_payload
        if post_load_helper_payload.get("post_load_active"):
            role_context["post_load_helper"] = post_load_helper_payload
        return role_context

    picker_ctx = role_specific_context.get("picker_support_context", {})
    role_context = {
        "phase": picker_ctx.get("phase"),
        "battery_need": picker_ctx.get("battery_need"),
        "support_needed_now": bool(picker_ctx.get("support_needed_now", False)),
        "charging_allowed_now": bool(picker_ctx.get("charging_allowed_now", True)),
        "support_target_action_id": picker_ctx.get("support_target_action_id"),
        "support_target_position": picker_ctx.get("support_target_position"),
        "support_type": picker_ctx.get("support_type"),
        "at_support_target": bool(picker_ctx.get("at_support_target", False)),
        "agv_waiting_for_picker_now": bool(picker_ctx.get("agv_waiting_for_picker_now", False)),
        "next_required_flow_state": control_state_payload.get("next_required_flow_state"),
        "support_happens_at_shelf_not_goal": True,
        "align_to_same_shelf_action_as_agv": True,
        "agv_support_timing_view": list(picker_ctx.get("agv_support_timing_view", [])),
        "picker_support_candidate_actions": list(role_specific_context.get("picker_support_candidate_actions", [])),
    }
    if post_delivery_helper_payload.get("post_delivery_active"):
        role_context["post_delivery_helper"] = post_delivery_helper_payload
    if post_load_helper_payload.get("post_load_active"):
        role_context["post_load_helper"] = post_load_helper_payload
    return role_context


def build_current_control_state_json(
    agent,
    control_state_payload: Dict[str, Any],
    role_context_json: Dict[str, Any],
    structured_candidate_actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    state = dict(control_state_payload)
    state["battery_need"] = str(role_context_json.get("battery_need", battery_need_label(float(agent.battery))))
    if agent.type.name == "AGV":
        state["preferred_action_family"] = role_context_json.get("preferred_action_family")
        state["forbidden_action_family"] = role_context_json.get("forbidden_action_family")
        state["goal_allowed"] = bool(role_context_json.get("goal_allowed", False))
        state["empty_shelf_return_required"] = bool(role_context_json.get("empty_shelf_return_required", False))
    state["charging_allowed_now"] = charging_allowed_now_from_candidates(structured_candidate_actions)
    return state


def decision_rules_for_agent(agent_type: str) -> List[str]:
    common_rules = [
        "Use next_required_flow_state before any longer-term preference.",
        "If charging_allowed_now is false, do not choose a charging action.",
        "If battery is 100, charging is forbidden.",
        "Prefer a candidate with recommended=true when it matches the required flow.",
    ]
    if agent_type == "AGV":
        return common_rules + [
            "If next_required_flow_state is move_to_requested_shelf, choose a requested shelf action, not a goal action.",
            "If next_required_flow_state is move_requested_shelf_to_goal, choose a GOAL candidate.",
            "If next_required_flow_state is return_delivered_shelf_to_empty_shelf, choose an empty shelf return candidate.",
            "If waiting_for_picker_support_now is true or must_not_leave_current_shelf is true, stay on the current support shelf.",
            "If picker_support_inbound_now is true, do not leave the current support shelf.",
        ]
    return common_rules + [
        "If next_required_flow_state is move_to_support_load or move_to_support_unload, choose the matching support shelf candidate.",
        "If support_needed_now is true and battery is not critical, support overrides charging.",
        "Picker support should be at the same shelf action id as the AGV support target, not at GOAL.",
    ]


def decision_summary_for_agent(
    env,
    agent_idx: int,
    agent,
    control_state_payload: Dict[str, Any],
    role_specific_context: Dict[str, Any],
    structured_candidate_actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    carrying_shelf_id = int(agent.carrying_shelf.id) if getattr(agent, "carrying_shelf", None) is not None else None
    current_target_action = int(agent.target) if int(getattr(agent, "target", 0)) > 0 else None
    battery_need = str(control_state_payload.get("battery_need", battery_need_label(float(agent.battery))))
    support_type = control_state_payload.get("support_type")
    picker_support_distance_steps = control_state_payload.get("picker_support_distance_steps")
    unreachable_target_action_id = control_state_payload.get("unreachable_target_action_id")
    unreachable_target_reason = control_state_payload.get("unreachable_target_reason")
    summary: Dict[str, Any] = {
        "agent_id": int(agent_idx),
        "agent_type": agent.type.name,
        "current_position": {"x": int(agent.x), "y": int(agent.y)},
        "current_battery": int(round(float(agent.battery))),
        "battery_need": battery_need,
        "at_charging_station": bool((int(agent.x), int(agent.y)) in charging_cells_for_env(env)),
        "busy": bool(agent.busy),
        "carrying_shelf_id": carrying_shelf_id,
        "current_target_action": current_target_action,
        "next_required_flow_state": control_state_payload.get("next_required_flow_state"),
        "preferred_action_family": control_state_payload.get("preferred_action_family"),
        "support_needed_now": bool(control_state_payload.get("support_needed_now", False)),
        "support_type": support_type if support_type is not None else None,
        "waiting_for_picker_support_now": bool(control_state_payload.get("waiting_for_picker_support_now", False)),
        "picker_support_inbound_now": bool(control_state_payload.get("picker_support_inbound_now", False)),
        "picker_support_distance_steps": (
            int(picker_support_distance_steps) if picker_support_distance_steps is not None else None
        ),
        "charging_allowed_now": bool(control_state_payload.get("charging_allowed_now", True)),
        "must_not_leave_current_shelf": bool(
            control_state_payload.get("waiting_for_picker_support_now", False)
            or control_state_payload.get("picker_support_inbound_now", False)
        ),
        "resolution_failed_recently": bool(control_state_payload.get("resolution_failed_recently", False)),
        "unreachable_target_action_id": (
            int(unreachable_target_action_id) if unreachable_target_action_id not in (None, 0) else None
        ),
        "unreachable_target_reason": unreachable_target_reason if unreachable_target_reason else None,
    }
    if agent.type.name == "AGV":
        agv_state = role_specific_context.get("agv_state", {})
        summary.update(
            {
                "carrying_requested_shelf": bool(agv_state.get("carrying_requested_shelf", False)),
                "carrying_delivered_shelf": bool(agv_state.get("carrying_delivered_shelf", False)),
                "needs_picker_for_load": bool(agv_state.get("needs_picker_for_load", False)),
                "needs_picker_for_unload": bool(agv_state.get("needs_picker_for_unload", False)),
                "at_requested_shelf": bool(agv_state.get("at_requested_shelf", False)),
                "at_unload_shelf": bool(agv_state.get("at_empty_shelf_target", False)),
            }
        )
    else:
        picker_support_context = role_specific_context.get("picker_support_context", {})
        summary.update(
            {
                "assigned_support_target_action": picker_support_context.get("support_target_action_id"),
                "support_target_agent_id": picker_support_context.get("agv_waiting_for_picker_agent_id"),
                "agv_waiting_for_picker_now": bool(picker_support_context.get("agv_waiting_for_picker_now", False)),
                "support_urgency": "now" if bool(picker_support_context.get("support_needed_now", False)) else ("soon" if bool(picker_support_context.get("support_needed_soon", False)) else "none"),
                "on_charger_and_should_continue_charging": bool(
                    (int(agent.x), int(agent.y)) in charging_cells_for_env(env) and float(agent.battery) < 80.0
                ),
            }
        )
    summary["candidate_actions_compact"] = compact_candidate_action_groups(env, structured_candidate_actions)["candidate_actions_compact"]
    return summary


def candidate_section_helper_text(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> str:
    agent = env.agents[agent_idx]
    next_flow = next_required_flow_state(env, agent_idx, persisted_actions, hold_steps_remaining)
    if agent.type.name == "AGV" and next_flow == "wait_for_picker_at_unload_shelf":
        return "Choose the current unload shelf candidate with recommended_now=yes and hold there for Picker support.\n\n"
    if agent.type.name == "AGV" and next_flow == "return_delivered_shelf_to_empty_shelf":
        return "Choose the candidate with recommended_now=yes to return and unload the delivered shelf.\n\n"
    if agent.type.name == "PICKER" and next_flow == "move_to_support_unload":
        return "Choose the candidate with recommended_now=yes to support unloading on the exact same shelf cell.\n\n"
    return ""


def execution_warning_for_agent(env, agent_idx: int) -> Tuple[str, Dict[str, Any] | None]:
    agent = env.agents[agent_idx]
    if not bool(getattr(agent, "resolution_failed_recently", False)):
        return "", None
    unreachable_action_id = int(getattr(agent, "unreachable_target_action_id", 0))
    if unreachable_action_id <= 0 or int(getattr(agent, "unreachable_target_cooldown_steps", 0)) <= 0:
        return "", None
    reason = str(getattr(agent, "unreachable_target_reason", "target_not_reachable"))
    lines = [
        "Execution warning:",
        f"- The current agent is stuck on target action {unreachable_action_id}.",
        f"- Target action {unreachable_action_id} is temporarily unreachable right now.",
        f"- Do not select action {unreachable_action_id} in this response.",
        "- You must choose a different valid action from Candidate actions.",
    ]
    payload = {
        "agent_stuck_now": True,
        "unreachable_target_action_id": int(unreachable_action_id),
        "unreachable_target_reason": reason,
        "must_choose_different_action_now": True,
        "forbidden_action_id_now": int(unreachable_action_id),
        "cooldown_steps_remaining": int(getattr(agent, "unreachable_target_cooldown_steps", 0)),
    }
    return "\n".join(lines) + "\n\n", payload


def build_last_decided_action_state(
    env,
    agent_idx: int,
    persisted_action: int,
    remaining_hold_steps: int,
) -> Tuple[str, Dict[str, Any]]:
    agent = env.agents[agent_idx]
    if 0 <= int(persisted_action) < env.action_size:
        action_description = describe_action_id_for_agent(env, agent, int(persisted_action))
    else:
        action_description = "unknown"
    payload = {
        "last_decided_action_id": int(persisted_action),
        "last_decided_action_description": action_description,
        "remaining_hold_steps": int(max(0, remaining_hold_steps)),
        "requery_required_now": bool(int(max(0, remaining_hold_steps)) <= 0),
    }
    text = "\n".join(
        [
            f"- Last decided action: {action_description}",
            f"- Remaining hold steps: {int(max(0, remaining_hold_steps))}",
            f"- Requery required now: {bool(int(max(0, remaining_hold_steps)) <= 0)}",
        ]
    )
    return text, payload


def movement_warning_for_agent(
    env,
    agent_idx: int,
    last_decided_action_payload: Dict[str, Any],
    busy_steps_count: int = 0,
    busy_override_active: bool = False,
) -> Tuple[str, Dict[str, Any] | None]:
    not_moved_steps = 0
    if 0 <= agent_idx < len(getattr(env, "stuck_counters", [])):
        not_moved_steps = int(getattr(env.stuck_counters[agent_idx], "count", 0))
    busy_steps_count = int(max(0, busy_steps_count))
    if not_moved_steps <= 0 and not busy_override_active:
        return "", None

    previous_target_action_id = int(last_decided_action_payload.get("last_decided_action_id", 0))
    previous_target_description = str(last_decided_action_payload.get("last_decided_action_description", "none"))
    if previous_target_action_id <= 0:
        previous_target_description = "none"

    lines = ["Movement warning:"]
    if not_moved_steps > 0:
        lines.append(f"- You have not moved for {not_moved_steps} steps.")
        lines.append("- If you have not moved for long, consider a different target.")
    if busy_override_active:
        lines.append(f"- Busy override warning: You have remained busy for {busy_steps_count} steps, so you are being replanned now.")
    lines.append(f"- Your previous target was: {previous_target_description}.")
    lines.append("")
    payload = {
        "not_moved_steps": int(not_moved_steps),
        "consider_different_target_if_long": True,
        "busy_steps_count": busy_steps_count,
        "busy_override_active": bool(busy_override_active),
        "previous_target_action_id": int(previous_target_action_id) if previous_target_action_id > 0 else None,
        "previous_target_description": previous_target_description,
    }
    return "\n".join(lines), payload


def blocking_warning_for_agent(env, agent_idx: int) -> Tuple[str, Dict[str, Any] | None]:
    blocker_ids: List[int] = []
    blocker_reasons: List[Any] = []
    if hasattr(env, "_blocking_agent_info"):
        blocker_ids, blocker_reasons = env._blocking_agent_info()

    blocked_by_agent_idx = int(blocker_ids[agent_idx]) if 0 <= agent_idx < len(blocker_ids) else -1
    blocked_reason = blocker_reasons[agent_idx] if 0 <= agent_idx < len(blocker_reasons) else None
    has_blocked_by = 0 <= agent_idx < len(blocker_ids) and blocked_reason is not None

    blocking_agent_idx = -1
    blocking_reason = None
    for other_idx, blocker_idx in enumerate(blocker_ids):
        if int(blocker_idx) == int(agent_idx):
            blocking_agent_idx = int(other_idx)
            blocking_reason = blocker_reasons[other_idx] if other_idx < len(blocker_reasons) else None
            break
    has_blocking_agent = blocking_agent_idx >= 0 and blocking_reason is not None

    if not has_blocked_by and not has_blocking_agent:
        return "", None

    lines = ["Blocking warning:"]
    if has_blocked_by:
        lines.append(f"- You are currently blocked by agent_{blocked_by_agent_idx}.")
        if blocked_reason:
            lines.append(f"- Reason: {blocked_reason}.")
        lines.append("- If you are blocked for long, choose a different valid action.")
    if has_blocking_agent:
        lines.append(f"- You may be blocking agent_{blocking_agent_idx}.")
        if blocking_reason:
            lines.append(f"- Reason: {blocking_reason}.")
        lines.append("- If possible, choose a different valid action to clear the path.")
    lines.append("")

    payload = {
        "blocked_by_agent_id": int(blocked_by_agent_idx) if has_blocked_by else None,
        "blocking_agent_id": int(blocking_agent_idx) if has_blocking_agent else None,
        "reason": blocked_reason or blocking_reason,
    }
    return "\n".join(lines), payload


def build_agent_prompt(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    prompt_format: str,
    max_hold_steps: int,
    persisted_action: int = 0,
    remaining_hold_steps: int = 0,
    persisted_actions_all: List[int] | None = None,
    hold_steps_remaining_all: List[int] | None = None,
    busy_steps_by_agent: List[int] | None = None,
    busy_override_active: bool = False,
) -> str:
    agent = env.agents[agent_idx]
    effective_max_hold_steps = max(1, int(max_hold_steps))
    effective_persisted_actions = list(persisted_actions_all) if persisted_actions_all is not None else [0] * env.num_agents
    effective_hold_steps_remaining = (
        list(hold_steps_remaining_all) if hold_steps_remaining_all is not None else [0] * env.num_agents
    )
    if 0 <= agent_idx < len(effective_persisted_actions):
        effective_persisted_actions[agent_idx] = int(persisted_action)
    if 0 <= agent_idx < len(effective_hold_steps_remaining):
        effective_hold_steps_remaining[agent_idx] = int(remaining_hold_steps)
    shared_context = build_shared_context(env, agent, valid_masks, agent_idx)
    candidate_actions_text, candidate_actions_json = build_candidate_views(
        env,
        agent_idx,
        valid_masks,
        effective_max_hold_steps,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    structured_candidate_actions = build_structured_candidate_actions(
        env,
        agent_idx,
        valid_masks,
        effective_max_hold_steps,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    current_next_flow = next_required_flow_state(
        env,
        agent_idx,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    structured_candidate_action_groups = grouped_structured_candidate_actions(
        structured_candidate_actions,
        current_next_flow,
    )
    picker_support_candidates = (
        picker_support_candidate_action_lines(
            env,
            agent_idx,
            valid_masks,
            effective_persisted_actions,
            effective_hold_steps_remaining,
        )
        if agent.type.name == "PICKER"
        else ["none"]
    )
    shared_context_payload = build_shared_context_payload(
        env,
        agent,
        agent_idx,
        valid_masks,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    role_context_text = str(shared_context_payload.pop("_role_context_text", ""))
    agv_hard_constraints_text = (
        agv_hard_constraints_text_from_payload(shared_context_payload.get("role_specific_context", {}))
        if agent.type.name == "AGV"
        else ""
    )
    picker_hard_constraints_text = (
        picker_hard_constraints_text_from_payload(shared_context_payload.get("role_specific_context", {}))
        if agent.type.name == "PICKER"
        else ""
    )
    steps_guidance_payload = suggested_steps_guidance(env, agent_idx, effective_max_hold_steps)
    last_decided_action_text, last_decided_action_payload = build_last_decided_action_state(
        env,
        agent_idx,
        persisted_action,
        remaining_hold_steps,
    )
    movement_warning_text, movement_warning_payload = movement_warning_for_agent(
        env,
        agent_idx,
        last_decided_action_payload,
        busy_steps_count=(busy_steps_by_agent[agent_idx] if busy_steps_by_agent is not None and 0 <= agent_idx < len(busy_steps_by_agent) else 0),
        busy_override_active=busy_override_active,
    )
    blocking_warning_text, blocking_warning_payload = blocking_warning_for_agent(env, agent_idx)
    control_state_text, control_state_payload = control_state_for_agent(
        env,
        agent_idx,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    empty_shelf_return_task_text, empty_shelf_return_task_payload = empty_shelf_return_task_block(
        shared_context_payload.get("role_specific_context", {})
    )
    post_load_helper_text, post_load_helper_payload = post_load_helper_for_agent(
        env,
        agent_idx,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    post_delivery_helper_text, post_delivery_helper_payload = post_delivery_helper_for_agent(
        env,
        agent_idx,
        valid_masks,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    execution_warning_text, execution_warning_payload = execution_warning_for_agent(env, agent_idx)
    candidate_helper_text = candidate_section_helper_text(
        env,
        agent_idx,
        effective_persisted_actions,
        effective_hold_steps_remaining,
    )
    combined_phase_helper_text = (
        post_load_helper_text
        + post_delivery_helper_text
        + execution_warning_text
        + candidate_helper_text
    )
    shared_context_payload.setdefault("role_specific_context", {})["post_load_helper"] = post_load_helper_payload
    shared_context_payload.setdefault("role_specific_context", {})["post_delivery_helper"] = post_delivery_helper_payload
    if execution_warning_payload is not None:
        shared_context_payload.setdefault("role_specific_context", {})["execution_warning"] = execution_warning_payload
    agv_flow_rules_text = (
        agv_flow_rules_text_from_payload(shared_context_payload.get("role_specific_context", {}))
        if agent.type.name == "AGV"
        else ""
    )
    agent_state_json = json.dumps(
        {
            "agent_id": agent_idx,
            "agent_type": agent.type.name,
            "self_state": render_self_state_with_target_distance(env, agent),
            "phase": classify_agent_phase(env, agent_idx),
            "target_type": current_target_type(env, agent),
            "next_required_flow_state": control_state_payload.get("next_required_flow_state"),
        },
        indent=2,
        ensure_ascii=True,
    )
    role_context_json = build_role_context_json(
        env,
        agent,
        shared_context_payload.get("role_specific_context", {}),
        control_state_payload,
        empty_shelf_return_task_payload,
        post_delivery_helper_payload,
        post_load_helper_payload,
    )
    current_control_state_json = build_current_control_state_json(
        agent,
        control_state_payload,
        role_context_json,
        structured_candidate_actions,
    )
    decision_rules = (
        split_bullet_text(agv_flow_rules_text)
        + [
            "If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.",
            "Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.",
            "If battery is 100, charging is forbidden.",
            "If picker_support_inbound_now=yes and picker_at_same_cell=no, stay on this shelf until Picker arrives and the shelf interaction completes.",
            "Use suggested_steps as the default steps choice.",
        ]
        if agent.type.name == "AGV"
        else [
            "Picker support should be at the same shelf action id as the AGV support target, not at GOAL.",
            "Do not enter a shelf cell unless the AGV is already there for support; otherwise stay near the shelf and wait.",
            "If battery_need=not_needed and next_required_flow_state is not charge_now, charging is forbidden.",
            "Once you are on a charging station with battery below 80, you must stay there until battery reaches at least 80 so you have enough charge for long tasks. Ignore other requests and prioritize charging.",
            "If battery is 100, charging is forbidden.",
            "Use suggested_steps as the default steps choice.",
        ]
    )
    prompt_payload = {
        "objective": STATED_OBJECTIVE,
        "shared_coordination_context": shared_context_payload.get("shared_coordination_context", {}),
        "current_agent": {
            "agent_id": int(agent_idx),
            "agent_type": agent.type.name,
            "self_state": render_self_state_with_target_distance(env, agent),
            "phase": classify_agent_phase(env, agent_idx),
            "target_type": current_target_type(env, agent),
        },
        "last_decided_action_state": last_decided_action_payload,
        "movement_warning": movement_warning_payload,
        "blocking_warning": blocking_warning_payload,
        "execution_warning": execution_warning_payload,
        "current_control_state": current_control_state_json,
        "role_context": role_context_json,
        "hard_constraints": split_bullet_text(
            agv_hard_constraints_text if agent.type.name == "AGV" else picker_hard_constraints_text
        ),
        "decision_rules": decision_rules,
        "candidate_actions": simple_json_candidate_actions(structured_candidate_actions),
        "output_contract": {
            "required_keys": ["reason", "action", "steps"],
            "action_must_be_integer": True,
            "steps_must_be_integer": True,
            "steps_range": {"min": 1, "max": effective_max_hold_steps},
        },
    }
    fields = {
        "shared_context": shared_context,
        "agent_id": agent_idx,
        "self_state": render_self_state_with_target_distance(env, agent),
        "last_decided_action_text": last_decided_action_text,
        "movement_warning_text": movement_warning_text,
        "blocking_warning_text": blocking_warning_text,
        "control_state_text": control_state_text,
        "post_delivery_helper_text": combined_phase_helper_text,
        "candidate_actions": candidate_actions_text,
        "agent_state_json": agent_state_json,
        "candidate_actions_json": candidate_actions_json,
        "picker_support_candidate_actions": format_bullets(picker_support_candidates),
        "agv_role_context_text": role_context_text if agent.type.name == "AGV" else "",
        "agv_flow_rules_text": agv_flow_rules_text,
        "empty_shelf_return_task_text": empty_shelf_return_task_text,
        "agv_hard_constraints_text": agv_hard_constraints_text,
        "picker_role_context_text": role_context_text if agent.type.name == "PICKER" else "",
        "picker_hard_constraints_text": picker_hard_constraints_text,
        "max_hold_steps": effective_max_hold_steps,
        "prompt_json": json.dumps(prompt_payload, indent=2, ensure_ascii=True),
    }
    if prompt_format == "json":
        template = AGV_JSON_TEMPLATE if agent.type.name == "AGV" else PICKER_JSON_TEMPLATE
    else:
        template = AGV_LANGUAGE_TEMPLATE if agent.type.name == "AGV" else PICKER_LANGUAGE_TEMPLATE
    return render_template(template, fields)


def parse_single_action_from_text(text: str) -> int:
    line_patterns = [
        r"^\s*action\s*[:=-]\s*(-?\d+)\b",
        r"^\s*action\s*[:=-]\s*(-?\d+)\s*:",
    ]
    for pattern in line_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))

    fallback_patterns = [
        r"action\s*[:=-]?\s*(-?\d+)",
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))
    raise ValueError("No parseable single action found in LLM output")


def parse_steps_from_text(text: str, max_hold_steps: int) -> int:
    steps = 1
    step_patterns = [
        r"^\s*steps?\s*[:=-]\s*(-?\d+)\b",
        r"^\s*duration\s*[:=-]\s*(-?\d+)\b",
        r"hold(?:_steps)?\s*[:=-]\s*(-?\d+)\b",
        r"\bsteps?\s*[:=-]\s*(-?\d+)\b",
        r"\bduration\s*[:=-]\s*(-?\d+)\b",
        r"\bhold(?:_steps)?\s*[:=-]\s*(-?\d+)\b",
    ]
    for pattern in step_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            steps = int(match.group(1))
            break
    return max(1, min(int(steps), max(1, int(max_hold_steps))))


def extract_json_object_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Unterminated JSON object in model output")


def parse_action_and_steps_with_metadata(
    text: str,
    prompt_format: str,
    max_hold_steps: int,
) -> Tuple[int, int, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "parsed_llm_output": None,
        "json_generation_failed": False,
        "json_parse_error": None,
    }
    if prompt_format == "json":
        try:
            payload = json.loads(extract_json_object_text(text))
            if isinstance(payload, dict) and "reason" not in payload and "reasoning" in payload:
                payload["reason"] = payload.get("reasoning")
            metadata["parsed_llm_output"] = payload
            action = int(payload.get("action", 0))
            steps = max(1, min(int(payload.get("steps", 1)), max(1, int(max_hold_steps))))
            if isinstance(metadata["parsed_llm_output"], dict):
                metadata["parsed_llm_output"]["steps"] = steps
            return action, steps, metadata
        except Exception as exc:
            metadata["json_generation_failed"] = True
            metadata["json_parse_error"] = f"{type(exc).__name__}: {exc}"
            explicit_action_match = re.search(
                r'"\s*action\s*"\s*:\s*(-?\d+)\b|^\s*action\s*[:=-]\s*(-?\d+)\b',
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if explicit_action_match is None:
                raise
            action = int(next(group for group in explicit_action_match.groups() if group is not None))
            explicit_steps_match = re.search(
                r'"\s*steps\s*"\s*:\s*(-?\d+)\b|^\s*steps?\s*[:=-]\s*(-?\d+)\b',
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            steps = 1
            if explicit_steps_match is not None:
                steps = int(next(group for group in explicit_steps_match.groups() if group is not None))
            steps = max(1, min(int(steps), max(1, int(max_hold_steps))))
            metadata["parsed_llm_output"] = {
                "action": int(action),
                "steps": int(steps),
                "fallback_parser": "explicit_action_field",
            }
            return int(action), int(steps), metadata

    action = parse_single_action_from_text(text)
    steps = parse_steps_from_text(text, max_hold_steps)
    metadata["parsed_llm_output"] = {"action": int(action), "steps": int(steps)}
    return int(action), int(steps), metadata


def first_valid_action(valid_masks: np.ndarray, agent_idx: int) -> int:
    valid = np.where(valid_masks[agent_idx] > 0)[0].tolist()
    return int(valid[0]) if valid else 0


def agvs_needing_support_now(
    env,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> set[int]:
    urgent: set[int] = set()
    for row in effective_agv_support_rows(env, persisted_actions, hold_steps_remaining):
        if row["support_need"] == "waiting_for_picker_support_now":
            urgent.add(int(row["agv_idx"]))
    return urgent


def picker_urgent_support_pending(
    env,
    picker_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> bool:
    flags = picker_support_flags(env, picker_idx, persisted_actions, hold_steps_remaining)
    return bool(flags.get("agv_waiting_for_picker_now")) and flags.get("support_target_action_id") is not None


def agent_is_interruptible_when_busy(
    env,
    agent_idx: int,
    latest_env_info: Dict[str, Any] | None = None,
) -> bool:
    latest_env_info = latest_env_info or {}
    agent = env.agents[agent_idx]
    if not bool(getattr(agent, "busy", False)):
        return False
    if bool(getattr(agent, "charging", False)):
        return False
    if current_target_type(env, agent) == "CHARGING":
        return False
    resolving_conflict = list(latest_env_info.get("agent_resolving_conflict", []))
    if agent_idx < len(resolving_conflict) and bool(resolving_conflict[agent_idx]):
        return False
    if bool(getattr(agent, "resolving_conflict", False)):
        return False
    if agent.type.name == "AGV":
        current_flow = next_required_flow_state(env, agent_idx)
        if current_flow in {"wait_for_picker_at_requested_shelf", "wait_for_picker_at_unload_shelf"}:
            return False
    return True


def agv_requires_protected_unload_wait(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> bool:
    if agent_idx < 0 or agent_idx >= len(env.agents):
        return False
    agent = env.agents[agent_idx]
    if agent.type.name != "AGV":
        return False
    if carrying_status_label(env, agent) != "carrying_delivered_shelf":
        return False
    flags = support_flags_for_agv(env, agent_idx, persisted_actions, hold_steps_remaining)
    if str(flags.get("support_type")) != "unload":
        return False
    if not bool(flags.get("waiting_for_picker_support")):
        return False
    if bool(flags.get("picker_at_same_cell")):
        return False
    target_action = int(agent.target) if int(agent.target) > 0 else int(flags.get("support_target_action_id") or 0)
    if target_action <= 0 or classify_action(env, target_action) != "SHELF":
        return False
    return action_distance_steps_for_agent(env, agent, target_action) == 0


def agv_requires_post_arrival_unload_retry(
    env,
    agent_idx: int,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> bool:
    del persisted_actions, hold_steps_remaining
    if agent_idx < 0 or agent_idx >= len(env.agents):
        return False
    agent = env.agents[agent_idx]
    if agent.type.name != "AGV":
        return False
    if carrying_status_label(env, agent) != "carrying_delivered_shelf":
        return False
    target_action = int(agent.target or 0)
    if target_action <= 0 or classify_action(env, target_action) != "SHELF":
        return False
    if action_distance_steps_for_agent(env, agent, target_action) != 0:
        return False
    picker_id = int(env.grid[CollisionLayers.PICKERS, int(agent.y), int(agent.x)])
    return picker_id > 0


def protected_agv_unload_wait_agents(
    env,
    unload_wait_steps_by_agent: List[int] | None = None,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
) -> set[int]:
    if unload_wait_steps_by_agent is None:
        return set()
    protected: set[int] = set()
    for idx, steps in enumerate(unload_wait_steps_by_agent):
        if int(steps) <= 0 or int(steps) >= MIN_AGV_UNLOAD_WAIT_STEPS:
            continue
        if agv_requires_protected_unload_wait(env, idx, persisted_actions, hold_steps_remaining) or agv_requires_post_arrival_unload_retry(
            env, idx, persisted_actions, hold_steps_remaining
        ):
            protected.add(idx)
    return protected


def busy_override_agents_for_replan(
    env,
    max_busy_steps_for_replan: int,
    busy_steps_by_agent: List[int] | None = None,
    latest_env_info: Dict[str, Any] | None = None,
) -> set[int]:
    threshold = int(max(0, max_busy_steps_for_replan))
    if threshold <= 0 or busy_steps_by_agent is None:
        return set()
    forced: set[int] = set()
    for idx, agent in enumerate(env.agents):
        if not bool(getattr(agent, "busy", False)):
            continue
        if idx >= len(busy_steps_by_agent):
            continue
        if int(busy_steps_by_agent[idx]) < threshold:
            continue
        if not agent_is_interruptible_when_busy(env, idx, latest_env_info):
            continue
        forced.add(idx)
    return forced


def order_agents_for_planning(
    env,
    valid_masks: np.ndarray,
    persisted_actions: List[int] | None = None,
    hold_steps_remaining: List[int] | None = None,
    busy_steps_by_agent: List[int] | None = None,
    unload_wait_steps_by_agent: List[int] | None = None,
    max_busy_steps_for_replan: int = 0,
    latest_env_info: Dict[str, Any] | None = None,
) -> List[int]:
    del valid_masks
    urgent_support_agvs = agvs_needing_support_now(env, persisted_actions, hold_steps_remaining)
    busy_override_agents = busy_override_agents_for_replan(
        env,
        max_busy_steps_for_replan=max_busy_steps_for_replan,
        busy_steps_by_agent=busy_steps_by_agent,
        latest_env_info=latest_env_info,
    )
    protected_unload_wait_agents = protected_agv_unload_wait_agents(
        env,
        unload_wait_steps_by_agent=unload_wait_steps_by_agent,
        persisted_actions=persisted_actions,
        hold_steps_remaining=hold_steps_remaining,
    )
    busy_override_agents.difference_update(protected_unload_wait_agents)
    ordering: List[Tuple[int, int]] = []

    for idx, agent in enumerate(env.agents):
        if idx in protected_unload_wait_agents:
            continue
        if bool(agent.busy):
            if idx in busy_override_agents:
                pass
            elif not (
                agent_is_interruptible_on_charger(env, agent)
                and battery_need_label(float(agent.battery)) != "critical"
                and picker_urgent_support_pending(env, idx, persisted_actions, hold_steps_remaining)
            ):
                continue

        priority = 50
        carrying = agent.carrying_shelf is not None

        if agent.type.name == "AGV":
            if idx in urgent_support_agvs:
                priority = 0
            elif carrying:
                priority = 10
            elif float(agent.battery) < 25.0:
                priority = 20
            else:
                priority = 30
        else:
            if urgent_support_agvs:
                priority = 5
            elif float(agent.battery) < 25.0:
                priority = 25
            else:
                priority = 40

        ordering.append((priority, idx))

    ordering.sort(key=lambda row: (row[0], row[1]))
    return [idx for _, idx in ordering]


def charging_station_utilization_snapshot(env) -> Dict[str, Any]:
    total_stations = len(getattr(env, "charging_stations", []))
    if total_stations <= 0:
        return {
            "occupied_stations": 0,
            "total_stations": 0,
            "utilization_rate": 0.0,
        }
    charging_cells = {(int(station.x), int(station.y)) for station in env.charging_stations}
    occupied_stations = sum(1 for agent in env.agents if (int(agent.x), int(agent.y)) in charging_cells)
    return {
        "occupied_stations": int(occupied_stations),
        "total_stations": int(total_stations),
        "utilization_rate": float(occupied_stations / total_stations),
    }


def persisted_action_completed_by_env(
    env,
    agent_idx: int,
    persisted_action: int,
    remaining_hold_steps: int,
    latest_env_info: Dict[str, Any] | None = None,
) -> bool:
    latest_env_info = latest_env_info or {}
    if int(persisted_action) <= 0:
        return False

    agent = env.agents[agent_idx]
    if bool(getattr(agent, "charging", False)):
        if float(getattr(agent, "battery", 0.0)) >= 100.0:
            return True
        return False

    resolution_failed_recently = list(latest_env_info.get("agent_resolution_failed_recently", []))
    unreachable_target_action_ids = list(latest_env_info.get("agent_unreachable_target_action_id", []))
    if agent_idx < len(resolution_failed_recently) and bool(resolution_failed_recently[agent_idx]):
        failed_target = int(unreachable_target_action_ids[agent_idx]) if agent_idx < len(unreachable_target_action_ids) else 0
        return failed_target == 0 or failed_target == int(persisted_action)

    if bool(getattr(agent, "busy", False)):
        return False

    if int(getattr(agent, "target", 0) or 0) != 0:
        return False

    action_coords = env.action_id_to_coords_map.get(int(persisted_action))
    if action_coords is None:
        return False

    target_pos = (int(action_coords[1]), int(action_coords[0]))
    current_pos = (int(agent.x), int(agent.y))
    if current_pos == target_pos:
        return True

    return False


def plan_step_sequential(
    env,
    valid_masks: np.ndarray,
    args,
    model: str,
    persisted_actions: List[int],
    hold_steps_remaining: List[int],
    busy_steps_by_agent: List[int] | None = None,
    unload_wait_steps_by_agent: List[int] | None = None,
    latest_env_info: Dict[str, Any] | None = None,
) -> Tuple[List[int], Dict[str, int], List[Dict[str, Any]]]:
    actions = [0] * env.num_agents
    query_records: List[Dict[str, Any]] = []
    metrics = {
        "llm_calls_this_step": 0,
        "busy_agents_skipped_this_step": 0,
        "action_reuse_this_step": 0,
        "llm_failures_this_step": 0,
        "llm_missing_or_invalid_actions_this_step": 0,
        "forced_requeries_after_invalid_persisted_action": 0,
        "env_conflict_resolution_skips_this_step": 0,
    }

    latest_env_info = latest_env_info or {}
    effective_busy_steps = list(busy_steps_by_agent) if busy_steps_by_agent is not None else [0] * env.num_agents
    effective_unload_wait_steps = (
        list(unload_wait_steps_by_agent) if unload_wait_steps_by_agent is not None else [0] * env.num_agents
    )
    max_busy_steps_for_replan = int(getattr(args, "max_busy_steps_for_replan", 0))
    forced_busy_replan_agents = busy_override_agents_for_replan(
        env,
        max_busy_steps_for_replan=max_busy_steps_for_replan,
        busy_steps_by_agent=effective_busy_steps,
        latest_env_info=latest_env_info,
    )
    protected_unload_wait_agents = protected_agv_unload_wait_agents(
        env,
        unload_wait_steps_by_agent=effective_unload_wait_steps,
        persisted_actions=persisted_actions,
        hold_steps_remaining=hold_steps_remaining,
    )
    forced_busy_replan_agents.difference_update(protected_unload_wait_agents)
    planning_order = order_agents_for_planning(
        env,
        valid_masks,
        persisted_actions,
        hold_steps_remaining,
        effective_busy_steps,
        effective_unload_wait_steps,
        max_busy_steps_for_replan,
        latest_env_info,
    )
    resolving_conflict = list(latest_env_info.get("agent_resolving_conflict", []))
    previous_targets = list(latest_env_info.get("agent_targets", []))
    conflict_resolution_steps = list(latest_env_info.get("agent_conflict_resolution_steps", []))
    agent_req_actions = list(latest_env_info.get("agent_req_actions", []))
    resolution_failed_recently = list(latest_env_info.get("agent_resolution_failed_recently", []))
    unreachable_target_action_ids = list(latest_env_info.get("agent_unreachable_target_action_id", []))
    unreachable_target_reasons = list(latest_env_info.get("agent_unreachable_target_reason", []))
    unreachable_target_cooldowns = list(latest_env_info.get("agent_unreachable_target_cooldown_steps", []))

    for agent_idx in range(env.num_agents):
        failure_active = bool(resolution_failed_recently[agent_idx]) if agent_idx < len(resolution_failed_recently) else False
        if not failure_active:
            continue
        failed_target = int(unreachable_target_action_ids[agent_idx]) if agent_idx < len(unreachable_target_action_ids) else 0
        if int(persisted_actions[agent_idx]) == failed_target:
            persisted_actions[agent_idx] = 0
        hold_steps_remaining[agent_idx] = 0

    for agent_idx in range(env.num_agents):
        persisted = int(persisted_actions[agent_idx]) if agent_idx < len(persisted_actions) else 0
        remaining = int(hold_steps_remaining[agent_idx]) if agent_idx < len(hold_steps_remaining) else 0
        if remaining <= 0:
            continue
        if persisted_action_completed_by_env(env, agent_idx, persisted, remaining, latest_env_info):
            persisted_actions[agent_idx] = 0
            hold_steps_remaining[agent_idx] = 0

    for agent_idx in forced_busy_replan_agents:
        persisted_actions[agent_idx] = 0
        hold_steps_remaining[agent_idx] = 0

    for agent_idx in protected_unload_wait_agents:
        active_target = int(env.agents[agent_idx].target)
        if active_target <= 0:
            active_target = int(persisted_actions[agent_idx]) if agent_idx < len(persisted_actions) else 0
        if active_target > 0:
            persisted_actions[agent_idx] = int(active_target)
            hold_steps_remaining[agent_idx] = max(
                int(hold_steps_remaining[agent_idx]),
                max(0, MIN_AGV_UNLOAD_WAIT_STEPS - int(effective_unload_wait_steps[agent_idx])),
            )

    planning_order = [idx for idx in planning_order if idx not in protected_unload_wait_agents]

    for agent_idx in range(env.num_agents):
        current_agent = env.agents[agent_idx]
        env_owned_execution = bool(current_agent.busy) or (
            agent_idx < len(resolving_conflict) and bool(resolving_conflict[agent_idx])
        )
        if agent_idx in forced_busy_replan_agents:
            env_owned_execution = False
        if agent_idx in protected_unload_wait_agents:
            env_owned_execution = True
        if not env_owned_execution:
            continue
        active_target = int(current_agent.target) if int(current_agent.target) > 0 else (
            int(previous_targets[agent_idx]) if agent_idx < len(previous_targets) else 0
        )
        if active_target > 0:
            actions[agent_idx] = int(active_target)

    for agent_idx in planning_order:
        max_hold_for_agent = max(1, int(args.max_action_hold_steps))
        if env.agents[agent_idx].type.name == "PICKER":
            max_hold_for_agent = min(max_hold_for_agent, max(1, int(args.max_picker_hold_steps)))

        persisted = int(persisted_actions[agent_idx]) if agent_idx < len(persisted_actions) else 0
        remaining = int(hold_steps_remaining[agent_idx]) if agent_idx < len(hold_steps_remaining) else 0
        resolving_now = bool(resolving_conflict[agent_idx]) if agent_idx < len(resolving_conflict) else False
        if resolving_now:
            metrics["busy_agents_skipped_this_step"] += 1
            metrics["env_conflict_resolution_skips_this_step"] += 1
            query_records.append(
                {
                    "agent_id": agent_idx,
                    "agent_type": env.agents[agent_idx].type.name,
                    "prompt": None,
                    "llm_output": None,
                    "raw_llm_output": None,
                    "parsed_llm_output": None,
                    "parsed_action": int(actions[agent_idx]),
                    "parsed_steps": int(max(1, remaining)),
                    "json_generation_failed": False,
                    "json_parse_error": None,
                    "executed_action": int(actions[agent_idx]),
                    "executed_steps": int(max(1, remaining)),
                    "remaining_hold_steps_after_step": int(hold_steps_remaining[agent_idx]),
                    "resolution": "env_owned_conflict_resolution_without_llm_query",
                    "env_conflict_resolution_active": True,
                    "agent_conflict_resolution_steps": int(conflict_resolution_steps[agent_idx]) if agent_idx < len(conflict_resolution_steps) else 0,
                    "agent_req_action": str(agent_req_actions[agent_idx]) if agent_idx < len(agent_req_actions) else None,
                    "suppressed_llm_query_reason": "env_owned_conflict_resolution_active",
                    "resolution_failed_recently": False,
                    "unreachable_target_action_id": 0,
                }
            )
            continue

        if remaining > 0 and 0 <= persisted < env.action_size and valid_masks[agent_idx, persisted] > 0:
            actions[agent_idx] = persisted
            hold_steps_remaining[agent_idx] = max(0, remaining - 1)
            metrics["action_reuse_this_step"] += 1
            query_records.append(
                {
                    "agent_id": agent_idx,
                    "agent_type": env.agents[agent_idx].type.name,
                    "prompt": None,
                    "llm_output": None,
                    "raw_llm_output": None,
                    "parsed_llm_output": None,
                    "parsed_action": int(persisted),
                    "parsed_steps": int(remaining),
                    "json_generation_failed": False,
                    "json_parse_error": None,
                    "executed_action": int(persisted),
                    "executed_steps": int(remaining),
                    "remaining_hold_steps_after_step": int(hold_steps_remaining[agent_idx]),
                    "resolution": "reused_previous_action_without_llm_query",
                    "env_conflict_resolution_active": False,
                    "resolution_failed_recently": bool(resolution_failed_recently[agent_idx]) if agent_idx < len(resolution_failed_recently) else False,
                    "unreachable_target_action_id": int(unreachable_target_action_ids[agent_idx]) if agent_idx < len(unreachable_target_action_ids) else 0,
                    "unreachable_target_reason": unreachable_target_reasons[agent_idx] if agent_idx < len(unreachable_target_reasons) else None,
                }
            )
            continue

        if remaining > 0 and (persisted < 0 or persisted >= env.action_size or valid_masks[agent_idx, persisted] <= 0):
            hold_steps_remaining[agent_idx] = 0
            metrics["forced_requeries_after_invalid_persisted_action"] += 1

        prompt = build_agent_prompt(
            env,
            agent_idx,
            valid_masks,
            args.prompt_format,
            max_hold_for_agent,
            persisted_action=persisted,
            remaining_hold_steps=remaining,
            persisted_actions_all=persisted_actions,
            hold_steps_remaining_all=hold_steps_remaining,
            busy_steps_by_agent=effective_busy_steps,
            busy_override_active=(agent_idx in forced_busy_replan_agents),
        )
        metrics["llm_calls_this_step"] += 1
        llm_text = ""
        parse_metadata: Dict[str, Any] = {
            "parsed_llm_output": None,
            "json_generation_failed": False,
            "json_parse_error": None,
        }
        try:
            llm_text = query_ollama_text(
                model=model,
                ollama_url=args.ollama_url,
                prompt=prompt,
                timeout_s=args.request_timeout_s,
                temperature=args.temperature,
                num_predict=args.num_predict,
                prompt_format=args.prompt_format,
            )
            parsed_action, parsed_steps, parse_metadata = parse_action_and_steps_with_metadata(
                llm_text,
                args.prompt_format,
                max_hold_for_agent,
            )
        except Exception as exc:
            raw_llm_text = llm_text
            llm_text = f"LLM_ERROR: {type(exc).__name__}: {exc}"
            parsed_action = None
            parsed_steps = 1
            metrics["llm_failures_this_step"] += 1
        else:
            raw_llm_text = llm_text

        if (
            parsed_action is None
            or parsed_action < 0
            or parsed_action >= env.action_size
            or valid_masks[agent_idx, parsed_action] <= 0
        ):
            executed_action = first_valid_action(valid_masks, agent_idx)
            metrics["llm_missing_or_invalid_actions_this_step"] += 1
            resolution = "fallback_after_missing_or_invalid_action"
            executed_steps = 1
        else:
            executed_action = int(parsed_action)
            executed_steps = int(parsed_steps)
            resolution = "accepted_llm_action"
            if agent_idx in forced_busy_replan_agents:
                resolution = "busy_override_replanned_after_max_busy_steps"

        actions[agent_idx] = executed_action
        persisted_actions[agent_idx] = int(executed_action)
        hold_steps_remaining[agent_idx] = max(0, int(executed_steps) - 1)
        query_records.append(
            {
                "agent_id": agent_idx,
                "agent_type": env.agents[agent_idx].type.name,
                "prompt": prompt,
                "llm_output": llm_text,
                "raw_llm_output": raw_llm_text,
                "parsed_llm_output": parse_metadata.get("parsed_llm_output"),
                "parsed_action": parsed_action,
                "parsed_steps": int(parsed_steps),
                "json_generation_failed": bool(parse_metadata.get("json_generation_failed")),
                "json_parse_error": parse_metadata.get("json_parse_error"),
                "executed_action": executed_action,
                "executed_steps": int(executed_steps),
                "remaining_hold_steps_after_step": int(hold_steps_remaining[agent_idx]),
                "resolution": resolution,
                "env_conflict_resolution_active": False,
                "busy_override_active": bool(agent_idx in forced_busy_replan_agents),
                "busy_steps_count": int(effective_busy_steps[agent_idx]) if agent_idx < len(effective_busy_steps) else 0,
                "busy_override_threshold": int(max_busy_steps_for_replan),
                "protected_unload_wait_active": bool(agent_idx in protected_unload_wait_agents),
                "unload_wait_steps_count": int(effective_unload_wait_steps[agent_idx]) if agent_idx < len(effective_unload_wait_steps) else 0,
                "min_unload_wait_steps": int(MIN_AGV_UNLOAD_WAIT_STEPS),
                "resolution_failed_recently": bool(resolution_failed_recently[agent_idx]) if agent_idx < len(resolution_failed_recently) else False,
                "unreachable_target_action_id": int(unreachable_target_action_ids[agent_idx]) if agent_idx < len(unreachable_target_action_ids) else 0,
                "unreachable_target_reason": unreachable_target_reasons[agent_idx] if agent_idx < len(unreachable_target_reasons) else None,
                "unreachable_target_cooldown_steps": int(unreachable_target_cooldowns[agent_idx]) if agent_idx < len(unreachable_target_cooldowns) else 0,
                "suppressed_llm_query_reason": None,
            }
        )

    return actions, metrics, query_records


def normalize_reset_output(reset_output: Any) -> Any:
    if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
        return reset_output[0]
    return reset_output


def run_single_episode(
    args,
    model: str,
    model_class: str,
    scenario: ScenarioSpec,
    run_dir: Path,
) -> Dict[str, Any]:
    effective_env_max_steps = None if int(args.max_steps) <= 0 else int(args.max_steps)
    env = gym.make(
        scenario.env_id,
        disable_env_checker=True,
        allow_busy_replan=True,
        max_steps=effective_env_max_steps,
        max_inactivity_steps=args.max_inactivity_steps,
        find_path_agent_aware_always=bool(args.find_path_agent_aware_always),
    )
    log_path = run_dir / "run.log"
    log(f"Starting scenario={scenario.label} env_id={scenario.env_id}", log_path)
    log(f"Scenario focus: {scenario.focus}", log_path)

    reset_output = env.reset(seed=args.seed)
    _ = normalize_reset_output(reset_output)
    base_env = env.unwrapped

    delivery_timeline: List[Dict[str, int]] = []
    step_records: List[Dict[str, Any]] = []
    total_deliveries = 0
    total_llm_calls = 0
    total_action_reuse = 0
    total_llm_failures = 0
    total_llm_missing_or_invalid_actions = 0
    total_json_generation_failures = 0
    total_charging_station_occupied = 0
    total_charging_station_capacity = 0
    persisted_actions = [0] * base_env.num_agents
    hold_steps_remaining = [0] * base_env.num_agents
    busy_steps_by_agent = [0] * base_env.num_agents
    unload_wait_steps_by_agent = [0] * base_env.num_agents
    latest_env_info: Dict[str, Any] | None = None
    start_time = time.time()

    step_idx = 1
    while True:
        valid_masks = safe_valid_action_masks(base_env)
        pre_step_agents = agent_step_snapshot(base_env)
        actions, step_metrics, query_records = plan_step_sequential(
            base_env,
            valid_masks,
            args,
            model,
            persisted_actions,
            hold_steps_remaining,
            busy_steps_by_agent,
            unload_wait_steps_by_agent,
            latest_env_info,
        )

        total_llm_calls += step_metrics["llm_calls_this_step"]
        total_action_reuse += step_metrics["action_reuse_this_step"]
        total_llm_failures += step_metrics["llm_failures_this_step"]
        total_llm_missing_or_invalid_actions += step_metrics["llm_missing_or_invalid_actions_this_step"]
        total_json_generation_failures += sum(
            1 for row in query_records if bool(row.get("json_generation_failed"))
        )

        _, _, terminateds, truncateds, info = env.step(actions)
        latest_env_info = dict(info)
        if args.render:
            env.render()
        post_step_agents = agent_step_snapshot(base_env)
        for idx, agent in enumerate(base_env.agents):
            if bool(agent.busy):
                busy_steps_by_agent[idx] = int(busy_steps_by_agent[idx]) + 1
            else:
                busy_steps_by_agent[idx] = 0
            if agv_requires_protected_unload_wait(
                base_env,
                idx,
                persisted_actions=persisted_actions,
                hold_steps_remaining=hold_steps_remaining,
            ):
                unload_wait_steps_by_agent[idx] = int(unload_wait_steps_by_agent[idx]) + 1
            elif agv_requires_post_arrival_unload_retry(
                base_env,
                idx,
                persisted_actions=persisted_actions,
                hold_steps_remaining=hold_steps_remaining,
            ):
                unload_wait_steps_by_agent[idx] = max(
                    int(unload_wait_steps_by_agent[idx]),
                    int(MIN_AGV_UNLOAD_WAIT_STEPS - 1),
                )
            else:
                unload_wait_steps_by_agent[idx] = 0
        charging_snapshot = charging_station_utilization_snapshot(base_env)
        total_charging_station_occupied += int(charging_snapshot["occupied_stations"])
        total_charging_station_capacity += int(charging_snapshot["total_stations"])

        step_deliveries = int(info.get("shelf_deliveries", 0))
        if step_deliveries > 0:
            total_deliveries += step_deliveries
            delivery_timeline.append(
                {
                    "step": step_idx,
                    "step_shelf_deliveries": step_deliveries,
                    "total_shelf_deliveries": total_deliveries,
                }
            )
            log(
                f"Delivery milestone step={step_idx} delivered_this_step={step_deliveries} cumulative_deliveries={total_deliveries}",
                log_path,
            )

        step_records.append(
            {
                "step": step_idx,
                "actions": actions,
                "step_metrics": step_metrics,
                "agents_before_step": pre_step_agents,
                "agents_after_step": post_step_agents,
                "query_records": query_records,
                "persisted_actions_after_step": [int(v) for v in persisted_actions],
                "hold_steps_remaining_after_step": [int(v) for v in hold_steps_remaining],
                "busy_steps_by_agent_after_step": [int(v) for v in busy_steps_by_agent],
                "unload_wait_steps_by_agent_after_step": [int(v) for v in unload_wait_steps_by_agent],
                "charging_station_utilization": charging_snapshot,
                "info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()},
            }
        )

        if all(terminateds) or all(truncateds):
            break
        if int(args.max_steps) > 0 and step_idx >= int(args.max_steps):
            break
        step_idx += 1

    elapsed_s = time.time() - start_time
    env.close()
    charging_station_utilization_rate = (
        float(total_charging_station_occupied / total_charging_station_capacity)
        if total_charging_station_capacity > 0
        else 0.0
    )
    average_charging_stations_occupied = (
        float(total_charging_station_occupied / max(1, len(step_records)))
        if step_records
        else 0.0
    )

    summary = {
        "stated_objective": STATED_OBJECTIVE,
        "model": model,
        "model_class": model_class,
        "scenario": scenario.label,
        "scenario_env_id": scenario.env_id,
        "scenario_focus": scenario.focus,
        "prompt_format": args.prompt_format.upper(),
        "total_shelf_deliveries": total_deliveries,
        "delivery_timeline": delivery_timeline,
        "steps_executed": len(step_records),
        "max_steps": int(args.max_steps),
        "max_inactivity_steps": int(args.max_inactivity_steps),
        "max_busy_steps_for_replan": int(getattr(args, "max_busy_steps_for_replan", 0)),
        "disable_support_needed_soon": bool(args.disable_support_needed_soon),
        "find_path_agent_aware_always": bool(args.find_path_agent_aware_always),
        "seed": int(args.seed),
        "llm_calls": total_llm_calls,
        "llm_calls_skipped_busy_agents": 0,
        "action_reuse_without_llm_query": total_action_reuse,
        "llm_failures": total_llm_failures,
        "llm_missing_or_invalid_actions": total_llm_missing_or_invalid_actions,
        "json_generation_failures": total_json_generation_failures,
        "charging_station_utilization_rate": charging_station_utilization_rate,
        "average_charging_stations_occupied_per_step": average_charging_stations_occupied,
        "elapsed_seconds": elapsed_s,
        "step_records_path": str(run_dir / "step_records.json"),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "step_records.json").write_text(json.dumps(step_records, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_text_lines = [
        f"Stated Objective: {summary['stated_objective']}",
        f"Model: {summary['model']} ({summary['model_class']})",
        f"Scenario: {summary['scenario']} [{summary['scenario_env_id']}]",
        f"Scenario Focus: {summary['scenario_focus']}",
        f"Prompt format: {summary['prompt_format']}",
        f"Total Shelf Delivery: {summary['total_shelf_deliveries']}",
        f"Steps executed: {summary['steps_executed']}",
        f"LLM calls: {summary['llm_calls']}",
        f"LLM calls skipped because agent was busy: {summary['llm_calls_skipped_busy_agents']}",
        f"Action reuse without LLM query: {summary['action_reuse_without_llm_query']}",
        f"LLM failures: {summary['llm_failures']}",
        f"LLM missing/invalid planned actions: {summary['llm_missing_or_invalid_actions']}",
        f"JSON generation failures: {summary['json_generation_failures']}",
        f"Charging station utilization rate: {summary['charging_station_utilization_rate']:.4f}",
        f"Average charging stations occupied per step: {summary['average_charging_stations_occupied_per_step']:.4f}",
        "Shelf delivery timeline:",
    ]
    if delivery_timeline:
        summary_text_lines.extend(
            f"- step={row['step']} delivered_this_step={row['step_shelf_deliveries']} total_shelf_deliveries={row['total_shelf_deliveries']}"
            for row in delivery_timeline
        )
    else:
        summary_text_lines.append("- none")
    summary_text = "\n".join(summary_text_lines) + "\n"
    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    log(
        f"Completed scenario={scenario.label} model={model} deliveries={total_deliveries} "
        f"steps={len(step_records)} llm_calls={total_llm_calls} action_reuse={total_action_reuse}",
        log_path,
    )
    return summary


def main() -> int:
    args = parser.parse_args()
    parser._parsed_disable_support_needed_soon = bool(args.disable_support_needed_soon)
    results_root = Path(args.results_dir)
    if args.session_dir:
        session_dir = Path(args.session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_dir = results_root / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = session_dir / "aggregate_results.jsonl"

    resume_model_idx = 0
    resume_scenario_idx = 0
    only_model_idx = find_model_index(args.only_model) if args.only_model is not None else None
    only_scenario_idx = find_scenario_index(args.only_scenario) if args.only_scenario is not None else None
    if args.resume_from_model is not None:
        resume_model_idx = find_model_index(args.resume_from_model)
    if args.resume_from_scenario is not None:
        resume_scenario_idx = find_scenario_index(args.resume_from_scenario)

    for model_idx, (model, model_class) in enumerate(MODELS):
        if only_model_idx is not None and model_idx != only_model_idx:
            continue
        if model_idx < resume_model_idx:
            continue
        model_safe = model.replace(":", "_")
        model_log = session_dir / f"{model_safe}.log"
        if args.clean_resumed_model and model_idx == resume_model_idx:
            model_dir = session_dir / model_safe
            if model_dir.exists():
                import shutil

                shutil.rmtree(model_dir)
                log(f"Removed previous model result folder for resumed model: {model}", model_log)

        log(f"Pulling model={model} class={model_class}", model_log)
        pull_model(model)
        try:
            for scenario_idx, scenario in enumerate(SCENARIOS):
                if only_scenario_idx is not None and scenario_idx != only_scenario_idx:
                    continue
                if model_idx == resume_model_idx and scenario_idx < resume_scenario_idx:
                    continue
                run_dir = session_dir / model_safe / scenario.label / args.prompt_format
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.clean_resumed_model and model_idx == resume_model_idx and scenario_idx >= resume_scenario_idx:
                    import shutil

                    shutil.rmtree(run_dir)
                    run_dir.mkdir(parents=True, exist_ok=True)
                summary = run_single_episode(args, model, model_class, scenario, run_dir)
                if args.clean_resumed_model:
                    remove_prior_aggregate_entries(aggregate_path, model, scenario.label)
                append_jsonl(aggregate_path, summary)
        finally:
            log(f"Removing model={model} to save disk space", model_log)
            remove_model(model)

    print(f"Finished. Aggregated results written to {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
