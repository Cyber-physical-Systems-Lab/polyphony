import json
import re
import subprocess
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request

import gymnasium as gym
import numpy as np
import tarware

from fixed_prompt_builder import render_template
from state_translation_helper import (
    candidate_ids_for_prompt,
    charging_station_occupancy,
    classify_action,
    describe_action_id_for_agent,
    get_requested_shelves,
    render_self_state,
    waiting_agv_support_lines,
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


STATED_OBJECTIVE = "Increase the total shelf deliveries."


SHARED_CONTEXT_TEXT_TEMPLATE = (
    "Main objective: {stated_objective}\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load moves to requested shelf.\n"
    "- Picker and AGV meet at that shelf for loading.\n"
    "- AGV carries shelf to GOAL.\n"
    "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.\n"
    "\n"
    "Shared coordination context:\n"
    "Requested shelves:\n{requested_shelves}\n"
    "\n"
    "Charging place occupancy:\n{charging_occupancy}\n"
    "\n"
    "AGVs requesting picker support:\n{agv_support_requests}\n"
)


BASIC_SUMMARY_LINES = [
    "Multi-agent warehouse with AGV and Picker collaboration.",
    "Avoid any out-of-charge scenario.",
    "Use charging strategically so throughput does not collapse.",
    "AGV-Picker coordinate at shelf interactions (load/unload at shelf location).",
]


TYPICAL_FLOW_LINES = [
    "AGV without load moves to requested shelf.",
    "Picker and AGV meet at that shelf for loading.",
    "AGV carries shelf to GOAL.",
    "AGV returns shelf to storage shelf location with Picker support for unload.",
    "Repeat with next request.",
]


BATTERY_MODEL_LINES = [
    "Move step consumption = 1.",
    "Load/Unload consumption = 2.",
    "Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.",
]


AGV_LANGUAGE_TEMPLATE = (
    "{shared_context}\n"
    "You are planning only for one AGV.\n"
    "Current agent:\n"
    "- Agent: agent_{agent_id} (AGV)\n"
    "- Self state: {self_state}\n"
    "\n"
    "AGV decision rules:\n"
    "- If carrying a requested shelf, prioritize GOAL.\n"
    "- If battery is critical, prioritize charging.\n"
    "- If you are currently waiting for Picker support, do not abandon a productive support shelf unless battery is critical.\n"
    "- Use only the listed candidate action ids.\n"
    "\n"
    "Candidate actions:\n"
    "{candidate_actions}\n"
    "\n"
    "Respond in plain text only:\n"
    "Reasoning: <short>\n"
    "Action: <action_id>\n"
)


PICKER_LANGUAGE_TEMPLATE = (
    "{shared_context}\n"
    "You are planning only for one Picker.\n"
    "Current agent:\n"
    "- Agent: agent_{agent_id} (PICKER)\n"
    "- Self state: {self_state}\n"
    "\n"
    "Picker decision rules:\n"
    "- Highest priority is AGV support needed now.\n"
    "- If any AGV is waiting for support now, prioritize moving to that support shelf when valid.\n"
    "- If battery is critical, prioritize charging.\n"
    "- Avoid unrelated shelf moves when AGV support is pending.\n"
    "- Use only the listed candidate action ids.\n"
    "\n"
    "Candidate actions:\n"
    "{candidate_actions}\n"
    "\n"
    "Respond in plain text only:\n"
    "Reasoning: <short>\n"
    "Action: <action_id>\n"
)


AGV_JSON_TEMPLATE = (
    "Main objective: Increase the total shelf deliveries.\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load moves to requested shelf.\n"
    "- Picker and AGV meet at that shelf for loading.\n"
    "- AGV carries shelf to GOAL.\n"
    "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.\n"
    "\n"
    "You are planning only for one AGV.\n"
    "Choose exactly one action for the current AGV using the shared global coordination context and the current-agent candidate actions.\n"
    "AGV decision rules:\n"
    "- If carrying a requested shelf, prioritize GOAL.\n"
    "- If battery is critical, prioritize charging.\n"
    "- If currently waiting for Picker support, do not abandon a productive support shelf unless battery is critical.\n"
    "- Use only candidate action ids listed for planning_context.current_agent.candidate_actions.\n"
    "\n"
    "Shared planning context in JSON:\n"
    "{prompt_json}\n"
    "\n"
    "The planning context ends above.\n"
    "Now return exactly one JSON object and nothing else.\n"
    "Do not repeat the planning context.\n"
    "Do not output keys other than reasoning and action.\n"
    "Do not output markdown.\n"
    "The action must be one integer from planning_context.current_agent.candidate_actions.\n"
    "Required response shape:\n"
    "{{\"reasoning\":\"short\",\"action\":12}}\n"
)


PICKER_JSON_TEMPLATE = (
    "Main objective: Increase the total shelf deliveries.\n"
    "Basic summary:\n"
    "- Multi-agent warehouse with AGV and Picker collaboration.\n"
    "- Avoid any out-of-charge scenario.\n"
    "- Use charging strategically so throughput does not collapse.\n"
    "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
    "\n"
    "Typical flow:\n"
    "- AGV without load moves to requested shelf.\n"
    "- Picker and AGV meet at that shelf for loading.\n"
    "- AGV carries shelf to GOAL.\n"
    "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
    "- Repeat with next request.\n"
    "\n"
    "Battery model:\n"
    "- Move step consumption = 1.\n"
    "- Load/Unload consumption = 2.\n"
    "- Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.\n"
    "\n"
    "You are planning only for one Picker.\n"
    "Choose exactly one action for the current Picker using the shared global coordination context and the current-agent candidate actions.\n"
    "Picker decision rules:\n"
    "- Highest priority is AGV support needed now.\n"
    "- If any AGV is waiting for support now, prioritize moving to that support shelf when valid.\n"
    "- If battery is critical, prioritize charging.\n"
    "- Avoid unrelated shelf moves when AGV support is pending.\n"
    "- Use only candidate action ids listed for planning_context.current_agent.candidate_actions.\n"
    "\n"
    "Shared planning context in JSON:\n"
    "{prompt_json}\n"
    "\n"
    "The planning context ends above.\n"
    "Now return exactly one JSON object and nothing else.\n"
    "Do not repeat the planning context.\n"
    "Do not output keys other than reasoning and action.\n"
    "Do not output markdown.\n"
    "The action must be one integer from planning_context.current_agent.candidate_actions.\n"
    "Required response shape:\n"
    "{{\"reasoning\":\"short\",\"action\":12}}\n"
)


parser = ArgumentParser(
    description="Run sequential shared-context LLM experiments across models and warehouse scenarios",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=0, help="Base episode seed")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
parser.add_argument("--prompt_format", choices=["language", "json"], default="language", help="Prompt/output format")
parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama generate endpoint")
parser.add_argument("--request_timeout_s", type=int, default=60, help="HTTP timeout for Ollama generation")
parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
parser.add_argument("--num_predict", type=int, default=700, help="Generation token budget")
parser.add_argument("--results_dir", type=str, default="results/shared_context_llm_experiments", help="Directory for logs and summaries")
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


def query_ollama_text(
    model: str,
    ollama_url: str,
    prompt: str,
    timeout_s: int,
    temperature: float,
    num_predict: int,
    prompt_format: str,
    retry_500_count: int = 5,
    retry_500_backoff_s: float = 1.0,
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
    total_attempts = max(1, int(retry_500_count) + 1)
    for attempt_idx in range(total_attempts):
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return str(body.get("response", "")).strip()
        except error.HTTPError as exc:
            if exc.code != 500 or attempt_idx + 1 >= total_attempts:
                raise
            if retry_500_backoff_s > 0:
                time.sleep(float(retry_500_backoff_s) * (2**attempt_idx))
    raise RuntimeError("Unreachable: exhausted Ollama retries without returning or raising")


def pull_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def remove_model(model: str) -> None:
    subprocess.run(["ollama", "rm", model], check=True)


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


def build_shared_context(env) -> str:
    fields = {
        "stated_objective": STATED_OBJECTIVE,
        "requested_shelves": format_bullets(get_requested_shelves(env)),
        "charging_occupancy": format_bullets(charging_station_occupancy(env)),
        "agv_support_requests": format_bullets(waiting_agv_support_lines(env)),
    }
    return render_template(SHARED_CONTEXT_TEXT_TEMPLATE, fields)


def build_shared_context_payload(env) -> Dict[str, Any]:
    return {
        "stated_objective": STATED_OBJECTIVE,
        "basic_summary": BASIC_SUMMARY_LINES,
        "typical_flow": TYPICAL_FLOW_LINES,
        "battery_model": BATTERY_MODEL_LINES,
        "shared_coordination_context": {
            "requested_shelves": get_requested_shelves(env),
            "charging_place_occupancy": charging_station_occupancy(env),
            "agvs_requesting_picker_support": waiting_agv_support_lines(env),
        },
    }


def build_candidate_views(env, agent_idx: int, valid_masks: np.ndarray) -> Tuple[str, str]:
    agent = env.agents[agent_idx]
    candidate_ids = candidate_ids_for_prompt(env, agent_idx, valid_masks, 0)
    candidate_lines = [
        describe_action_id_for_agent(env, agent, action_id)
        for action_id in candidate_ids
        if valid_masks[agent_idx, action_id] > 0
    ]
    if not candidate_lines:
        candidate_lines = ["0:NOOP (distance_steps=0)"]
    text_view = "\n".join(f"- {line}" for line in candidate_lines)
    json_view = json.dumps(candidate_lines, indent=2, ensure_ascii=True)
    return text_view, json_view


def build_structured_candidate_actions(env, agent_idx: int, valid_masks: np.ndarray) -> List[Dict[str, Any]]:
    agent = env.agents[agent_idx]
    candidate_ids = candidate_ids_for_prompt(env, agent_idx, valid_masks, 0)
    rows: List[Dict[str, Any]] = []
    for action_id in candidate_ids:
        if valid_masks[agent_idx, action_id] <= 0:
            continue
        coords = env.action_id_to_coords_map.get(action_id) if action_id > 0 else None
        distance_steps = 0
        if action_id > 0 and coords is not None:
            path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
            distance_steps = max(1, len(path)) if path else 9999
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
                "description": describe_action_id_for_agent(env, agent, int(action_id)),
            }
        )
    if not rows:
        rows.append(
            {
                "action_id": 0,
                "action_type": "NOOP",
                "target_position": None,
                "distance_steps": 0,
                "description": "0:NOOP (distance_steps=0)",
            }
        )
    return rows


def build_agent_prompt(env, agent_idx: int, valid_masks: np.ndarray, prompt_format: str) -> str:
    agent = env.agents[agent_idx]
    shared_context = build_shared_context(env)
    candidate_actions_text, candidate_actions_json = build_candidate_views(env, agent_idx, valid_masks)
    structured_candidate_actions = build_structured_candidate_actions(env, agent_idx, valid_masks)
    agent_state_json = json.dumps(
        {
            "agent_id": agent_idx,
            "agent_type": agent.type.name,
            "self_state": render_self_state(env, agent),
        },
        indent=2,
        ensure_ascii=True,
    )
    fields = {
        "shared_context": shared_context,
        "agent_id": agent_idx,
        "self_state": render_self_state(env, agent),
        "candidate_actions": candidate_actions_text,
        "agent_state_json": agent_state_json,
        "candidate_actions_json": candidate_actions_json,
        "prompt_json": json.dumps(
            {
                "planning_context": {
                    **build_shared_context_payload(env),
                    "current_agent": {
                        "agent_id": agent_idx,
                        "agent_type": agent.type.name,
                        "self_state": render_self_state(env, agent),
                        "candidate_actions": structured_candidate_actions,
                    },
                    "role_specific_rules": (
                        [
                            "If carrying a requested shelf, prioritize GOAL.",
                            "If battery is critical, prioritize charging.",
                            "If you are currently waiting for Picker support, do not abandon a productive support shelf unless battery is critical.",
                        ]
                        if agent.type.name == "AGV"
                        else [
                            "Highest priority is AGV support needed now.",
                            "If any AGV is waiting for support now, prioritize moving to that support shelf when valid.",
                            "If battery is critical, prioritize charging.",
                            "Avoid unrelated shelf moves when AGV support is pending.",
                        ]
                    ),
                }
            },
            indent=2,
            ensure_ascii=True,
        ),
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


def parse_single_action_with_metadata(text: str, prompt_format: str) -> Tuple[int, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "parsed_llm_output": None,
        "json_generation_failed": False,
        "json_parse_error": None,
    }
    if prompt_format == "json":
        try:
            payload = json.loads(extract_json_object_text(text))
            metadata["parsed_llm_output"] = payload
            return int(payload.get("action", 0)), metadata
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
            metadata["parsed_llm_output"] = {
                "action": int(action),
                "fallback_parser": "explicit_action_field",
            }
            return int(action), metadata

    action = parse_single_action_from_text(text)
    metadata["parsed_llm_output"] = {"action": int(action)}
    return int(action), metadata


def discard_invalid_action(valid_masks: np.ndarray, agent_idx: int) -> int:
    if 0 <= agent_idx < valid_masks.shape[0] and valid_masks[agent_idx, 0] > 0:
        return 0
    valid = np.where(valid_masks[agent_idx] > 0)[0].tolist()
    return int(valid[0]) if valid else 0


def agvs_needing_support_now(env) -> set[int]:
    urgent: set[int] = set()
    picker_positions = {(int(agent.x), int(agent.y)) for agent in env.agents if agent.type.name == "PICKER"}
    for idx, agv in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agv.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        if not bool(agv.busy) or coords is None:
            continue
        at_target = (int(agv.x), int(agv.y)) == (int(coords[1]), int(coords[0]))
        if at_target and (int(agv.x), int(agv.y)) not in picker_positions:
            urgent.add(idx)
    return urgent


def order_agents_for_planning(env, valid_masks: np.ndarray) -> List[int]:
    del valid_masks
    urgent_support_agvs = agvs_needing_support_now(env)
    ordering: List[Tuple[int, int]] = []

    for idx, agent in enumerate(env.agents):
        if bool(agent.busy):
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


def plan_step_sequential(
    env,
    valid_masks: np.ndarray,
    args,
    model: str,
) -> Tuple[List[int], Dict[str, int], List[Dict[str, Any]]]:
    actions = [0] * env.num_agents
    query_records: List[Dict[str, Any]] = []
    metrics = {
        "llm_calls_this_step": 0,
        "busy_agents_skipped_this_step": 0,
        "llm_failures_this_step": 0,
        "llm_missing_or_invalid_actions_this_step": 0,
    }

    for agent in env.agents:
        if bool(agent.busy):
            metrics["busy_agents_skipped_this_step"] += 1

    planning_order = order_agents_for_planning(env, valid_masks)

    for agent_idx in planning_order:
        prompt = build_agent_prompt(env, agent_idx, valid_masks, args.prompt_format)
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
            parsed_action, parse_metadata = parse_single_action_with_metadata(llm_text, args.prompt_format)
        except Exception as exc:
            raw_llm_text = llm_text
            llm_text = f"LLM_ERROR: {type(exc).__name__}: {exc}"
            parsed_action = None
            metrics["llm_failures_this_step"] += 1
        else:
            raw_llm_text = llm_text

        if (
            parsed_action is None
            or parsed_action < 0
            or parsed_action >= env.action_size
            or valid_masks[agent_idx, parsed_action] <= 0
        ):
            executed_action = discard_invalid_action(valid_masks, agent_idx)
            metrics["llm_missing_or_invalid_actions_this_step"] += 1
            resolution = "discarded_invalid_action_to_noop"
        else:
            executed_action = int(parsed_action)
            resolution = "accepted_llm_action"

        actions[agent_idx] = executed_action
        query_records.append(
            {
                "agent_id": agent_idx,
                "agent_type": env.agents[agent_idx].type.name,
                "prompt": prompt,
                "llm_output": llm_text,
                "raw_llm_output": raw_llm_text,
                "parsed_llm_output": parse_metadata.get("parsed_llm_output"),
                "parsed_action": parsed_action,
                "json_generation_failed": bool(parse_metadata.get("json_generation_failed")),
                "json_parse_error": parse_metadata.get("json_parse_error"),
                "executed_action": executed_action,
                "resolution": resolution,
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
    env = gym.make(scenario.env_id, disable_env_checker=True)
    log_path = run_dir / "run.log"
    log(f"Starting scenario={scenario.label} env_id={scenario.env_id}", log_path)
    log(f"Scenario focus: {scenario.focus}", log_path)

    reset_output = env.reset(seed=args.seed)
    _ = normalize_reset_output(reset_output)

    delivery_timeline: List[Dict[str, int]] = []
    step_records: List[Dict[str, Any]] = []
    total_deliveries = 0
    total_llm_calls = 0
    total_busy_skipped_calls = 0
    total_llm_failures = 0
    total_llm_missing_or_invalid_actions = 0
    total_json_generation_failures = 0
    start_time = time.time()

    for step_idx in range(1, max(1, int(args.max_steps)) + 1):
        base_env = env.unwrapped
        valid_masks = safe_valid_action_masks(base_env)
        pre_step_agents = agent_step_snapshot(base_env)
        actions, step_metrics, query_records = plan_step_sequential(base_env, valid_masks, args, model)

        total_llm_calls += step_metrics["llm_calls_this_step"]
        total_busy_skipped_calls += step_metrics["busy_agents_skipped_this_step"]
        total_llm_failures += step_metrics["llm_failures_this_step"]
        total_llm_missing_or_invalid_actions += step_metrics["llm_missing_or_invalid_actions_this_step"]
        total_json_generation_failures += sum(
            1 for row in query_records if bool(row.get("json_generation_failed"))
        )

        _, _, terminateds, truncateds, info = env.step(actions)
        if args.render:
            env.render()
        post_step_agents = agent_step_snapshot(base_env)

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
                "info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()},
            }
        )

        if all(terminateds) or all(truncateds):
            break

    elapsed_s = time.time() - start_time
    env.close()

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
        "seed": int(args.seed),
        "llm_calls": total_llm_calls,
        "llm_calls_skipped_busy_agents": total_busy_skipped_calls,
        "llm_failures": total_llm_failures,
        "llm_missing_or_invalid_actions": total_llm_missing_or_invalid_actions,
        "json_generation_failures": total_json_generation_failures,
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
        f"LLM failures: {summary['llm_failures']}",
        f"LLM missing/invalid planned actions: {summary['llm_missing_or_invalid_actions']}",
        f"JSON generation failures: {summary['json_generation_failures']}",
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
        f"steps={len(step_records)} llm_calls={total_llm_calls} busy_skips={total_busy_skipped_calls}",
        log_path,
    )
    return summary


def main() -> int:
    args = parser.parse_args()
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
