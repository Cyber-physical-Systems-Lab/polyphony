import json
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

import gymnasium as gym
import numpy as np

from tarware.heuristic import heuristic_episode
from tarware.warehouse import AgentType


parser = ArgumentParser(
    description="Run plain-language LLM control on Battery-TA-RWARE",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_id", default="tarware-tiny-1agvs-1pickers-partialobs-chg-v1", type=str, help="Gym environment id")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes")
parser.add_argument("--seed", default=0, type=int, help="Base seed")
parser.add_argument("--model", default="llama3.2:3b", type=str, help="Ollama model name")
parser.add_argument(
    "--experiment_mode",
    default="legacy_rich",
    choices=["legacy_rich", "fixed_prompt_action", "agent_type_prompt", "message_or_action"],
    help="Prompting strategy mode",
)
parser.add_argument(
    "--config_path",
    default="",
    type=str,
    help="Optional JSON experiment config with prompt templates",
)
parser.add_argument(
    "--llm_backend",
    default="native",
    choices=["native", "langchain"],
    help="LLM query backend: native urllib call or LangChain Ollama integration",
)
parser.add_argument("--ollama_url", default="http://localhost:11434/api/generate", type=str, help="Ollama generate endpoint")
parser.add_argument(
    "--ollama_base_url",
    default="",
    type=str,
    help="Ollama base URL for LangChain backend, e.g. http://localhost:11434 (default derives from --ollama_url)",
)
parser.add_argument("--request_timeout_s", default=45, type=int, help="HTTP timeout")
parser.add_argument("--max_steps_per_episode", default=250, type=int, help="Safety cap for episode length")
parser.add_argument("--decision_interval", default=1, type=int, help="Query LLM every N steps")
parser.add_argument(
    "--force_llm_replan_steps",
    default=20,
    type=int,
    help="Force an LLM replan every N steps even if no event requires replanning",
)
parser.add_argument("--progress_every", default=10, type=int, help="Print progress every N steps")
parser.add_argument(
    "--max_action_hold_steps",
    default=10,
    type=int,
    help="Max number of steps an agent may keep the same LLM-decided action without re-query",
)
parser.add_argument(
    "--max_picker_hold_steps",
    default=3,
    type=int,
    help="Max hold steps for picker agents (applied even if max_action_hold_steps is higher)",
)
parser.add_argument(
    "--max_candidate_ids",
    default=0,
    type=int,
    help="Max candidate action ids per agent in prompt (0 means all valid actions)",
)
parser.add_argument(
    "--log_text_chars",
    default=5000,
    type=int,
    help="Max chars for PROMPT/LLM_OUTPUT logs (0 = no truncation)",
)
parser.add_argument("--render", action="store_true")
parser.add_argument("--skip_heuristic", action="store_true", help="Only run LLM policy")
parser.add_argument(
    "--warehouse_event_logs",
    action="store_true",
    help="Enable warehouse-level [COLLAB]/[DELIVERY] prints from the environment",
)
parser.add_argument(
    "--rag_db_dir",
    default="rag_db",
    type=str,
    help="Persistent vector DB directory for RAG index",
)
parser.add_argument(
    "--rag_docs_dir",
    default="knowledge",
    type=str,
    help="Knowledge documents directory (used when rebuilding index)",
)
parser.add_argument(
    "--rag_top_k",
    default=3,
    type=int,
    help="Number of retrieved chunks to inject into prompts",
)
parser.add_argument(
    "--rag_max_chars",
    default=3000,
    type=int,
    help="Max chars of retrieved context injected into each prompt",
)
parser.add_argument(
    "--rag_embedding_model",
    default="nomic-embed-text",
    type=str,
    help="Ollama embedding model used by RAG",
)
parser.add_argument(
    "--rag_rebuild_index",
    action="store_true",
    help="Rebuild RAG index from rag_docs_dir before run",
)
parser.add_argument(
    "--enable_episode_reflection",
    action="store_true",
    help="After each episode, use an LLM to generate a knowledge note from transcript",
)
parser.add_argument(
    "--reflection_model",
    default="",
    type=str,
    help="Model for episode reflection (default: same as --model)",
)
parser.add_argument(
    "--reflection_backend",
    default="auto",
    choices=["auto", "native", "langchain"],
    help="Backend for reflection LLM calls (auto uses --llm_backend)",
)
parser.add_argument(
    "--reflection_timeout_s",
    default=60,
    type=int,
    help="Timeout for reflection LLM calls",
)
parser.add_argument(
    "--reflection_max_chars",
    default=30000,
    type=int,
    help="Max transcript chars passed to reflection LLM",
)
parser.add_argument(
    "--reflection_notes_dir",
    default="knowledge/reflections",
    type=str,
    help="Directory where generated reflection markdown files are written",
)
parser.add_argument(
    "--transcript_dir",
    default="transcripts",
    type=str,
    help="Directory where episode transcript JSON files are written",
)
parser.add_argument(
    "--transcript_text_chars",
    default=2000,
    type=int,
    help="Max chars for prompt/output text stored in transcript (0 = no truncation)",
)
parser.add_argument(
    "--episode_results_path",
    default="results/llm_episode_results.jsonl",
    type=str,
    help="Path to newline-delimited JSON file where per-episode LLM results are appended",
)
parser.add_argument(
    "--episode_summary_dir",
    default="results/episode_summaries",
    type=str,
    help="Directory where per-episode human-readable summary files are written",
)
parser.add_argument(
    "--reflection_skip_rag_rebuild",
    action="store_true",
    help="Do not rebuild RAG after writing reflection notes",
)


def log_block(tag: str, text: str) -> None:
    print(f"\n[{tag}]\n{text}\n")

def maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return text[:max_chars] + f"\n...[TRUNCATED {remaining} chars]"


def default_experiment_config() -> Dict[str, Any]:
    return {
        "templates": {
            "fixed_prompt_action": (
                "You are controlling one warehouse agent.\n"
                "Goal: maximize deliveries while keeping batteries safe.\n"
                "Use only valid candidate action ids.\n"
                "Choose a suitable Steps value.\n"
                "Fewer LLM calls improve performance, so use a larger Steps value when the plan is stable and unlikely to need revision.\n"
                "But do not overcommit: use a smaller Steps value when the environment may change soon, coordination may be needed, battery risk is rising, or urgent actions may appear.\n"
                "Step: {step}\n"
                "Agent: agent_{agent_id} ({agent_type})\n"
                "Self state: {self_state}\n"
                "Requested shelves:\n{requested_shelves}\n"
                "All agents:\n{all_agents}\n"
                "Candidate actions:\n{candidate_actions}\n"
                "Respond in plain text only:\n"
                "Reasoning: <short>\n"
                "Action: <action_id>\n"
                "Steps: <1..{max_hold_steps}>\n"
            ),
            "agent_type_prompt": {
                "AGV": (
                    "You are an AGV in a warehouse.\n"
                    "Priorities: deliver requested shelves, then return shelves, then charge when needed.\n"
                    "Choose a suitable Steps value.\n"
                    "Fewer LLM calls improve performance, so use a larger Steps value when the plan is stable and unlikely to need revision.\n"
                    "But do not overcommit: use a smaller Steps value when the environment may change soon, coordination may be needed, battery risk is rising, or urgent actions may appear.\n"
                    "Step: {step}\n"
                    "Self: {self_state}\n"
                    "Requested shelves:\n{requested_shelves}\n"
                    "All agents:\n{all_agents}\n"
                    "Candidate actions:\n{candidate_actions}\n"
                    "Respond in plain text only:\n"
                    "Reasoning: <short>\n"
                    "Action: <action_id>\n"
                    "Steps: <1..{max_hold_steps}>\n"
                ),
                "PICKER": (
                    "You are a Picker in a warehouse.\n"
                    "Priorities: support AGVs at shelf interactions, avoid unnecessary charging.\n"
                    "Choose a suitable Steps value.\n"
                    "Fewer LLM calls improve performance, so use a larger Steps value when the plan is stable and unlikely to need revision.\n"
                    "But do not overcommit: use a smaller Steps value when the environment may change soon, coordination may be needed, battery risk is rising, or urgent actions may appear.\n"
                    "Step: {step}\n"
                    "Self: {self_state}\n"
                    "AGV support view:\n{picker_agv_tasks}\n"
                    "All agents:\n{all_agents}\n"
                    "Candidate actions:\n{candidate_actions}\n"
                    "Respond in plain text only:\n"
                    "Reasoning: <short>\n"
                    "Action: <action_id>\n"
                    "Steps: <1..{max_hold_steps}>\n"
                ),
                "default": (
                    "You are controlling one warehouse agent.\n"
                    "Choose a suitable Steps value.\n"
                    "Fewer LLM calls improve performance, so use a larger Steps value when the plan is stable and unlikely to need revision.\n"
                    "But do not overcommit: use a smaller Steps value when the environment may change soon, coordination may be needed, battery risk is rising, or urgent actions may appear.\n"
                    "Step: {step}\n"
                    "Self state: {self_state}\n"
                    "Candidate actions:\n{candidate_actions}\n"
                    "Respond in plain text only:\n"
                    "Reasoning: <short>\n"
                    "Action: <action_id>\n"
                    "Steps: <1..{max_hold_steps}>\n"
                ),
            },
            "message_or_action": (
                "You are controlling one warehouse agent.\n"
                "At this step you must choose exactly one primary decision:\n"
                "- ACTION: take one valid environment action now.\n"
                "- MESSAGE: send one coordination message to another agent now.\n"
                "Choose MESSAGE when communication is more valuable than immediate movement, such as support-request timing, "
                "plan-intent coordination, battery coordination, or shelf handoff coordination.\n"
                "Choose ACTION when immediate execution is more valuable than messaging.\n"
                "If you choose ACTION, the action id must be one of the valid candidate action ids.\n"
                "If you choose MESSAGE, do not invent an action id; leave the Action line blank.\n"
                "Choose a suitable Steps value.\n"
                "Fewer LLM calls improve performance, so use a larger Steps value when the chosen action is stable and unlikely to need revision.\n"
                "But do not overcommit: use a smaller Steps value when message handling, support timing, battery changes, or environment changes may require a quick replanning response.\n"
                "Step: {step}\n"
                "Agent: agent_{agent_id} ({agent_type})\n"
                "Self state: {self_state}\n"
                "Pending messages for you:\n{inbox_messages}\n"
                "Current priority message:\n{active_message}\n"
                "Recently handled messages:\n{recently_handled_messages}\n"
                "Requested shelves:\n{requested_shelves}\n"
                "All agents:\n{all_agents}\n"
                "Candidate actions:\n{candidate_actions}\n"
                "Respond in plain text only.\n"
                "Use EXACTLY this field order so the output can be parsed:\n"
                "Reasoning: <1-2 short sentences on why you chose ACTION or MESSAGE>\n"
                "Decision: <ACTION or MESSAGE>\n"
                "Decision Detail: <why this is better than the other option right now>\n"
                "Focus Message: <message_id to work on now, or blank if no message is relevant>\n"
                "Message Status: <KEEP, ACTIVE, or HANDLED>\n"
                "Status Detail: <why the chosen message should stay pending, become active, or be marked handled>\n"
                "Action: <valid action_id for ACTION, blank for MESSAGE>\n"
                "Action Detail: <why this specific action fits the current state, or write none for MESSAGE>\n"
                "Steps: <1..{max_hold_steps}>\n"
                "To: <agent_id as integer, or blank for ACTION>\n"
                "Message: <short coordination message, or blank for ACTION>\n"
                "Message Detail: <why this specific message helps coordination now, or write none for ACTION>\n"
                "Rules:\n"
                "- Output exactly one block with the twelve fields above.\n"
                "- Do not add bullets, code fences, JSON, or extra commentary.\n"
                "- If a pending message should remain in progress, set Focus Message to its id and Message Status to ACTIVE.\n"
                "- If a pending message has been fully dealt with or is no longer needed, set Focus Message to its id and Message Status to HANDLED.\n"
                "- If no pending message should change state this turn, leave Focus Message blank and set Message Status to KEEP.\n"
                "- For ACTION: fill Action and Steps, leave To and Message blank, set Message Detail to none.\n"
                "- For MESSAGE: fill To and Message, leave Action blank, set Action Detail to none.\n"
                "- Keep Message concise and operational.\n"
            ),
        },
        "message_settings": {
            "inbox_max_messages": 50,
            "inbox_prompt_messages": 20,
            "message_default_action": "noop",
        },
        "reflection_settings": {
            "system_goal": "Increase total shelf deliveries through proactive AGV/Picker collaboration.",
            "delivery_metric": "shelf_deliveries",
            "message_samples": 12,
            "delivery_after_message_window_steps": 6,
        },
    }


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    cfg = default_experiment_config()
    if not config_path.strip():
        return cfg
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config_path not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    if not isinstance(user_cfg, dict):
        raise ValueError("config_path JSON must be an object at top-level")
    templates = user_cfg.get("templates")
    if isinstance(templates, dict):
        cfg_templates = cfg.setdefault("templates", {})
        for k, v in templates.items():
            cfg_templates[k] = v
    for key in ("message_settings", "reflection_settings"):
        user_obj = user_cfg.get(key)
        if isinstance(user_obj, dict):
            base = cfg.setdefault(key, {})
            if isinstance(base, dict):
                base.update(user_obj)
    return cfg


class _TemplateSafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render_template(template: str, fields: Dict[str, Any]) -> str:
    return template.format_map(_TemplateSafeDict(**fields))


def message_settings_from_config(experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_settings = default_experiment_config().get("message_settings", {})
    user_settings = experiment_cfg.get("message_settings", {})
    merged = dict(default_settings)
    if isinstance(user_settings, dict):
        merged.update(user_settings)
    return merged


def reflection_settings_from_config(experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_settings = default_experiment_config().get("reflection_settings", {})
    user_settings = experiment_cfg.get("reflection_settings", {})
    merged = dict(default_settings)
    if isinstance(user_settings, dict):
        merged.update(user_settings)
    return merged


def info_statistics(infos: List[Dict[str, Any]], global_episode_return: float, episode_returns: np.ndarray) -> Dict[str, Any]:
    total_deliveries = 0
    total_clashes = 0
    total_stuck = 0
    total_agvs_distance = 0
    total_pickers_distance = 0
    for info in infos:
        total_deliveries += info.get("shelf_deliveries", 0)
        total_clashes += info.get("clashes", 0)
        total_stuck += info.get("stucks", 0)
        total_agvs_distance += info.get("agvs_distance_travelled", 0)
        total_pickers_distance += info.get("pickers_distance_travelled", 0)
        info["total_deliveries"] = total_deliveries
        info["total_clashes"] = total_clashes
        info["total_stuck"] = total_stuck
        info["total_agvs_distance"] = total_agvs_distance
        info["total_pickers_distance"] = total_pickers_distance

    last_info = infos[-1] if infos else {}
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = float(global_episode_return)
    last_info["episode_returns"] = episode_returns
    if infos and len(infos) > 0:
        last_info["overall_pick_rate"] = last_info["total_deliveries"] * 3600 / (5 * len(infos))
    else:
        last_info["overall_pick_rate"] = 0.0
    return last_info


def safe_valid_action_masks(env) -> np.ndarray:
    try:
        masks = np.asarray(env.compute_valid_action_masks(), dtype=np.float64)
        if masks.shape != (env.num_agents, env.action_size):
            raise ValueError(f"mask shape={masks.shape} expected={(env.num_agents, env.action_size)}")
        return masks
    except Exception:
        return np.ones((env.num_agents, env.action_size), dtype=np.float64)


def classify_action(env, action_id: int) -> str:
    if action_id == 0:
        return "NOOP"
    if 1 <= action_id <= len(env.goals):
        return "GOAL"
    charging_start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    if charging_start <= action_id <= len(env.action_id_to_coords_map):
        return "CHARGING"
    return "SHELF"


def nearest_id_by_path(env, agent, candidate_ids: List[int]) -> int:
    best_id = 0
    best_len = float("inf")
    for action_id in candidate_ids:
        if action_id not in env.action_id_to_coords_map:
            continue
        target = env.action_id_to_coords_map[action_id]
        path = env.find_path((agent.y, agent.x), target, agent, care_for_agents=False)
        if path and len(path) < best_len:
            best_len = len(path)
            best_id = int(action_id)
    return best_id


def get_requested_action_ids(env) -> List[int]:
    inv = {v: k for k, v in env.action_id_to_coords_map.items()}
    ids: List[int] = []
    for shelf in env.request_queue:
        aid = inv.get((int(shelf.y), int(shelf.x)))
        if aid is not None:
            ids.append(int(aid))
    return list(dict.fromkeys(ids))


def charging_action_ids(env) -> List[int]:
    start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    return list(range(start, len(env.action_id_to_coords_map) + 1))


def goal_action_ids(env) -> List[int]:
    return list(range(1, len(env.goals) + 1))


def empty_shelf_action_ids(env) -> List[int]:
    empty_mask = env.get_empty_shelf_information()
    return list(np.array(env.shelf_action_ids)[empty_mask > 0])


def fallback_actions(env, valid_masks: np.ndarray) -> List[int]:
    actions = [0] * env.num_agents
    requested_ids = get_requested_action_ids(env)
    goal_ids = goal_action_ids(env)
    charging_ids = charging_action_ids(env)

    for i, agent in enumerate(env.agents):
        if agent.battery < 25:
            actions[i] = nearest_id_by_path(env, agent, charging_ids)
            continue

        if agent.type == AgentType.AGV:
            if agent.carrying_shelf is not None:
                requested_shelf_ids = {int(s.id) for s in env.request_queue}
                if int(agent.carrying_shelf.id) in requested_shelf_ids:
                    actions[i] = nearest_id_by_path(env, agent, goal_ids)
                else:
                    empties = empty_shelf_action_ids(env)
                    actions[i] = nearest_id_by_path(env, agent, empties)
            else:
                actions[i] = nearest_id_by_path(env, agent, requested_ids)
        else:
            agv_targets = [int(a.target) for a in env.agents[: env.num_agvs] if int(a.target) > 0]
            candidates = agv_targets if agv_targets else requested_ids
            actions[i] = nearest_id_by_path(env, agent, candidates)

    for i in range(env.num_agents):
        if actions[i] < 0 or actions[i] >= env.action_size or valid_masks[i, actions[i]] <= 0:
            actions[i] = 0
    return actions


def candidate_ids_for_agent(env, idx: int, valid_masks: np.ndarray, max_candidate_ids: int) -> List[int]:
    valid_ids = sorted(int(i) for i in np.where(valid_masks[idx] > 0)[0].tolist())
    if not valid_ids:
        valid_ids = [0]
    if max_candidate_ids and max_candidate_ids > 0:
        return valid_ids[:max_candidate_ids]
    return valid_ids


def candidate_ids_for_prompt(env, idx: int, valid_masks: np.ndarray, max_candidate_ids: int) -> List[int]:
    base_ids = candidate_ids_for_agent(env, idx, valid_masks, 0)
    if idx < env.num_agvs:
        if max_candidate_ids and max_candidate_ids > 0:
            return base_ids[:max_candidate_ids]
        return base_ids

    # Picker prompt optimization:
    # Keep only action ids relevant to supporting AGVs plus charging/NOOP.
    # Execution still validates against full mask; this only narrows prompt choices.
    agv_shelf_targets = {
        int(a.target)
        for a in env.agents[: env.num_agvs]
        if int(a.target) > 0 and classify_action(env, int(a.target)) == "SHELF"
    }
    allowed = {0, *charging_action_ids(env), *agv_shelf_targets}
    filtered = [aid for aid in base_ids if aid in allowed]
    if not filtered:
        filtered = base_ids
    if max_candidate_ids and max_candidate_ids > 0:
        return filtered[:max_candidate_ids]
    return filtered


def battery_need_label(battery: float) -> str:
    if battery < 25.0:
        return "critical"
    if battery < 50.0:
        return "need_charging_soon"
    return "not_needed"


def describe_action_id(env, action_id: int) -> str:
    if int(action_id) == 0:
        return "0:NOOP"
    action_type = classify_action(env, int(action_id))
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return f"{int(action_id)}:{action_type}"
    return f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])})"


def describe_action_id_for_agent(env, agent, action_id: int) -> str:
    if int(action_id) == 0:
        return "0:NOOP (distance_steps=0)"
    action_type = classify_action(env, int(action_id))
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return f"{int(action_id)}:{action_type}"
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    distance = len(path) if path else 0
    return f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])}) (distance_steps={distance})"


def get_requested_shelves(env) -> List[str]:
    requested = []
    inv = {v: k for k, v in env.action_id_to_coords_map.items()}
    for shelf in env.request_queue:
        aid = inv.get((int(shelf.y), int(shelf.x)))
        if aid is None:
            requested.append(
                f"Requested shelf {int(shelf.id)} is at position ({int(shelf.x)},{int(shelf.y)}), but no action id was found."
            )
        else:
            requested.append(
                f"Requested shelf {int(shelf.id)} is at position ({int(shelf.x)},{int(shelf.y)}). "
                f"To go there, choose action id {int(aid)}."
            )
    return requested


def get_requested_shelves_for_agent(env, agent) -> List[str]:
    rows: List[Tuple[int, int, int, int, int]] = []
    inv = {v: k for k, v in env.action_id_to_coords_map.items()}
    for shelf in env.request_queue:
        sx, sy = int(shelf.x), int(shelf.y)
        aid = inv.get((sy, sx))
        if aid is None:
            continue
        coords = env.action_id_to_coords_map.get(int(aid))
        if coords is None:
            continue
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if path is None:
            continue
        rows.append((len(path), int(shelf.id), int(aid), sx, sy))

    rows.sort(key=lambda r: (r[0], r[2]))
    return [
        f"shelf_id={sid} action_id={aid} pos=({sx},{sy}) distance_steps={steps}"
        for steps, sid, aid, sx, sy in rows
    ]


def get_picker_agv_task_view(env) -> List[str]:
    requested_shelf_ids = {int(s.id) for s in env.request_queue}
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    lines: List[str] = []
    for i, agv in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agv.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agv.x), int(agv.y)) == (int(coords[1]), int(coords[0])))
        carrying = agv.carrying_shelf is not None
        carrying_requested = bool(carrying and int(agv.carrying_shelf.id) in requested_shelf_ids)
        target_type = classify_action(env, target_id) if target_id > 0 else "NOOP"
        picker_here = bool((int(agv.x), int(agv.y)) in picker_positions)

        need = "none"
        if target_type == "SHELF" and at_target and not picker_here:
            need = "needs_picker_now_at_target"
        elif target_type == "SHELF" and (not at_target) and (not carrying) and (target_id in requested_shelf_ids):
            need = "may_need_picker_soon_for_load"
        elif target_type == "SHELF" and (not at_target) and carrying and (not carrying_requested):
            need = "may_need_picker_soon_for_unload"

        lines.append(
            f"agv agent_{i}: target={target_id}:{target_type} "
            f"busy={bool(agv.busy)} carrying={int(agv.carrying_shelf.id) if carrying else None} "
            f"battery={float(agv.battery):.1f} support_need={need}"
        )
    return lines


def nearest_charging_station_for_agent(env, agent) -> Tuple[int, int]:
    charging_ids = charging_action_ids(env)
    if not charging_ids:
        return 0, 0
    nearest_id = nearest_id_by_path(env, agent, charging_ids)
    coords = env.action_id_to_coords_map.get(int(nearest_id))
    if nearest_id <= 0 or coords is None:
        return 0, 0
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    return int(nearest_id), len(path) if path else 0


def nearest_goal_for_agent(env, agent) -> Tuple[int, int]:
    goal_ids = goal_action_ids(env)
    if not goal_ids:
        return 0, 0
    best_goal_id = 0
    best_steps: Optional[int] = None
    for gid in goal_ids:
        coords = env.action_id_to_coords_map.get(int(gid))
        if coords is None:
            continue
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if path is None:
            continue
        steps = len(path)
        if best_steps is None or steps < best_steps:
            best_steps = steps
            best_goal_id = int(gid)
    if best_steps is None:
        return 0, 0
    return best_goal_id, best_steps


def nearest_charging_station_from_coords(env, start_coords: Tuple[int, int], agent) -> Tuple[int, int]:
    charging_ids = charging_action_ids(env)
    if not charging_ids:
        return 0, 0
    best_charge_id = 0
    best_steps: Optional[int] = None
    for cid in charging_ids:
        coords = env.action_id_to_coords_map.get(int(cid))
        if coords is None:
            continue
        path = env.find_path(start_coords, coords, agent, care_for_agents=False)
        if path is None:
            continue
        steps = len(path)
        if best_steps is None or steps < best_steps:
            best_steps = steps
            best_charge_id = int(cid)
    if best_steps is None:
        return 0, 0
    return best_charge_id, best_steps


def nearest_empty_shelf_slots_for_agent(env, agent, limit: int = 5) -> List[str]:
    slot_rows: List[Tuple[int, int, int, int]] = []
    for aid in empty_shelf_action_ids(env):
        coords = env.action_id_to_coords_map.get(int(aid))
        if coords is None:
            continue
        path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
        if path is None:
            continue
        steps = len(path)
        slot_rows.append((steps, int(aid), int(coords[1]), int(coords[0])))
    slot_rows.sort(key=lambda r: (r[0], r[1]))
    out: List[str] = []
    for steps, aid, x, y in slot_rows[: max(0, int(limit))]:
        out.append(f"action_id={aid} at ({x},{y}) estimated_steps={steps}")
    return out


def charging_station_occupancy(env) -> List[str]:
    lines: List[str] = []
    charging_ids = charging_action_ids(env)
    for cid in charging_ids:
        coords = env.action_id_to_coords_map.get(int(cid))
        if coords is None:
            continue
        x, y = int(coords[1]), int(coords[0])
        occupants: List[str] = []
        for i, agent in enumerate(env.agents):
            if (int(agent.x), int(agent.y)) == (x, y):
                occupants.append(
                    f"agent_{i}({agent.type.name},battery={float(agent.battery):.1f},need={battery_need_label(float(agent.battery))})"
                )
        lines.append(f"charging_action={cid} at ({x},{y}) occupants=[{', '.join(occupants) if occupants else 'none'}]")
    return lines


def charging_now_agents(env) -> List[str]:
    station_cells = {(int(s.x), int(s.y)) for s in env.charging_stations}
    out: List[str] = []
    for i, agent in enumerate(env.agents):
        if (int(agent.x), int(agent.y)) in station_cells:
            out.append(f"agent_{i}({agent.type.name},battery={float(agent.battery):.1f})")
    return out


def assign_picker_support_pairs(env) -> Dict[int, int]:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    requested_shelf_ids = {int(s.id) for s in env.request_queue}
    support_agvs_waiting: List[int] = []
    support_agvs_enroute_unload: List[int] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        picker_here = (int(agent.x), int(agent.y)) in picker_positions
        carrying = agent.carrying_shelf is not None
        carrying_is_requested = bool(carrying and int(agent.carrying_shelf.id) in requested_shelf_ids)

        needs_waiting_support = at_target and (not picker_here)
        needs_enroute_load_support = (not at_target) and (not carrying) and (target_id in requested_shelf_ids)
        needs_enroute_unload_support = (not at_target) and carrying and (not carrying_is_requested)

        if needs_waiting_support:
            support_agvs_waiting.append(i)
            continue
        if needs_enroute_load_support:
            # Early support: AGV is moving to a requested shelf and will need picker for load at arrival.
            support_agvs_enroute_unload.append(i)
            continue
        if needs_enroute_unload_support:
            # Early support: AGV is returning delivered shelf to empty slot and will need picker at arrival.
            support_agvs_enroute_unload.append(i)

    picker_indices = list(range(env.num_agvs, env.num_agents))
    support_agvs = support_agvs_waiting + support_agvs_enroute_unload
    if not picker_indices or not support_agvs:
        return {}

    remaining_pickers = set(picker_indices)
    pairs: Dict[int, int] = {}
    for agv_idx in support_agvs:
        agv = env.agents[agv_idx]
        target = env.action_id_to_coords_map.get(int(agv.target))
        if target is None:
            continue
        best_picker = None
        best_len = float("inf")
        for p_idx in remaining_pickers:
            picker = env.agents[p_idx]
            path = env.find_path((picker.y, picker.x), target, picker, care_for_agents=False)
            if path and len(path) < best_len:
                best_len = len(path)
                best_picker = p_idx
        if best_picker is not None:
            pairs[agv_idx] = int(best_picker)
            remaining_pickers.remove(best_picker)
    return pairs


def _ranked_picker_distances_to_target(env, target_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
    ranked: List[Tuple[int, int]] = []
    for p_idx in range(env.num_agvs, env.num_agents):
        picker = env.agents[p_idx]
        path = env.find_path((picker.y, picker.x), target_coords, picker, care_for_agents=False)
        if path:
            ranked.append((int(p_idx), int(len(path))))
    ranked.sort(key=lambda x: x[1])
    return ranked


def waiting_agv_support_lines(env, max_picker_suggestions: int = 3) -> List[str]:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    requested_shelf_ids = {int(s.id) for s in env.request_queue}
    lines: List[str] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        picker_here = (int(agent.x), int(agent.y)) in picker_positions
        carrying = agent.carrying_shelf is not None
        carrying_is_requested = bool(carrying and int(agent.carrying_shelf.id) in requested_shelf_ids)
        needs_waiting_support = at_target and (not picker_here)
        needs_enroute_load_support = (not at_target) and (not carrying) and (target_id in requested_shelf_ids)
        needs_enroute_unload_support = (not at_target) and carrying and (not carrying_is_requested)
        if not needs_waiting_support and not needs_enroute_load_support and not needs_enroute_unload_support:
            continue
        ranked = _ranked_picker_distances_to_target(env, coords)
        if needs_waiting_support:
            event_label = "waiting for Picker support"
        elif needs_enroute_load_support:
            event_label = "en route to requested shelf; picker support needed soon for load"
        else:
            event_label = "en route to unload shelf; picker support needed soon"
        if ranked:
            recommended_picker, recommended_steps = ranked[0]
            top = ranked[: max(1, max_picker_suggestions)]
            top_text = ", ".join([f"agent_{p}(steps={d})" for p, d in top])
            lines.append(
                f"AGV agent_{i} is {event_label} at shelf action id {target_id} "
                f"position ({int(coords[1])},{int(coords[0])}). "
                f"Recommended picker is agent_{recommended_picker} (shortest path {recommended_steps} steps). "
                f"Closest pickers: {top_text}."
            )
            continue
        lines.append(
            f"AGV agent_{i} is {event_label} at shelf action id {target_id} "
            f"position ({int(coords[1])},{int(coords[0])})."
        )
    return lines


def picker_support_priority_line(env, picker_idx: int) -> str:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    requested_shelf_ids = {int(s.id) for s in env.request_queue}
    parts: List[str] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        picker_here = (int(agent.x), int(agent.y)) in picker_positions
        carrying = agent.carrying_shelf is not None
        carrying_is_requested = bool(carrying and int(agent.carrying_shelf.id) in requested_shelf_ids)
        needs_waiting_support = at_target and (not picker_here)
        needs_enroute_load_support = (not at_target) and (not carrying) and (target_id in requested_shelf_ids)
        needs_enroute_unload_support = (not at_target) and carrying and (not carrying_is_requested)
        if not needs_waiting_support and not needs_enroute_load_support and not needs_enroute_unload_support:
            continue
        ranked = _ranked_picker_distances_to_target(env, coords)
        order = [p for p, _ in ranked]
        if picker_idx in order:
            rank = order.index(picker_idx) + 1
            distance = dict(ranked)[picker_idx]
            if needs_waiting_support:
                support_label = "waiting support"
            elif needs_enroute_load_support:
                support_label = "early load support"
            else:
                support_label = "early unload support"
            parts.append(f"For AGV agent_{i} ({support_label}) at action {target_id}, your picker rank is {rank}/{len(order)} (distance {distance} steps).")
    if not parts:
        return "You are not currently ranked for any waiting AGV support."
    return " ".join(parts)


def build_prompt_fields(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    step_count: int,
    max_candidate_ids: int,
    max_action_hold_steps: int,
    inbox_messages: Optional[List[Dict[str, Any]]] = None,
    handled_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    agent = env.agents[agent_idx]
    candidate_ids = candidate_ids_for_prompt(env, agent_idx, valid_masks, max_candidate_ids)
    candidate_text = "\n".join(
        f"- {describe_action_id_for_agent(env, agent, aid)}" for aid in candidate_ids
    )
    requested = get_requested_shelves(env)
    requested_text = "\n".join(f"- {row}" for row in requested) if requested else "- none"
    all_agents_overview: List[str] = []
    for i, a in enumerate(env.agents):
        target_id = int(a.target)
        target_coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        target_text = (
            "none"
            if target_coords is None
            else f"{target_id}:{classify_action(env, target_id)}@({int(target_coords[1])},{int(target_coords[0])})"
        )
        all_agents_overview.append(
            f"agent_{i} type={a.type.name} pos=({int(a.x)},{int(a.y)}) "
            f"target={target_text} busy={bool(a.busy)} battery={float(a.battery):.1f} "
            f"carrying={int(a.carrying_shelf.id) if a.carrying_shelf else None}"
        )
    picker_agv_tasks = get_picker_agv_task_view(env)

    self_target_coords = env.action_id_to_coords_map.get(int(agent.target)) if int(agent.target) > 0 else None
    self_target_text = (
        "none"
        if self_target_coords is None
        else f"{int(agent.target)}:{classify_action(env, int(agent.target))}@({int(self_target_coords[1])},{int(self_target_coords[0])})"
    )
    self_state = (
        f"pos=({int(agent.x)},{int(agent.y)}), "
        f"target={self_target_text}, "
        f"busy={bool(agent.busy)}, "
        f"battery={float(agent.battery):.1f} ({battery_need_label(float(agent.battery))}), "
        f"carrying={int(agent.carrying_shelf.id) if agent.carrying_shelf else None}"
    )

    pending_rows: List[str] = []
    active_message_row = "- none"
    unresolved_messages = list(inbox_messages or [])
    for msg in unresolved_messages:
        row = (
            f"{msg.get('message_id', 'unknown')} status={msg.get('status', 'UNREAD')} "
            f"from=agent_{int(msg.get('from_agent', -1))} "
            f"sent_step={int(msg.get('step', -1))} delivered_step={int(msg.get('delivered_step', -1))} "
            f"scenario={msg.get('scenario', 'unknown')} text={msg.get('text', '')}"
        )
        pending_rows.append(row)
    if unresolved_messages:
        active = next((m for m in unresolved_messages if str(m.get("status", "")).upper() == "ACTIVE"), None)
        if active is None:
            active = unresolved_messages[0]
        active_message_row = (
            f"- {active.get('message_id', 'unknown')} status={active.get('status', 'UNREAD')} "
            f"from=agent_{int(active.get('from_agent', -1))} scenario={active.get('scenario', 'unknown')} "
            f"text={active.get('text', '')}"
        )

    handled_rows: List[str] = []
    for msg in list(handled_messages or [])[-3:]:
        handled_rows.append(
            f"{msg.get('message_id', 'unknown')} handled_step={int(msg.get('handled_step', -1))} "
            f"from=agent_{int(msg.get('from_agent', -1))} scenario={msg.get('scenario', 'unknown')} "
            f"text={msg.get('text', '')}"
        )

    return {
        "step": step_count,
        "agent_id": agent_idx,
        "agent_type": agent.type.name,
        "self_state": self_state,
        "requested_shelves": requested_text,
        "all_agents": "\n".join(f"- {row}" for row in all_agents_overview),
        "candidate_actions": candidate_text,
        "picker_agv_tasks": "\n".join(f"- {row}" for row in picker_agv_tasks) if picker_agv_tasks else "- none",
        "max_hold_steps": max(1, int(max_action_hold_steps)),
        "inbox_messages": "\n".join(f"- {row}" for row in pending_rows) if pending_rows else "- none",
        "active_message": active_message_row,
        "recently_handled_messages": "\n".join(f"- {row}" for row in handled_rows) if handled_rows else "- none",
    }


def build_agent_prompt(
    env,
    agent_idx: int,
    valid_masks: np.ndarray,
    step_count: int,
    max_candidate_ids: int,
    max_action_hold_steps: int,
) -> str:
    picker_pairs = assign_picker_support_pairs(env)
    picker_to_agv = {p: a for a, p in picker_pairs.items()}
    charge_now = charging_now_agents(env)
    charge_occupancy = charging_station_occupancy(env)
    agent = env.agents[agent_idx]
    requested = get_requested_shelves(env)
    requested_for_agent = get_requested_shelves_for_agent(env, agent)
    picker_agv_tasks = get_picker_agv_task_view(env)
    candidate_ids = candidate_ids_for_prompt(env, agent_idx, valid_masks, max_candidate_ids)
    candidate_text = [describe_action_id_for_agent(env, agent, c) for c in candidate_ids]

    target_id = int(agent.target)
    target_coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
    target_text = "none" if target_coords is None else f"{target_id}:{classify_action(env, target_id)}@({int(target_coords[1])},{int(target_coords[0])})"
    is_moving = bool(agent.busy and target_id > 0)
    target_distance_steps = 0
    at_current_target = False
    if target_id > 0 and target_coords is not None:
        target_path = env.find_path((agent.y, agent.x), target_coords, agent, care_for_agents=False)
        target_distance_steps = len(target_path) if target_path else 0
        at_current_target = (int(agent.x), int(agent.y)) == (int(target_coords[1]), int(target_coords[0]))
    target_type = classify_action(env, target_id) if target_id > 0 else "NOOP"
    nearest_charge_id, nearest_charge_steps = nearest_charging_station_for_agent(env, agent)
    carrying = bool(agent.carrying_shelf is not None)
    carrying_text = f"carrying shelf_id={int(agent.carrying_shelf.id)}" if carrying else "not carrying a shelf"
    battery_need = battery_need_label(float(agent.battery))
    carrying_feasibility_line = ""
    carrying_policy_line = ""
    carrying_delivery_status_line = ""
    carrying_hard_constraint_line = ""
    empty_slot_lines: List[str] = []
    empty_slot_block = ""
    carrying_is_requested: Optional[bool] = None
    if carrying and agent.type == AgentType.AGV:
        requested_shelf_ids = {int(s.id) for s in env.request_queue}
        carrying_shelf_id = int(agent.carrying_shelf.id)
        carrying_is_requested = carrying_shelf_id in requested_shelf_ids
        if carrying_is_requested:
            carrying_delivery_status_line = (
                f"Carrying delivery status: shelf_id={carrying_shelf_id} is still requested (not delivered yet).\n"
            )
        else:
            carrying_delivery_status_line = (
                f"Carrying delivery status: shelf_id={carrying_shelf_id} is already delivered (not in request queue).\n"
            )
            empty_slot_lines = nearest_empty_shelf_slots_for_agent(env, agent, limit=6)
        reserve_steps = 5
        if carrying_is_requested:
            nearest_goal_id, nearest_goal_steps = nearest_goal_for_agent(env, agent)
            goal_coords = env.action_id_to_coords_map.get(int(nearest_goal_id)) if nearest_goal_id > 0 else None
            next_charge_id, next_charge_steps = (
                nearest_charging_station_from_coords(env, goal_coords, agent) if goal_coords is not None else (0, 0)
            )
            required_steps = nearest_goal_steps + next_charge_steps + reserve_steps
            feasible = (
                nearest_goal_id > 0
                and next_charge_id > 0
                and float(agent.battery) >= float(required_steps)
            )
            feasibility_text = "SUFFICIENT" if feasible else "INSUFFICIENT"
            carrying_feasibility_line = (
                f"Carrying feasibility: nearest GOAL is action_id={nearest_goal_id} (estimated_steps={nearest_goal_steps}); "
                f"from that GOAL nearest charging station is action_id={next_charge_id} (estimated_steps={next_charge_steps}); "
                f"battery check (GOAL + charge travel + 5 reserve): required={required_steps}, current={float(agent.battery):.1f} => {feasibility_text}.\n"
            )
            carrying_policy_line = (
                "- If you are an AGV carrying a shelf and carrying feasibility is SUFFICIENT, prioritize GOAL delivery now.\n"
                "- If you are an AGV carrying a shelf and carrying feasibility is INSUFFICIENT, prioritize charging now.\n"
            )
        else:
            nearest_empty_id = 0
            nearest_empty_steps = 0
            if empty_slot_lines:
                m = re.search(r"action_id=(\d+).*estimated_steps=(\d+)", empty_slot_lines[0])
                if m:
                    nearest_empty_id = int(m.group(1))
                    nearest_empty_steps = int(m.group(2))
            empty_coords = env.action_id_to_coords_map.get(nearest_empty_id) if nearest_empty_id > 0 else None
            next_charge_id, next_charge_steps = (
                nearest_charging_station_from_coords(env, empty_coords, agent) if empty_coords is not None else (0, 0)
            )
            required_steps = nearest_empty_steps + next_charge_steps + reserve_steps
            feasible = (
                nearest_empty_id > 0
                and next_charge_id > 0
                and float(agent.battery) >= float(required_steps)
            )
            feasibility_text = "SUFFICIENT" if feasible else "INSUFFICIENT"
            carrying_feasibility_line = (
                f"Post-delivery feasibility: nearest EMPTY slot is action_id={nearest_empty_id} (estimated_steps={nearest_empty_steps}); "
                f"from that slot nearest charging station is action_id={next_charge_id} (estimated_steps={next_charge_steps}); "
                f"battery check (empty-slot return + charge travel + 5 reserve): required={required_steps}, current={float(agent.battery):.1f} => {feasibility_text}.\n"
            )
            carrying_hard_constraint_line = (
                "HARD CONSTRAINT: your carried shelf is already delivered; GOAL action_ids are forbidden now. "
                "Choose an EMPTY shelf slot action_id.\n"
            )
            carrying_policy_line = (
                "- If you are an AGV carrying a shelf that is already delivered (not in request queue), do NOT select GOAL.\n"
                "- Move to an EMPTY shelf slot and unload with picker support.\n"
                "- Only choose charging first if post-delivery feasibility is INSUFFICIENT.\n"
            )
    if empty_slot_lines:
        empty_slot_block = (
            "- Candidate EMPTY shelf slots for returning carried shelf:\n"
            "  - " + "\n  - ".join(empty_slot_lines) + "\n"
        )

    meet_text = "no one"
    if agent.type == AgentType.AGV:
        assigned_picker = picker_pairs.get(agent_idx)
        if assigned_picker is not None:
            meet_text = f"picker agent_{assigned_picker}"
    else:
        assigned_agv = picker_to_agv.get(agent_idx)
        if assigned_agv is not None:
            meet_text = f"agv agent_{assigned_agv}"

    all_agents_overview: List[str] = []
    for i, a in enumerate(env.agents):
        t_id = int(a.target)
        t_coords = env.action_id_to_coords_map.get(t_id) if t_id > 0 else None
        t_str = "none" if t_coords is None else f"{t_id}@({int(t_coords[1])},{int(t_coords[0])})/{classify_action(env, t_id)}"
        all_agents_overview.append(
            f"agent_{i} type={a.type.name} pos=({int(a.x)},{int(a.y)}) target={t_str} "
            f"busy={bool(a.busy)} battery={float(a.battery):.1f} need={battery_need_label(float(a.battery))} "
            f"carrying={int(a.carrying_shelf.id) if a.carrying_shelf else None}"
        )

    pair_lines = [f"AGV agent_{a} -> Picker agent_{p}" for a, p in picker_pairs.items()]
    if not pair_lines:
        pair_lines = ["none"]
    waiting_support = waiting_agv_support_lines(env)
    if not waiting_support:
        waiting_support = ["none"]
    picker_priority_note = (
        picker_support_priority_line(env, agent_idx)
        if agent.type == AgentType.PICKER
        else "Not a picker agent."
    )
    goal_id_span = f"1..{len(env.goals)}" if len(env.goals) > 0 else "none"
    charging_targets: List[str] = []
    for i, a in enumerate(env.agents):
        t_id = int(a.target)
        if t_id > 0 and classify_action(env, t_id) == "CHARGING":
            charging_targets.append(f"agent_{i} -> charging_action={t_id}")
    if not charging_targets:
        charging_targets = ["none"]

    if agent.type == AgentType.AGV:
        requested_block = (
            "- Requested shelves:\n"
            "  - " + ("\n  - ".join(requested) if requested else "none") + "\n"
            f"- Requested shelves ranked by YOUR distance (agent_{agent_idx}):\n"
            "  - " + ("\n  - ".join(requested_for_agent) if requested_for_agent else "none_reachable") + "\n"
        )
    else:
        requested_block = (
            "- AGV request/support view for picker coordination:\n"
            "  - " + ("\n  - ".join(picker_agv_tasks) if picker_agv_tasks else "none") + "\n"
        )

    if agent.type == AgentType.AGV:
        role_policy_block = (
            "- AGV policy:\n"
            "- If carrying a requested shelf, move it to a GOAL action id.\n"
            "- If not carrying, prefer requested shelf action_ids that are closer to you (smaller distance_steps in your ranked list), unless charging/safety constraints dominate.\n"
            "- At a target SHELF with non-critical battery, avoid detouring to charging before shelf interaction/coordination.\n"
        )
    else:
        role_policy_block = (
            "- PICKER policy:\n"
            "- If any AGV is waiting for support OR en route to unload/load support shelf, prioritize moving to that AGV target shelf action id.\n"
            "- If your battery is not_needed (>=50) or need_charging_soon (25..49), and any AGV support_need is needs_picker_now_at_target or may_need_picker_soon_for_load/unload, always choose the AGV support shelf action_id.\n"
            "- In that case, do not choose CHARGING and do not choose unrelated shelves.\n"
            "- Keep support aligned with AGV target to reduce AGV waiting.\n"
        )

    return (
        "You are planning ONE warehouse robot action for this time step.\n"
        "Core context:\n"
        "- Multi-agent warehouse with AGV and Picker collaboration.\n"
        "- Objective: maximize pick/delivery throughput while keeping batteries safe.\n"
        "- Avoid any out-of-charge scenario.\n"
        "- Use charging strategically so throughput does not collapse.\n"
        "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
        "\n"
        "Typical flow:\n"
        "- AGV without load moves to requested shelf.\n"
        "- Picker and AGV meet at that shelf for loading.\n"
        "- AGV carries shelf to GOAL.\n"
        "- IMPORTANT: GOAL marks delivery only; AGV still carries the shelf after delivery.\n"
        "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
        "- Repeat with next request.\n"
        "\n"
        "Battery model:\n"
        "- Move step consumption = 1.\n"
        "- Load/Unload consumption = 2.\n"
        "- Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.\n"
        "\n"
        "Charging situation (global):\n"
        "- Agents currently on charging cells: " + (", ".join(charge_now) if charge_now else "none") + "\n"
        "- Charging station occupancy:\n"
        "  - " + "\n  - ".join(charge_occupancy if charge_occupancy else ["none"]) + "\n"
        "- Agents currently targeting charging actions:\n"
        "  - " + "\n  - ".join(charging_targets) + "\n"
        "\n"
        "Current global state:\n"
        f"{requested_block}"
        "- AGV/Picker support pairs to reduce waiting:\n"
        "  - " + "\n  - ".join(pair_lines) + "\n"
        "- AGVs needing picker support now/soon:\n"
        "  - " + "\n  - ".join(waiting_support) + "\n"
        f"- Picker support priority note for you: {picker_priority_note}\n"
        "- All agents battery/position/target snapshot:\n"
        "  - " + "\n  - ".join(all_agents_overview) + "\n"
        "\n"
        f"You are agent_{agent_idx} ({agent.type.name}).\n"
        f"Current step index: {step_count}.\n"
        f"Your current position is ({int(agent.x)},{int(agent.y)}).\n"
        f"Your movement status: {'moving' if is_moving else 'not moving'}.\n"
        f"Your current target is {target_text}.\n"
        f"Your current target type is {target_type}.\n"
        f"Your current target is {target_distance_steps} steps away.\n"
        f"At current target: {'yes' if at_current_target else 'no'}.\n"
        f"You need to meet: {meet_text}.\n"
        f"Your battery level is {float(agent.battery):.1f} ({battery_need}).\n"
        f"You are {carrying_text}.\n"
        f"Nearest charging station is action_id={nearest_charge_id} (estimated_steps={nearest_charge_steps}).\n"
        f"{carrying_delivery_status_line}"
        f"{carrying_hard_constraint_line}"
        f"{carrying_feasibility_line}"
        f"{empty_slot_block}"
        "\n"
        "Decision policy:\n"
        "- Choose exactly ONE action_id for this agent.\n"
        "- Choose from the valid action ids listed below.\n"
        f"{role_policy_block}"
        f"{carrying_policy_line}"
        f"- GOAL action ids are {goal_id_span}. Any GOAL position is acceptable for delivery.\n"
        "- If battery is critical, strongly prioritize charging.\n"
        "- If battery needs charging soon, decide between charging vs productive task based on risk and current coordination demand.\n"
        "- If battery is not_needed and there is productive work, avoid charging.\n"
        "- If you choose a CHARGING action, select a slot with occupants=[none] from the charging occupancy list.\n"
        "- If multiple slots are free, avoid a slot already targeted by another agent in 'Agents currently targeting charging actions'.\n"
        "- Prefer the nearest free charging slot among your valid candidate action ids.\n"
        "- Keep AGV/Picker coordination tight to reduce AGV waiting.\n"
        "\n"
        "Valid action ids for this agent (from environment action mask):\n"
        "- " + "\n- ".join(candidate_text) + "\n"
        "\n"
        "Respond in plain text only:\n"
        "Reasoning: <short>\n"
        "Action: <action_id>\n"
        f"Steps: <integer between 1 and {max(1, max_action_hold_steps)}>\n"
        "If unsure, use Steps: 1.\n"
    )


def build_prompt_by_mode(
    env,
    args,
    experiment_cfg: Dict[str, Any],
    agent_idx: int,
    valid_masks: np.ndarray,
    step_count: int,
    max_candidate_ids: int,
    max_action_hold_steps: int,
    inboxes: Optional[List[deque]] = None,
    inboxes_handled: Optional[List[deque]] = None,
    rag_retriever=None,
) -> str:
    prompt: str
    if args.experiment_mode == "legacy_rich":
        prompt = build_agent_prompt(
            env,
            agent_idx,
            valid_masks,
            step_count,
            max_candidate_ids,
            max_action_hold_steps,
        )
        rag_fields = build_prompt_fields(
            env,
            agent_idx,
            valid_masks,
            step_count,
            max_candidate_ids,
            max_action_hold_steps,
            inbox_messages=[],
            handled_messages=[],
        )
        rag_query = build_rag_query(rag_fields)
        rag_context = retrieve_rag_context(rag_retriever, rag_query, args.rag_top_k, args.rag_max_chars)
        return append_rag_context(prompt, rag_context)

    msg_settings = message_settings_from_config(experiment_cfg)
    prompt_window = max(0, int(msg_settings.get("inbox_prompt_messages", 20)))
    agent_inbox = list(inboxes[agent_idx])[-prompt_window:] if inboxes is not None else []
    handled_history = list(inboxes_handled[agent_idx]) if inboxes_handled is not None else []
    fields = build_prompt_fields(
        env,
        agent_idx,
        valid_masks,
        step_count,
        max_candidate_ids,
        max_action_hold_steps,
        inbox_messages=agent_inbox,
        handled_messages=handled_history,
    )
    templates = experiment_cfg.get("templates", {})

    if args.experiment_mode == "fixed_prompt_action":
        template = templates.get("fixed_prompt_action", default_experiment_config()["templates"]["fixed_prompt_action"])
        if not isinstance(template, str):
            raise ValueError("templates.fixed_prompt_action must be a string")
        prompt = render_template(template, fields)
    elif args.experiment_mode == "agent_type_prompt":
        role_templates = templates.get("agent_type_prompt", {})
        if not isinstance(role_templates, dict):
            raise ValueError("templates.agent_type_prompt must be an object")
        agent_type = str(fields["agent_type"])
        template = role_templates.get(agent_type) or role_templates.get("default")
        if not isinstance(template, str):
            raise ValueError(f"No template found for agent_type={agent_type}")
        prompt = render_template(template, fields)
    elif args.experiment_mode == "message_or_action":
        template = templates.get("message_or_action", default_experiment_config()["templates"]["message_or_action"])
        if not isinstance(template, str):
            raise ValueError("templates.message_or_action must be a string")
        prompt = render_template(template, fields)
    else:
        raise ValueError(f"Unsupported experiment_mode: {args.experiment_mode}")

    rag_query = build_rag_query(fields)
    rag_context = retrieve_rag_context(rag_retriever, rag_query, args.rag_top_k, args.rag_max_chars)
    return append_rag_context(prompt, rag_context)


def build_language_prompt(env, valid_masks: np.ndarray, step_count: int, max_candidate_ids: int) -> str:
    # Retained for backward compatibility with existing logs/tools.
    # New planning loop uses per-agent prompts via build_agent_prompt.
    requested = get_requested_shelves(env)

    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    agent_lines = []
    for idx, agent in enumerate(env.agents):
        cands = candidate_ids_for_agent(env, idx, valid_masks, max_candidate_ids)
        cand_desc = []
        for aid in cands:
            if aid == 0:
                cand_desc.append("0:NOOP")
            else:
                ty = classify_action(env, int(aid))
                yx = env.action_id_to_coords_map.get(int(aid))
                if yx is None:
                    cand_desc.append(f"{int(aid)}:{ty}")
                else:
                    cand_desc.append(f"{int(aid)}:{ty}@({int(yx[1])},{int(yx[0])})")

        current_target = int(agent.target)
        target_type = classify_action(env, current_target) if current_target > 0 else "NOOP"
        target_coords = env.action_id_to_coords_map.get(current_target) if current_target > 0 else None
        at_target = bool(target_coords is not None and (int(agent.x), int(agent.y)) == (int(target_coords[1]), int(target_coords[0])))
        picker_at_same_cell = bool((int(agent.x), int(agent.y)) in picker_positions)
        needs_picker_for_load = bool(
            agent.type == AgentType.AGV
            and at_target
            and bool(agent.busy)
            and agent.carrying_shelf is None
            and target_type == "SHELF"
        )
        needs_picker_for_unload = bool(
            agent.type == AgentType.AGV
            and at_target
            and bool(agent.busy)
            and agent.carrying_shelf is not None
            and target_type == "SHELF"
        )
        waiting_for_picker_support = bool(
            agent.type == AgentType.AGV
            and (needs_picker_for_load or needs_picker_for_unload)
            and not picker_at_same_cell
        )

        agent_lines.append(
            f"agent_{idx} type={agent.type.name} pos=({int(agent.x)},{int(agent.y)}) battery={float(agent.battery):.1f} "
            f"busy={bool(agent.busy)} carrying_shelf_id={int(agent.carrying_shelf.id) if agent.carrying_shelf else None} "
            f"current_target={current_target} target_type={target_type} at_target={at_target} "
            f"picker_at_same_cell={picker_at_same_cell} "
            f"needs_picker_for_load={needs_picker_for_load} "
            f"needs_picker_for_unload={needs_picker_for_unload} "
            f"waiting_for_picker_support={waiting_for_picker_support} "
            f"candidates=[{', '.join(cand_desc)}]"
        )

    charging_ids = charging_action_ids(env)
    charging_span = f"{charging_ids[0]}..{charging_ids[-1]}" if charging_ids else "none"

    return (
        "You are dispatching robots in a warehouse.\n"
        "Goal: maximize shelf pickups/deliveries while keeping batteries safe and coordinating AGV with Picker.\n"
        "\n"
        "WAREHOUSE ABSTRACTION (must follow):\n"
        "- A request is for a SHELF (pod), not an item unit.\n"
        "- A request is fulfilled when the requested shelf is carried by an AGV to a GOAL location.\n"
        "- AGV-Picker co-location is required at shelf interactions only:\n"
        "  (a) pickup/load at shelf location, and (b) return/unload at shelf location.\n"
        "- Picker does NOT need to travel with AGV to GOAL.\n"
        "- After GOAL delivery, AGV still carries the same shelf and should return it to an empty shelf slot.\n"
        "\n"
        "BATTERY POLICY (important):\n"
        "- critical battery: < 10 => choose CHARGING.\n"
        "- medium battery: 10..29 => charging is optional, prefer task progress unless clearly risky.\n"
        "- healthy battery: >= 30 => DO NOT choose CHARGING for AGV/Picker.\n"
        "- With battery around 50+, charging is a bad decision in this task.\n"
        "\n"
        "CRITICAL RULE:\n"
        "- For each agent_i, choose ONLY one action_id from that same agent_i candidates=[...].\n"
        "- Do NOT invent new IDs. Do NOT use IDs from another agent's candidate list.\n"
        "- If an action_id is not listed in that agent's candidates, you must NOT choose it.\n"
        "- Requested shelves list is global context, but final choice must still come from agent_i candidates.\n"
        "- Avoid arbitrary retargeting when busy=True, but DO reassign when it reduces AGV waiting_for_picker_support or resolves idling.\n"
        "\n"
        "Action ID legend:\n"
        "- 0 = NOOP\n"
        f"- 1..{len(env.goals)} = GOAL locations\n"
        f"- {charging_span} = CHARGING stations\n"
        "- Other listed IDs are shelf locations.\n"
        "\n"
        "ACTION SELECTION ORDER (strict):\n"
        "1) If carrying_shelf_id is not None: prioritize GOAL action_ids (1..N), not requested shelf ids.\n"
        "2) Else if battery is critical (<10): choose CHARGING.\n"
        "3) Else choose a requested shelf/support shelf from candidates.\n"
        "4) Use NOOP only if no productive candidate is appropriate.\n"
        "5) Minimize waiting: schedule AGV and Picker movements so AGV waiting_for_picker_support stays low.\n"
        "6) CO-LOCATION RULE: for shelf interaction, AGV and Picker must go to the SAME shelf action_id in the same time window.\n"
        "   If AGV targets shelf X for load/unload support, Picker should also target shelf X unless battery is critical.\n"
        "7) Dynamic reassignment for throughput: if one agent is busy/unavailable, assign the nearest AVAILABLE counterpart\n"
        "   (nearest by visible coordinates among valid candidates) to avoid waiting and increase pickup rate.\n"
        "8) In 1-AGV/1-Picker mode, align picker action with AGV support shelf whenever possible.\n"
        "\n"
        "VARIABLE-DRIVEN SUPPORT RULE (strict):\n"
        "- Read AGV flags directly from agent lines: picker_at_same_cell, needs_picker_for_load, needs_picker_for_unload, waiting_for_picker_support.\n"
        "- If an AGV has waiting_for_picker_support=true, assign the Picker to that AGV current_target shelf action_id immediately.\n"
        "- If an AGV has needs_picker_for_load=true and picker_at_same_cell=false, Picker must move to that AGV shelf (same action_id).\n"
        "- If an AGV has needs_picker_for_unload=true and picker_at_same_cell=false, Picker must move to that AGV shelf (same action_id).\n"
        "- Do NOT send Picker to a different requested shelf while any AGV waiting_for_picker_support=true (unless picker battery is critical).\n"
        "- In conflicts, AGV support has higher priority than starting a new shelf request.\n"
        "- Simple mapping clue: if agent_x (AGV) has waiting_for_picker_support=true, then Picker action := agent_x.current_target.\n"
        "- Simple mapping clue: if agent_x needs_picker_for_load=true and picker_at_same_cell=false, then Picker action := agent_x.current_target.\n"
        "- Simple mapping clue: if agent_x needs_picker_for_unload=true and picker_at_same_cell=false, then Picker action := agent_x.current_target.\n"
        "Important scenarios:\n"
        "- If battery is critical (<10), send that agent to a CHARGING action_id.\n"
        "- If battery is healthy (>=30), do not send that agent to charging.\n"
        "- If AGV carries requested shelf, send to GOAL.\n"
        "- If AGV carries non-requested shelf, send to empty shelf return location.\n"
        "- If AGV goes to a requested shelf, Picker should choose the SAME shelf action_id unless picker battery is critical.\n"
        "- If AGV is returning a carried shelf to storage, Picker should support at that return shelf action_id.\n"
        "- Avoid sending picker to charging when AGV needs immediate shelf support and picker battery is not critical.\n"
        "- Deadlock breaker: if AGV is at a requested shelf with carrying_shelf_id=None, picker must move to that same shelf action_id.\n"
        "- If waiting_for_picker_support=true for AGV, assign nearest available picker to AGV support shelf immediately.\n"
        "- Picker NOOP is discouraged; only use NOOP when battery is critical, or when picker is already at the AGV support shelf.\n"
        "- Do not claim AGV is carrying shelf unless carrying_shelf_id is not None in the provided state.\n"
        "\n"
        "Respond in plain text ONLY in this format:\n"
        "Reasoning: <short explanation>\n"
        "Actions:\n"
        "agent_0 -> <action_id>\n"
        "agent_1 -> <action_id>\n"
        "... one line per agent\n"
        "Example:\n"
        "Reasoning: AGV goes to requested shelf; picker supports same shelf.\n"
        "Actions:\n"
        "agent_0 -> 12\n"
        "agent_1 -> 12\n"
        "Bad example (avoid): battery=51 and choosing charging.\n"
        "\n"
        "Do not output JSON.\n"
        "Use only listed candidate action ids for each agent.\n"
        f"\nStep: {step_count}\n"
        f"Requested shelves:\n- " + ("\n- ".join(requested) if requested else "none") + "\n"
        f"\nAgents:\n- " + "\n- ".join(agent_lines)
    )


def query_ollama_text(model: str, ollama_url: str, prompt: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 700},
    }
    req = request.Request(
        ollama_url,
        data=str.encode(json.dumps(payload)),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def derive_ollama_base_url(ollama_url: str) -> str:
    suffix = "/api/generate"
    if ollama_url.endswith(suffix):
        return ollama_url[: -len(suffix)]
    return ollama_url


def init_rag_retriever(args):
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            "RAG requires additional dependencies. Install with: pip install -e .[llm]"
        ) from exc

    rag_db_dir = Path(args.rag_db_dir)
    base_url = args.ollama_base_url.strip() or derive_ollama_base_url(args.ollama_url.strip())
    embeddings = OllamaEmbeddings(model=args.rag_embedding_model, base_url=base_url)

    if bool(args.rag_rebuild_index):
        build_rag_index_from_docs(args)

    if not rag_db_dir.exists():
        raise FileNotFoundError(
            f"RAG DB not found: {rag_db_dir}. Build it with --rag_rebuild_index --rag_docs_dir <dir>."
        )
    candidate_k = max(max(1, int(args.rag_top_k)) * 4, 12)
    vectorstore = Chroma(persist_directory=str(rag_db_dir), embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": candidate_k})


def build_rag_query(fields: Dict[str, Any]) -> str:
    return (
        "Warehouse coordination reflection query. "
        f"agent_type={fields.get('agent_type', '')}; "
        f"self_state={fields.get('self_state', '')}; "
        f"requested={fields.get('requested_shelves', '')}; "
        f"inbox={fields.get('inbox_messages', '')}; "
        "Retrieve reusable coordination lessons for AGV/Picker teamwork, message strategy, "
        "support-request timing, plan-intent communication, battery-aware coordination, "
        "delivery-oriented action rules, and prompt additions that improve shelf deliveries."
    )


def retrieve_rag_context(rag_retriever, query: str, top_k: int, max_chars: int) -> str:
    if rag_retriever is None:
        return ""
    if bool(getattr(rag_retriever, "_disabled_due_to_error", False)):
        return ""

    try:
        docs = rag_retriever.invoke(query)
    except Exception as exc:
        try:
            setattr(rag_retriever, "_disabled_due_to_error", True)
        except Exception:
            pass
        log_block("RAG_QUERY_ERROR", f"{type(exc).__name__}: {exc!r}; disabling RAG for remainder of run")
        return ""
    if not docs:
        return ""

    reflection_prefix = "knowledge/reflections/"
    filtered_docs = []
    for d in docs:
        source = str(d.metadata.get("source", "")) if hasattr(d, "metadata") else ""
        normalized = source.replace("\\", "/")
        if normalized.startswith(reflection_prefix) or f"/{reflection_prefix}" in normalized:
            filtered_docs.append(d)

    docs = filtered_docs[: max(1, int(top_k))]
    if not docs:
        return ""

    pieces: List[str] = []
    for d in docs[: max(1, int(top_k))]:
        source = str(d.metadata.get("source", "unknown")) if hasattr(d, "metadata") else "unknown"
        text = str(getattr(d, "page_content", "")).strip()
        if not text:
            continue
        pieces.append(f"[source={source}]\n{text}")
    out = "\n\n".join(pieces)
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars] + f"\n...[TRUNCATED {len(out) - max_chars} chars]"
    return out


def append_rag_context(prompt: str, rag_context: str) -> str:
    if not rag_context:
        return prompt
    return (
        prompt
        + "\n"
        + "Grounding context (retrieved knowledge):\n"
        + rag_context
        + "\n"
        + "Use this context only when relevant. Current state and valid candidate action IDs are authoritative.\n"
    )


def query_text_with_backend(
    backend: str,
    model: str,
    prompt: str,
    timeout_s: int,
    ollama_url: str,
    ollama_base_url: str,
) -> str:
    if backend == "langchain":
        return query_langchain_ollama_text(model, ollama_base_url, prompt, timeout_s)
    return query_ollama_text(model, ollama_url, prompt, timeout_s)


def build_reflection_input(transcript: Dict[str, Any], max_chars: int) -> str:
    lines: List[str] = []
    goal = str(transcript.get("system_goal", "maximize shelf deliveries"))
    delivery_metric = str(transcript.get("delivery_metric", "shelf_deliveries"))
    lines.append("Episode metadata:")
    lines.append(
        f"- episode={transcript.get('episode_idx')} seed={transcript.get('seed')} "
        f"mode={transcript.get('experiment_mode')} total_steps={transcript.get('total_steps')}"
    )
    lines.append(f"- system_goal={goal}")
    lines.append(f"- primary_metric={delivery_metric}")
    lines.append(
        f"- total_{delivery_metric}={transcript.get('total_shelf_deliveries', 0)} "
        f"llm_calls={transcript.get('llm_calls', 0)} llm_failures={transcript.get('llm_failures', 0)}"
    )
    lines.append(f"- episode_end_reason={transcript.get('episode_end_reason', 'unknown')}")
    lines.append(f"- episode_end_reasons={transcript.get('episode_end_reasons', [])}")

    message_summary = transcript.get("message_summary", {})
    lines.append("Messaging summary:")
    lines.append(f"- total_messages_sent={message_summary.get('total_messages_sent', 0)}")
    lines.append(f"- total_messages_delivered={message_summary.get('total_messages_delivered', 0)}")
    lines.append(f"- total_messages_activated={message_summary.get('total_messages_activated', 0)}")
    lines.append(f"- total_messages_handled={message_summary.get('total_messages_handled', 0)}")
    lines.append(f"- total_messages_pending_end={message_summary.get('total_messages_pending_end', 0)}")
    lines.append(
        f"- messages_with_delivery_within_{message_summary.get('delivery_after_message_window_steps', 0)}_steps="
        f"{message_summary.get('messages_with_delivery_within_window', 0)} "
        f"(ratio={float(message_summary.get('message_to_delivery_window_ratio', 0.0)):.2f})"
    )
    lines.append(f"- messages_by_scenario={message_summary.get('messages_by_scenario', {})}")
    lines.append(f"- messages_by_pair={message_summary.get('messages_by_pair', {})}")
    lines.append(f"- message_status_counts={message_summary.get('message_status_counts', {})}")

    sample_limit = max(1, int(transcript.get("reflection_message_samples", 12)))
    samples: List[Dict[str, Any]] = []
    for step in transcript.get("steps", []):
        for evt in step.get("messages_sent", []):
            if isinstance(evt, dict):
                samples.append(evt)

    # Prioritize examples that were followed by delivery within window.
    samples.sort(key=lambda e: (not bool(e.get("delivery_within_window", False)), int(e.get("step", 0))))
    lines.append(f"Message samples (up to {sample_limit}):")
    for evt in samples[:sample_limit]:
        lines.append(
            f"- step={evt.get('step')} from=agent_{evt.get('from_agent')} to=agent_{evt.get('to_agent')} "
            f"scenario={evt.get('scenario')} delivery_within_window={evt.get('delivery_within_window', False)} "
            f"text={evt.get('text', '')}"
        )

    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[TRUNCATED {len(text) - max_chars} chars]"
    return text


def build_reflection_prompt(reflection_input: str) -> str:
    return (
        "You are analyzing a multi-agent warehouse episode transcript.\n"
        "There are multiple agents involved and they are expected to communicate and coordinate to achieve the system goal.\n"
        "The agents are controlled by an LLM and it is taking input from your suggestions.\n"
        "You will see metrics of an episode, with number of messages, and what they messaged.\n"
        "Your task is to reflect on how the messaging and coordination may have influenced the episode outcome, and how it could be improved.\n"
        "If number of messages is low, consider encouraging the LLM agent controlling them to improve coordination.\n"
        f"{reflection_input}\n"
    )


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_episode_summary_text(
    args,
    episode_idx: int,
    seed: int,
    elapsed_s: float,
    metrics: Dict[str, Any],
    transcript: Dict[str, Any],
    transcript_path: Path,
) -> str:
    message_summary = transcript.get("message_summary", {})
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "Episode Summary",
        f"Created: {created_at}",
        f"Episode: {episode_idx}",
        f"Seed: {seed}",
        f"Environment: {args.env_id}",
        f"Model: {args.model}",
        f"Backend: {args.llm_backend}",
        f"Experiment mode: {args.experiment_mode}",
        f"Elapsed seconds: {elapsed_s:.2f}",
        "",
        "Key Metrics",
        f"- Deliveries: {int(metrics.get('total_deliveries', 0))}",
        f"- Clashes: {int(metrics.get('total_clashes', 0))}",
        f"- Messages sent: {int(message_summary.get('total_messages_sent', 0))}",
        f"- Messages handled: {int(message_summary.get('total_messages_handled', 0))}",
        "",
        "Additional Metrics",
        f"- Episode length: {int(metrics.get('episode_length', 0))}",
        f"- Global return: {float(metrics.get('global_episode_return', 0.0)):.2f}",
        f"- Overall pick rate: {float(metrics.get('overall_pick_rate', 0.0)):.2f}",
        f"- LLM calls: {int(transcript.get('llm_calls', 0))}",
        f"- LLM failures: {int(transcript.get('llm_failures', 0))}",
        f"- Messages delivered: {int(message_summary.get('total_messages_delivered', 0))}",
        f"- Messages activated: {int(message_summary.get('total_messages_activated', 0))}",
        f"- Pending messages at end: {int(message_summary.get('total_messages_pending_end', 0))}",
        f"- Episode end reason: {transcript.get('episode_end_reason', 'unknown')}",
        f"- Episode end reasons: {transcript.get('episode_end_reasons', [])}",
        "",
        f"Transcript: {transcript_path}",
    ]
    return "\n".join(lines) + "\n"


def rebuild_rag_index_via_subprocess(args) -> None:
    script_path = Path(__file__).with_name("build_rag_index.py")
    base_url = args.ollama_base_url.strip() or derive_ollama_base_url(args.ollama_url.strip())
    cmd = [
        sys.executable,
        str(script_path),
        "--docs_dir",
        str(args.rag_docs_dir),
        "--db_dir",
        str(args.rag_db_dir),
        "--embedding_model",
        str(args.rag_embedding_model),
        "--ollama_base_url",
        str(base_url),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rebuild_subprocess_failed rc={proc.returncode} "
            f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
        )


def build_rag_index_from_docs(args) -> None:
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "RAG requires additional dependencies. Install with: pip install -e .[llm]"
        ) from exc

    rag_db_dir = Path(args.rag_db_dir)
    rag_docs_dir = Path(args.rag_docs_dir)
    base_url = args.ollama_base_url.strip() or derive_ollama_base_url(args.ollama_url.strip())
    embeddings = OllamaEmbeddings(model=args.rag_embedding_model, base_url=base_url)

    if not rag_docs_dir.exists():
        raise FileNotFoundError(f"rag_docs_dir not found: {rag_docs_dir}")
    text_files = [p for p in rag_docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst"}]
    if not text_files:
        raise ValueError(f"No supported docs found in {rag_docs_dir} (.md/.txt/.rst)")
    if rag_db_dir.exists():
        shutil.rmtree(rag_db_dir)
    raw_docs: List[Any] = []
    for path in text_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        raw_docs.append(Document(page_content=text, metadata={"source": str(path)}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=str(rag_db_dir))
    try:
        vectorstore.persist()
    except Exception:
        pass


def query_langchain_ollama_text(
    model: str,
    ollama_base_url: str,
    prompt: str,
    timeout_s: int,
) -> str:
    try:
        from langchain_ollama import OllamaLLM
    except ImportError as exc:
        raise ImportError(
            "LangChain backend requires langchain-ollama. "
            "Install with: pip install -e .[llm]"
        ) from exc

    llm_kwargs = {
        "model": model,
        "base_url": ollama_base_url,
        "temperature": 0.1,
        "num_predict": 700,
    }

    try:
        llm = OllamaLLM(**llm_kwargs, client_kwargs={"timeout": timeout_s})
    except TypeError:
        llm = OllamaLLM(**llm_kwargs)

    return str(llm.invoke(prompt)).strip()


def query_llm_text(args, prompt: str) -> str:
    if args.llm_backend == "langchain":
        base_url = args.ollama_base_url.strip() or derive_ollama_base_url(args.ollama_url.strip())
        return query_langchain_ollama_text(
            model=args.model,
            ollama_base_url=base_url,
            prompt=prompt,
            timeout_s=args.request_timeout_s,
        )
    return query_ollama_text(args.model, args.ollama_url, prompt, args.request_timeout_s)


def normalize_message_status_update(raw_status: str) -> str:
    text = str(raw_status or "").strip().upper()
    if "HANDLED" in text or "DONE" in text or "RESOLVED" in text:
        return "HANDLED"
    if "ACTIVE" in text or "IN_PROGRESS" in text or "WORKING" in text:
        return "ACTIVE"
    return "KEEP"


def apply_message_status_update(
    inbox: deque,
    handled_messages: deque,
    focus_message_id: str,
    status_update: str,
    step_count: int,
    agent_idx: int,
    detail: str,
) -> Optional[Dict[str, Any]]:
    focus_message_id = str(focus_message_id or "").strip()
    normalized_status = normalize_message_status_update(status_update)
    if not focus_message_id or normalized_status == "KEEP":
        return None

    target_idx = -1
    target_message: Optional[Dict[str, Any]] = None
    for idx, msg in enumerate(inbox):
        if str(msg.get("message_id", "")).strip() == focus_message_id:
            target_idx = idx
            target_message = msg
            break
    if target_message is None:
        return {
            "agent_id": int(agent_idx),
            "message_id": focus_message_id,
            "status": "INVALID",
            "detail": str(detail or "").strip(),
            "step": int(step_count),
        }

    old_status = str(target_message.get("status", "UNREAD")).upper()
    target_message["last_status_detail"] = str(detail or "").strip()
    target_message["last_status_step"] = int(step_count)
    if normalized_status == "ACTIVE":
        target_message["status"] = "ACTIVE"
        target_message["active_step"] = int(step_count)
        return {
            "agent_id": int(agent_idx),
            "message_id": focus_message_id,
            "from_status": old_status,
            "status": "ACTIVE",
            "detail": str(detail or "").strip(),
            "step": int(step_count),
        }

    handled_message = dict(target_message)
    handled_message["status"] = "HANDLED"
    handled_message["handled_step"] = int(step_count)
    handled_message["handled_by_agent"] = int(agent_idx)
    del inbox[target_idx]
    handled_messages.append(handled_message)
    return {
        "agent_id": int(agent_idx),
        "message_id": focus_message_id,
        "from_status": old_status,
        "status": "HANDLED",
        "detail": str(detail or "").strip(),
        "step": int(step_count),
    }


def parse_actions_from_text(text: str, num_agents: int) -> List[int]:
    actions = [0] * num_agents
    found = 0

    # Primary pattern: agent_0 -> 12
    for m in re.finditer(r"agent[_\s-]*(\d+)\s*[-:=]>?\s*(-?\d+)", text, flags=re.IGNORECASE):
        idx = int(m.group(1))
        act = int(m.group(2))
        if 0 <= idx < num_agents:
            actions[idx] = act
            found += 1

    # Secondary fallback: line starts with index
    if found == 0:
        for m in re.finditer(r"(?:^|\n)\s*(\d+)\s*[:=-]\s*(-?\d+)", text):
            idx = int(m.group(1))
            act = int(m.group(2))
            if 0 <= idx < num_agents:
                actions[idx] = act
                found += 1

    if found == 0:
        raise ValueError("No parseable actions found in LLM text output")

    return actions


def parse_single_action_from_text(text: str) -> int:
    # Prefer explicit final action lines. This avoids accidentally reading
    # "action_id=..." values that may appear in reasoning sentences.
    line_patterns = [
        r"^\s*action\s*[:=-]\s*(-?\d+)\b",
        r"^\s*action\s*[:=-]\s*(-?\d+)\s*:",
        r"^\s*agent[_\s-]*\d+\s*[-:=]>\s*(-?\d+)\b",
    ]
    for pattern in line_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))

    patterns = [
        r"action\s*[:=-]?\s*(-?\d+)",
        r"agent[_\s-]*\d+\s*[-:=]>\s*(-?\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Last-resort fallback: first integer anywhere.
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group(0))
    raise ValueError("No parseable single action found in LLM text output")


def parse_action_and_steps_from_text(text: str, max_action_hold_steps: int) -> tuple[int, int]:
    action_id = parse_single_action_from_text(text)
    steps = 1

    step_patterns = [
        r"^\s*steps?\s*[:=-]\s*(-?\d+)\b",
        r"^\s*duration\s*[:=-]\s*(-?\d+)\b",
        r"hold(?:_steps)?\s*[:=-]\s*(-?\d+)\b",
    ]
    for pattern in step_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            steps = int(m.group(1))
            break

    steps = max(1, min(int(steps), max(1, int(max_action_hold_steps))))
    return int(action_id), int(steps)


def parse_decision_message_action_from_text(
    text: str,
    num_agents: int,
    max_action_hold_steps: int,
) -> Dict[str, Any]:
    decision = "ACTION"
    decision_match = re.search(r"^\s*decision\s*[:=-]\s*([A-Za-z_ -]+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if decision_match:
        parsed = decision_match.group(1).strip().upper()
        if "MESSAGE" in parsed:
            decision = "MESSAGE"
        elif "ACTION" in parsed:
            decision = "ACTION"

    to_agent: Optional[int] = None
    to_match = re.search(r"^\s*to\s*[:=-]\s*agent[_\s-]*(\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if to_match:
        to_agent = int(to_match.group(1))
    else:
        to_match = re.search(r"^\s*to\s*[:=-]\s*(\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
        if to_match:
            to_agent = int(to_match.group(1))

    message_text = ""
    msg_match = re.search(r"^\s*message\s*[:=-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if msg_match:
        message_text = msg_match.group(1).strip()

    focus_message_id = ""
    focus_match = re.search(r"^\s*focus\s+message\s*[:=-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if focus_match:
        focus_message_id = focus_match.group(1).strip()
        if focus_message_id.lower() in {"none", "blank", "n/a"}:
            focus_message_id = ""

    message_status = "KEEP"
    status_match = re.search(r"^\s*message\s+status\s*[:=-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if status_match:
        message_status = normalize_message_status_update(status_match.group(1))

    status_detail = ""
    status_detail_match = re.search(r"^\s*status\s+detail\s*[:=-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if status_detail_match:
        status_detail = status_detail_match.group(1).strip()

    parsed_action: Optional[int] = None
    explicit_action = re.search(r"^\s*action\s*[:=-]\s*(-?\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if explicit_action:
        parsed_action = int(explicit_action.group(1))

    steps = 1
    step_patterns = [
        r"^\s*steps?\s*[:=-]\s*(-?\d+)\b",
        r"^\s*duration\s*[:=-]\s*(-?\d+)\b",
        r"hold(?:_steps)?\s*[:=-]\s*(-?\d+)\b",
    ]
    for pattern in step_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            steps = int(m.group(1))
            break
    steps = max(1, min(int(steps), max(1, int(max_action_hold_steps))))

    has_message = (
        decision == "MESSAGE"
        and to_agent is not None
        and 0 <= int(to_agent) < int(num_agents)
        and bool(message_text)
    )
    return {
        "decision": decision,
        "to_agent": int(to_agent) if to_agent is not None else None,
        "message_text": message_text,
        "has_message": bool(has_message),
        "action_id": parsed_action,
        "steps": int(steps),
        "focus_message_id": focus_message_id,
        "message_status": message_status,
        "status_detail": status_detail,
    }


def infer_message_scenario(env, sender_idx: int) -> str:
    agent = env.agents[int(sender_idx)]
    target_id = int(agent.target)
    target_type = classify_action(env, target_id) if target_id > 0 else "NOOP"
    coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
    at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    picker_here = bool((int(agent.x), int(agent.y)) in picker_positions)

    if float(agent.battery) < 20.0:
        return "battery_critical"
    if float(agent.battery) < 35.0:
        return "battery_low"
    if agent.type == AgentType.AGV and target_type == "SHELF" and at_target and not picker_here:
        return "support_needed_now_at_shelf"
    if agent.type == AgentType.AGV and target_type == "SHELF" and bool(agent.busy) and not at_target:
        return "support_planning_for_shelf"
    if agent.type == AgentType.PICKER:
        return "picker_support_coordination"
    return "plan_intent_coordination"


def compute_message_summary(steps: List[Dict[str, Any]], delivery_window_steps: int) -> Dict[str, Any]:
    sent_events: List[Dict[str, Any]] = []
    delivery_steps: List[int] = []
    by_scenario: Dict[str, int] = {}
    by_pair: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    status_by_agent: Dict[str, int] = {}
    handled_message_ids: set[str] = set()
    active_message_ids: set[str] = set()
    activated_message_ids: set[str] = set()

    for step in steps:
        step_idx = int(step.get("step", 0))
        if int(step.get("shelf_deliveries", 0)) > 0:
            delivery_steps.append(step_idx)
        for evt in step.get("messages_sent", []):
            if not isinstance(evt, dict):
                continue
            sent_events.append(evt)
            scenario = str(evt.get("scenario", "unknown"))
            key = f"agent_{int(evt.get('from_agent', -1))}->agent_{int(evt.get('to_agent', -1))}"
            by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
            by_pair[key] = by_pair.get(key, 0) + 1
        for update in step.get("message_status_updates", []):
            if not isinstance(update, dict):
                continue
            status = str(update.get("status", "UNKNOWN")).upper()
            message_id = str(update.get("message_id", "")).strip()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in {"ACTIVE", "HANDLED"} and message_id:
                agent_key = f"agent_{int(update.get('agent_id', -1))}"
                status_by_agent[agent_key] = status_by_agent.get(agent_key, 0) + 1
            if status == "ACTIVE" and message_id:
                activated_message_ids.add(message_id)
                active_message_ids.add(message_id)
            if status == "HANDLED" and message_id:
                handled_message_ids.add(message_id)
                active_message_ids.discard(message_id)

    window = max(1, int(delivery_window_steps))
    worked = 0
    for evt in sent_events:
        sent_step = int(evt.get("step", -1))
        if sent_step < 0:
            continue
        has_followup_delivery = any((d > sent_step and d <= sent_step + window) for d in delivery_steps)
        if has_followup_delivery:
            worked += 1
        evt["delivery_within_window"] = bool(has_followup_delivery)
        evt["window_steps"] = window

    return {
        "total_messages_sent": int(len(sent_events)),
        "total_messages_delivered": int(sum(len(step.get("messages_delivered", [])) for step in steps)),
        "messages_by_scenario": by_scenario,
        "messages_by_pair": by_pair,
        "delivery_after_message_window_steps": window,
        "messages_with_delivery_within_window": int(worked),
        "message_to_delivery_window_ratio": float(worked / len(sent_events)) if sent_events else 0.0,
        "message_status_counts": status_counts,
        "message_status_updates_by_agent": status_by_agent,
        "total_messages_activated": int(len(activated_message_ids)),
        "total_messages_handled": int(len(handled_message_ids)),
        "active_message_ids": sorted(active_message_ids),
        "activated_message_ids": sorted(activated_message_ids),
        "handled_message_ids": sorted(handled_message_ids),
    }


def validate_actions(env, actions: List[int], valid_masks: np.ndarray) -> List[int]:
    out = [0] * env.num_agents
    for i in range(env.num_agents):
        a = int(actions[i]) if i < len(actions) else 0
        if a < 0 or a >= env.action_size:
            a = 0
        if valid_masks[i, a] <= 0:
            a = 0
        out[i] = a
    return out


def _at_target(env, agent) -> bool:
    if int(agent.target) <= 0:
        return False
    coords = env.action_id_to_coords_map.get(int(agent.target))
    if coords is None:
        return False
    return (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0]))


def replan_signals(env) -> tuple[bool, List[str]]:
    """
    Query LLM only when there is a meaningful planning event:
    - some agent is free (not busy),
    - some busy agent reached target (transition point),
    - AGV is waiting for picker support at target shelf.
    """
    reasons: List[str] = []
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}

    free_agents = [i for i, a in enumerate(env.agents) if not bool(a.busy)]
    if free_agents:
        reasons.append(f"free_agents={free_agents}")

    reached_targets = [i for i, a in enumerate(env.agents) if bool(a.busy) and _at_target(env, a)]
    if reached_targets:
        reasons.append(f"reached_targets={reached_targets}")

    waiting_support: List[int] = []
    for i, a in enumerate(env.agents[: env.num_agvs]):
        if not bool(a.busy) or not _at_target(env, a):
            continue
        if int(a.target) <= 0:
            continue
        target_type = classify_action(env, int(a.target))
        if target_type != "SHELF":
            continue
        picker_here = (int(a.x), int(a.y)) in picker_positions
        if not picker_here:
            waiting_support.append(i)
    if waiting_support:
        reasons.append(f"agv_waiting_picker={waiting_support}")

    return (len(reasons) > 0), reasons


def explain_actions(env, actions: List[int]) -> str:
    lines: List[str] = []
    for i, a in enumerate(actions):
        agent = env.agents[i]
        if int(a) == 0:
            lines.append(f"agent_{i} {agent.type.name} action=0 NOOP busy={bool(agent.busy)}")
            continue
        yx = env.action_id_to_coords_map.get(int(a))
        if yx is None:
            lines.append(f"agent_{i} {agent.type.name} action={int(a)} UNKNOWN busy={bool(agent.busy)}")
            continue
        note = "will_apply_new_target" if not agent.busy else "currently_busy_keeps_current_path"
        lines.append(
            f"agent_{i} {agent.type.name} action={int(a)} {classify_action(env, int(a))}@({int(yx[1])},{int(yx[0])}) "
            f"busy={bool(agent.busy)} {note}"
        )
    return "\n".join(lines)


def explain_executing_actions(
    env,
    actions: List[int],
    step_agent_records: List[Dict[str, Any]],
    hold_steps_remaining: List[int],
) -> str:
    lines: List[str] = []
    record_by_agent = {int(r.get("agent_id", -1)): r for r in step_agent_records}
    for idx, action_id in enumerate(actions):
        agent = env.agents[idx]
        record = record_by_agent.get(idx, {})
        source = "reused_plan" if bool(record.get("reused_action", False)) else "fresh_plan"
        hold_total = int(record.get("action_steps", 1))
        remaining = int(hold_steps_remaining[idx]) if idx < len(hold_steps_remaining) else 0
        if int(action_id) == 0:
            action_desc = "0:NOOP"
        else:
            action_type = classify_action(env, int(action_id))
            coords = env.action_id_to_coords_map.get(int(action_id))
            if coords is None:
                action_desc = f"{int(action_id)}:{action_type}"
            else:
                action_desc = f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])})"
        lines.append(
            f"step={source} agent_{idx} type={agent.type.name} executes action={action_desc} "
            f"hold_steps_total={hold_total} remaining_reuse_steps={remaining}"
        )
    return "\n".join(lines)

def explain_candidates(env, valid_masks: np.ndarray, max_candidate_ids: int) -> str:
    lines: List[str] = []
    for idx, agent in enumerate(env.agents):
        cands = candidate_ids_for_prompt(env, idx, valid_masks, max_candidate_ids)
        cand_desc = []
        for aid in cands:
            if aid == 0:
                cand_desc.append("0:NOOP")
            else:
                ty = classify_action(env, int(aid))
                yx = env.action_id_to_coords_map.get(int(aid))
                if yx is None:
                    cand_desc.append(f"{int(aid)}:{ty}")
                else:
                    cand_desc.append(f"{int(aid)}:{ty}@({int(yx[1])},{int(yx[0])})")
        lines.append(f"agent_{idx} type={agent.type.name} candidates=[{', '.join(cand_desc)}]")
    return "\n".join(lines)


def snapshot_agents(env) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    for i, a in enumerate(env.agents):
        out.append(
            {
                "idx": i,
                "type": a.type.name,
                "x": int(a.x),
                "y": int(a.y),
                "busy": bool(a.busy),
                "target": int(a.target),
                "carrying": int(a.carrying_shelf.id) if a.carrying_shelf is not None else None,
                "battery": float(a.battery),
                "picker_at_same_cell": bool((int(a.x), int(a.y)) in picker_positions) if a.type == AgentType.AGV else None,
            }
        )
    return out


def explain_state_changes(before: List[Dict[str, Any]], env) -> str:
    lines: List[str] = []
    for b in before:
        i = int(b["idx"])
        a = env.agents[i]
        new_carry = int(a.carrying_shelf.id) if a.carrying_shelf is not None else None
        changes: List[str] = []
        if b["carrying"] != new_carry:
            if b["carrying"] is None and new_carry is not None:
                changes.append(f"load_event carrying None->{new_carry}")
            elif b["carrying"] is not None and new_carry is None:
                changes.append(f"unload_event carrying {b['carrying']}->None")
            else:
                changes.append(f"carrying {b['carrying']}->{new_carry}")
        if b["target"] != int(a.target):
            changes.append(f"target {b['target']}->{int(a.target)}")
        if b["busy"] != bool(a.busy):
            changes.append(f"busy {b['busy']}->{bool(a.busy)}")
        if (b["x"], b["y"]) != (int(a.x), int(a.y)):
            changes.append(f"pos ({b['x']},{b['y']})->({int(a.x)},{int(a.y)})")
        if abs(float(b["battery"]) - float(a.battery)) > 1e-9:
            changes.append(f"battery {float(b['battery']):.1f}->{float(a.battery):.1f}")
        if a.type == AgentType.AGV:
            picker_positions = {(int(p.x), int(p.y)) for p in env.agents if p.type == AgentType.PICKER}
            picker_here = bool((int(a.x), int(a.y)) in picker_positions)
            if b["picker_at_same_cell"] != picker_here:
                changes.append(f"picker_at_same_cell {b['picker_at_same_cell']}->{picker_here}")
        if changes:
            lines.append(f"agent_{i} type={a.type.name}: " + "; ".join(changes))
    return "\n".join(lines) if lines else "no_state_change"


def determine_episode_end_reasons(env, args, step_count: int, done: bool) -> List[str]:
    reasons: List[str] = []
    max_inactivity_steps = getattr(env, "max_inactivity_steps", None)
    cur_inactive_steps = int(getattr(env, "_cur_inactive_steps", 0))
    if max_inactivity_steps and cur_inactive_steps >= int(max_inactivity_steps):
        reasons.append(f"env_max_inactivity_steps_reached({cur_inactive_steps}/{int(max_inactivity_steps)})")

    env_max_steps = getattr(env, "max_steps", None)
    cur_steps = int(getattr(env, "_cur_steps", 0))
    if env_max_steps and cur_steps >= int(env_max_steps):
        reasons.append(f"env_max_steps_reached({cur_steps}/{int(env_max_steps)})")

    request_queue = getattr(env, "request_queue", [])
    if len(request_queue) == 0:
        reasons.append("request_queue_empty")

    if not done and step_count >= int(args.max_steps_per_episode):
        reasons.append(f"script_max_steps_per_episode_reached({step_count}/{int(args.max_steps_per_episode)})")

    if not reasons:
        reasons.append("unknown")
    return reasons


def llm_episode(
    env,
    args,
    seed: int,
    experiment_cfg: Dict[str, Any],
    rag_retriever=None,
    episode_idx: int = 0,
):
    _ = env.reset(seed=seed)
    done = False
    step_count = 0
    llm_calls = 0
    llm_failures = 0
    delivery_happened = False

    infos: List[Dict[str, Any]] = []
    episode_returns = np.zeros(env.num_agents)
    global_episode_return = 0.0
    persisted_actions = [0] * env.num_agents
    hold_steps_remaining = [0] * env.num_agents
    msg_settings = message_settings_from_config(experiment_cfg)
    reflection_settings = reflection_settings_from_config(experiment_cfg)
    inbox_max = max(1, int(msg_settings.get("inbox_max_messages", 50)))
    message_default_action = str(msg_settings.get("message_default_action", "noop")).strip().lower()
    inboxes: List[deque] = [deque(maxlen=inbox_max) for _ in range(env.num_agents)]
    handled_inboxes: List[deque] = [deque(maxlen=inbox_max) for _ in range(env.num_agents)]
    pending_messages: List[Dict[str, Any]] = []
    next_message_id = 1
    total_shelf_deliveries = 0
    transcript: Dict[str, Any] = {
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "experiment_mode": str(args.experiment_mode),
        "llm_backend": str(args.llm_backend),
        "system_goal": str(reflection_settings.get("system_goal", "maximize shelf deliveries")),
        "delivery_metric": str(reflection_settings.get("delivery_metric", "shelf_deliveries")),
        "reflection_message_samples": int(reflection_settings.get("message_samples", 12)),
        "steps": [],
    }

    log_block("START", f"seed={seed}, env={args.env_id}, agents={env.num_agents}, mode={args.experiment_mode}")

    while not done and step_count < args.max_steps_per_episode:
        delivered_events: List[Dict[str, Any]] = []
        message_status_updates: List[Dict[str, Any]] = []
        if pending_messages:
            for evt in pending_messages:
                receiver = int(evt.get("to_agent", -1))
                if 0 <= receiver < env.num_agents:
                    delivered_event = {
                        "message_id": str(evt.get("message_id", "")),
                        "step": int(step_count),
                        "sent_step": int(evt.get("step", -1)),
                        "from_agent": int(evt.get("from_agent", -1)),
                        "to_agent": receiver,
                        "scenario": str(evt.get("scenario", "unknown")),
                        "text": str(evt.get("text", "")),
                        "status": "UNREAD",
                        "delivered_step": int(step_count),
                    }
                    inboxes[receiver].append(delivered_event)
                    delivered_events.append(dict(delivered_event))
            pending_messages = []
            if delivered_events:
                log_block(
                    "INBOX_DELIVERY",
                    "\n".join(
                        [
                            (
                                f"id={e['message_id']} to=agent_{int(e['to_agent'])} from=agent_{int(e['from_agent'])} "
                                f"scenario={e['scenario']} text={e['text']}"
                            )
                            for e in delivered_events
                        ]
                    ),
                )

        valid_masks = safe_valid_action_masks(env)
        log_block("STEP", f"step={step_count}")
        log_block("CANDIDATES", explain_candidates(env, valid_masks, args.max_candidate_ids))
        fallback = fallback_actions(env, valid_masks)
        actions = [0] * env.num_agents
        outgoing_messages: List[Dict[str, Any]] = []
        step_agent_records: List[Dict[str, Any]] = []

        for idx in range(env.num_agents):
            max_hold_for_agent = max(1, int(args.max_action_hold_steps))
            if env.agents[idx].type == AgentType.PICKER:
                max_hold_for_agent = min(max_hold_for_agent, max(1, int(args.max_picker_hold_steps)))
            persisted = int(persisted_actions[idx])
            if (
                hold_steps_remaining[idx] > 0
                and 0 <= persisted < env.action_size
                and valid_masks[idx, persisted] > 0
            ):
                actions[idx] = persisted
                hold_steps_remaining[idx] -= 1
                step_agent_records.append(
                    {
                        "agent_id": int(idx),
                        "reused_action": True,
                        "action": int(persisted),
                        "action_steps": int(hold_steps_remaining[idx] + 1),
                        "fallback_used": False,
                    }
                )
                log_block(
                    f"ACTION_REUSE_agent_{idx}",
                    f"action={persisted} remaining_reuse_steps={hold_steps_remaining[idx]}",
                )
                continue

            if hold_steps_remaining[idx] > 0:
                log_block(
                    f"ACTION_REUSE_INVALID_agent_{idx}",
                    f"persisted_action={persisted} became invalid_by_mask; forcing re-query",
                )
                hold_steps_remaining[idx] = 0

            agent_prompt = build_prompt_by_mode(
                env,
                args,
                experiment_cfg,
                idx,
                valid_masks,
                step_count,
                args.max_candidate_ids,
                max_hold_for_agent,
                inboxes=inboxes,
                inboxes_handled=handled_inboxes,
                rag_retriever=rag_retriever,
            )
            log_block(f"PROMPT_agent_{idx}", maybe_truncate(agent_prompt, args.log_text_chars))
            agent_record: Dict[str, Any] = {
                "agent_id": int(idx),
                "prompt": maybe_truncate(agent_prompt, int(args.transcript_text_chars)),
                "fallback_used": False,
            }
            try:
                llm_text = query_llm_text(args, agent_prompt)
                llm_calls += 1
                log_block(f"LLM_OUTPUT_agent_{idx}", maybe_truncate(llm_text, args.log_text_chars))
                agent_record["llm_output"] = maybe_truncate(llm_text, int(args.transcript_text_chars))
                if args.experiment_mode == "message_or_action":
                    parsed = parse_decision_message_action_from_text(
                        llm_text,
                        num_agents=env.num_agents,
                        max_action_hold_steps=max_hold_for_agent,
                    )
                    action_id = parsed["action_id"]
                    action_steps = int(parsed["steps"])
                    if action_id is None:
                        if parsed["decision"] == "MESSAGE":
                            action_id = int(fallback[idx]) if message_default_action == "fallback" else 0
                        else:
                            action_id = parse_single_action_from_text(llm_text)
                    if bool(parsed["has_message"]):
                        message_event = {
                            "message_id": f"msg_{next_message_id:04d}",
                            "step": int(step_count),
                            "from_agent": int(idx),
                            "to_agent": int(parsed["to_agent"]),
                            "text": str(parsed["message_text"]),
                            "scenario": infer_message_scenario(env, idx),
                        }
                        next_message_id += 1
                        outgoing_messages.append(message_event)
                        agent_record["message_sent"] = {
                            "message_id": str(message_event["message_id"]),
                            "to_agent": int(parsed["to_agent"]),
                            "text": str(parsed["message_text"]),
                            "scenario": str(message_event["scenario"]),
                        }
                        log_block(
                            f"MESSAGE_SEND_agent_{idx}",
                            (
                                f"id={message_event['message_id']} to=agent_{int(parsed['to_agent'])} "
                                f"scenario={message_event['scenario']} "
                                f"text={parsed['message_text']}"
                            ),
                        )
                    status_update = apply_message_status_update(
                        inboxes[idx],
                        handled_inboxes[idx],
                        parsed.get("focus_message_id", ""),
                        parsed.get("message_status", "KEEP"),
                        step_count,
                        idx,
                        parsed.get("status_detail", ""),
                    )
                    if status_update is not None:
                        agent_record["message_status_update"] = status_update
                        message_status_updates.append(status_update)
                        log_block(
                            f"MESSAGE_STATUS_agent_{idx}",
                            (
                                f"id={status_update.get('message_id')} "
                                f"status={status_update.get('status')} "
                                f"from_status={status_update.get('from_status', 'n/a')} "
                                f"detail={status_update.get('detail', '')}"
                            ),
                        )
                else:
                    action_id, action_steps = parse_action_and_steps_from_text(llm_text, max_hold_for_agent)
                if action_id < 0 or action_id >= env.action_size or valid_masks[idx, action_id] <= 0:
                    raise ValueError(f"action_id={action_id} invalid_by_mask")
                actions[idx] = int(action_id)
                persisted_actions[idx] = int(action_id)
                hold_steps_remaining[idx] = max(0, int(action_steps) - 1)
                agent_record["action"] = int(actions[idx])
                agent_record["action_steps"] = int(action_steps)
                step_agent_records.append(agent_record)
                log_block(
                    f"ACTION_PLAN_agent_{idx}",
                    f"action={actions[idx]} hold_steps_total={action_steps} remaining_reuse_steps={hold_steps_remaining[idx]}",
                )
            except (error.URLError, TimeoutError, ValueError, ImportError) as exc:
                llm_failures += 1
                actions[idx] = int(fallback[idx])
                persisted_actions[idx] = int(actions[idx])
                hold_steps_remaining[idx] = 0
                agent_record["action"] = int(actions[idx])
                agent_record["action_steps"] = 1
                agent_record["fallback_used"] = True
                agent_record["error"] = repr(exc)
                step_agent_records.append(agent_record)
                log_block(f"LLM_PARSE_ERROR_agent_{idx}", repr(exc))
                log_block(f"FALLBACK_agent_{idx}", f"action={actions[idx]} hold_steps_total=1")

        log_block("VALIDATED_ACTIONS", str(actions))
        log_block("EXECUTING_ACTIONS", explain_executing_actions(env, actions, step_agent_records, hold_steps_remaining))
        log_block("ACTION_EFFECT", explain_actions(env, actions))

        if args.render:
            env.render(mode="human")

        prev_agents = snapshot_agents(env)
        _, reward, terminated, truncated, info = env.step(actions)
        state_change_text = explain_state_changes(prev_agents, env)
        log_block("STATE_CHANGE", state_change_text)
        if outgoing_messages:
            pending_messages.extend(outgoing_messages)
        step_deliveries = int(info.get("shelf_deliveries", 0))
        total_shelf_deliveries += step_deliveries
        transcript["steps"].append(
            {
                "step": int(step_count),
                "messages_delivered": delivered_events,
                "messages_sent": outgoing_messages,
                "message_status_updates": message_status_updates,
                "agents": step_agent_records,
                "actions": [int(a) for a in actions],
                "state_change": state_change_text,
                "shelf_deliveries": step_deliveries,
            }
        )
        infos.append(info)
        episode_returns += np.asarray(reward, dtype=np.float64)
        global_episode_return += float(np.sum(reward))

        if step_deliveries > 0:
            delivery_happened = True
            log_block("DELIVERY", f"step={step_count} shelf_deliveries={step_deliveries}")

        step_count += 1
        done = all(terminated) or all(truncated)

        if args.progress_every > 0 and step_count % args.progress_every == 0:
            log_block(
                "PROGRESS",
                f"step={step_count} total_deliveries={sum(i.get('shelf_deliveries', 0) for i in infos)} "
                f"llm_calls={llm_calls} llm_failures={llm_failures}",
            )

    if infos:
        infos[-1]["llm_calls"] = llm_calls
        infos[-1]["llm_failures"] = llm_failures
        infos[-1]["delivery_happened"] = delivery_happened
    transcript["total_steps"] = int(step_count)
    transcript["llm_calls"] = int(llm_calls)
    transcript["llm_failures"] = int(llm_failures)
    transcript["delivery_happened"] = bool(delivery_happened)
    transcript["total_shelf_deliveries"] = int(total_shelf_deliveries)
    transcript["episode_end_reasons"] = determine_episode_end_reasons(env, args, step_count, done)
    transcript["episode_end_reason"] = str(transcript["episode_end_reasons"][0])
    transcript["final_pending_messages"] = [
        {
            "agent_id": int(agent_idx),
            "messages": [dict(msg) for msg in inbox],
        }
        for agent_idx, inbox in enumerate(inboxes)
    ]
    transcript["handled_messages"] = [
        {
            "agent_id": int(agent_idx),
            "messages": [dict(msg) for msg in inbox],
        }
        for agent_idx, inbox in enumerate(handled_inboxes)
    ]
    transcript["message_summary"] = compute_message_summary(
        transcript.get("steps", []),
        int(reflection_settings.get("delivery_after_message_window_steps", 6)),
    )
    transcript["message_summary"]["total_messages_pending_end"] = int(sum(len(inbox) for inbox in inboxes))

    return infos, global_episode_return, episode_returns, delivery_happened, transcript


def print_episode_result(prefix: str, idx: int, elapsed: float, metrics: Dict[str, Any]):
    print(
        f"\n{prefix} Episode {idx}: "
        f"[Overall Pick Rate={metrics.get('overall_pick_rate', 0.0):.2f}] "
        f"[Global return={metrics.get('global_episode_return', 0.0):.2f}] "
        f"[Total shelf deliveries={metrics.get('total_deliveries', 0):.2f}] "
        f"[Total clashes={metrics.get('total_clashes', 0):.2f}] "
        f"[Total stuck={metrics.get('total_stuck', 0):.2f}] "
        f"[Total AGVs distance={metrics.get('total_agvs_distance', 0):.2f}] "
        f"[Total Pickers distance={metrics.get('total_pickers_distance', 0):.2f}] "
        f"[Episode length={metrics.get('episode_length', 0)}] "
        f"[FPS={metrics.get('episode_length', 0) / elapsed if elapsed > 0 else 0.0:.2f}]\n"
    )
    print(
        f"{prefix} Episode {idx} Summary: "
        f"[Deliveries={int(metrics.get('total_deliveries', 0))}] "
        f"[Clashes={int(metrics.get('total_clashes', 0))}] "
        f"[Messages sent={int(metrics.get('total_messages_sent', 0))}] "
        f"[Messages handled={int(metrics.get('total_messages_handled', 0))}]\n"
    )


def summarize_policy_results(name: str, metric_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metric_rows:
        return {"overall_pick_rate": 0.0, "global_episode_return": 0.0, "total_deliveries": 0.0}
    avg_pick = float(np.mean([m.get("overall_pick_rate", 0.0) for m in metric_rows]))
    avg_ret = float(np.mean([m.get("global_episode_return", 0.0) for m in metric_rows]))
    avg_del = float(np.mean([m.get("total_deliveries", 0.0) for m in metric_rows]))
    print(
        f"\n{name} AVG: [Overall Pick Rate={avg_pick:.2f}] "
        f"[Global return={avg_ret:.2f}] [Total deliveries={avg_del:.2f}]\n"
    )
    return {
        "overall_pick_rate": avg_pick,
        "global_episode_return": avg_ret,
        "total_deliveries": avg_del,
    }


if __name__ == "__main__":
    args = parser.parse_args()
    experiment_cfg = load_experiment_config(args.config_path)
    rag_retriever = None
    if args.llm_backend == "langchain":
        log_block(
            "RAG",
            f"auto_enabled=true db={args.rag_db_dir} docs={args.rag_docs_dir} top_k={args.rag_top_k} max_chars={args.rag_max_chars}",
        )
    else:
        log_block("RAG", "auto_enabled=false (backend is not langchain)")

    env = gym.make(args.env_id)
    env.unwrapped.verbose_events = bool(args.warehouse_event_logs)

    llm_rows: List[Dict[str, Any]] = []
    llm_delivery_any = False

    for ep in range(args.num_episodes):
        start = time.time()
        reflection_note_path: Optional[Path] = None
        rag_retriever = None
        if args.llm_backend == "langchain":
            try:
                rag_retriever = init_rag_retriever(args)
                _ = retrieve_rag_context(rag_retriever, f"episode_{ep}_reflection_retrieval_check", 1, 200)
                if bool(getattr(rag_retriever, "_disabled_due_to_error", False)):
                    rag_retriever = None
                log_block("RAG_EPISODE_INIT", f"episode={ep} enabled={str(rag_retriever is not None).lower()}")
            except Exception as exc:
                log_block("RAG_INIT_ERROR", f"episode={ep} {repr(exc)}")
                log_block("RAG_EPISODE_INIT", f"episode={ep} enabled=false")
                rag_retriever = None
        infos, global_ret, ep_rets, delivery_happened, transcript = llm_episode(
            env.unwrapped,
            args,
            seed=args.seed + ep,
            experiment_cfg=experiment_cfg,
            rag_retriever=rag_retriever,
            episode_idx=ep,
        )
        llm_delivery_any = llm_delivery_any or delivery_happened
        metrics = info_statistics(infos, global_ret, ep_rets)
        metrics["total_messages_sent"] = int(transcript.get("message_summary", {}).get("total_messages_sent", 0))
        metrics["total_messages_handled"] = int(transcript.get("message_summary", {}).get("total_messages_handled", 0))
        print_episode_result("LLM", ep, time.time() - start, metrics)
        llm_rows.append(metrics)

        transcript_dir = Path(args.transcript_dir)
        transcript_path = transcript_dir / f"episode_{ep:03d}_seed_{args.seed + ep}.json"
        write_json_file(transcript_path, transcript)
        log_block("TRANSCRIPT", f"saved={transcript_path}")

        summary_dir = Path(args.episode_summary_dir)
        summary_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = summary_dir / f"episode_summary_ep_{ep:03d}_{summary_timestamp}.txt"
        summary_text = build_episode_summary_text(
            args=args,
            episode_idx=ep,
            seed=args.seed + ep,
            elapsed_s=time.time() - start,
            metrics=metrics,
            transcript=transcript,
            transcript_path=transcript_path,
        )
        write_text_file(summary_path, summary_text)
        log_block("EPISODE_SUMMARY", f"saved={summary_path}")

        if args.enable_episode_reflection:
            reflection_backend = args.llm_backend if args.reflection_backend == "auto" else args.reflection_backend
            reflection_model = args.reflection_model.strip() or args.model
            reflection_input = build_reflection_input(transcript, max_chars=int(args.reflection_max_chars))
            reflection_prompt = build_reflection_prompt(reflection_input)
            reflection_text = ""
            try:
                reflection_text = query_text_with_backend(
                    backend=reflection_backend,
                    model=reflection_model,
                    prompt=reflection_prompt,
                    timeout_s=int(args.reflection_timeout_s),
                    ollama_url=args.ollama_url,
                    ollama_base_url=args.ollama_base_url.strip() or derive_ollama_base_url(args.ollama_url.strip()),
                )
            except Exception as exc:
                log_block("REFLECTION_ERROR", repr(exc))
                reflection_text = (
                    "## Episode Summary\n"
                    "Reflection generation failed.\n\n"
                    "## Observed Success Patterns\n"
                    "- none\n\n"
                    "## Failure Patterns\n"
                    "- reflection model call failed\n\n"
                    "## Actionable Coordination Rules\n"
                    "- keep previous prompt and inspect transcript manually\n\n"
                    "## Prompt Additions (short bullet list)\n"
                    "- none\n"
                )

            notes_dir = Path(args.reflection_notes_dir)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            note_path = notes_dir / f"episode_{ep:03d}_{timestamp}.md"
            header = (
                f"# Reflection Note\n\n"
                f"- episode: {ep}\n"
                f"- seed: {args.seed + ep}\n"
                f"- model: {reflection_model}\n"
                f"- backend: {reflection_backend}\n"
                f"- transcript: {transcript_path}\n\n"
            )
            write_text_file(note_path, header + reflection_text.strip() + "\n")
            log_block("REFLECTION_NOTE", f"saved={note_path}")
            reflection_note_path = note_path

            if args.llm_backend == "langchain" and not args.reflection_skip_rag_rebuild:
                try:
                    # Release current retriever reference before rebuilding the DB on disk.
                    rag_retriever = None
                    rebuild_rag_index_via_subprocess(args)
                    original_rebuild = bool(args.rag_rebuild_index)
                    try:
                        args.rag_rebuild_index = False
                        rag_retriever = init_rag_retriever(args)
                        _ = retrieve_rag_context(rag_retriever, "healthcheck reflection retrieval", 1, 200)
                        if bool(getattr(rag_retriever, "_disabled_due_to_error", False)):
                            rag_retriever = None
                    finally:
                        args.rag_rebuild_index = original_rebuild
                    log_block("RAG_REBUILD", f"rebuilt_from={args.rag_docs_dir} db={args.rag_db_dir}")
                except Exception as exc:
                    log_block("RAG_REBUILD_ERROR", repr(exc))

        append_jsonl(
            Path(args.episode_results_path),
            {
                "episode": int(ep),
                "seed": int(args.seed + ep),
                "elapsed_seconds": float(time.time() - start),
                "metrics": {
                    "overall_pick_rate": float(metrics.get("overall_pick_rate", 0.0)),
                    "global_episode_return": float(metrics.get("global_episode_return", 0.0)),
                    "total_deliveries": float(metrics.get("total_deliveries", 0.0)),
                    "total_clashes": float(metrics.get("total_clashes", 0.0)),
                    "total_messages_sent": int(metrics.get("total_messages_sent", 0)),
                    "total_messages_handled": int(transcript.get("message_summary", {}).get("total_messages_handled", 0)),
                    "total_messages_activated": int(transcript.get("message_summary", {}).get("total_messages_activated", 0)),
                    "total_messages_pending_end": int(transcript.get("message_summary", {}).get("total_messages_pending_end", 0)),
                    "total_stuck": float(metrics.get("total_stuck", 0.0)),
                    "total_agvs_distance": float(metrics.get("total_agvs_distance", 0.0)),
                    "total_pickers_distance": float(metrics.get("total_pickers_distance", 0.0)),
                    "episode_length": int(metrics.get("episode_length", 0)),
                },
                "delivery_happened": bool(delivery_happened),
                "transcript_path": str(Path(args.transcript_dir) / f"episode_{ep:03d}_seed_{args.seed + ep}.json"),
                "episode_summary_path": str(summary_path),
                "reflection_note_path": str(reflection_note_path) if reflection_note_path is not None else "",
            },
        )

    llm_avg = summarize_policy_results("LLM", llm_rows)
    log_block("RESULT", f"delivery_happened_any_episode={llm_delivery_any}")

    # Heuristic baseline run intentionally disabled so this script only writes/prints LLM episode results.
    # Re-enable by restoring the block below if you want LLM-vs-heuristic comparison in the same run.
    # if not args.skip_heuristic:
    #     heuristic_env = gym.make(args.env_id, max_steps=args.max_steps_per_episode)
    #     heuristic_env.unwrapped.verbose_events = bool(args.warehouse_event_logs)
    #     heuristic_rows: List[Dict[str, Any]] = []
    #     for ep in range(args.num_episodes):
    #         start = time.time()
    #         infos, global_ret, ep_rets = heuristic_episode(
    #             heuristic_env.unwrapped,
    #             render=args.render,
    #             seed=args.seed + ep,
    #             save_gif=False,
    #         )
    #         metrics = info_statistics(infos, global_ret, ep_rets)
    #         print_episode_result("HEURISTIC", ep, time.time() - start, metrics)
    #         heuristic_rows.append(metrics)
    #     heuristic_avg = summarize_policy_results("HEURISTIC", heuristic_rows)
    #
    #     print("\nCOMPARISON:\n")
    #     print(f"Pick Rate delta (LLM-HEURISTIC): {llm_avg['overall_pick_rate'] - heuristic_avg['overall_pick_rate']:.2f}")
    #     print(f"Return delta (LLM-HEURISTIC): {llm_avg['global_episode_return'] - heuristic_avg['global_episode_return']:.2f}")
    #     print(f"Deliveries delta (LLM-HEURISTIC): {llm_avg['total_deliveries'] - heuristic_avg['total_deliveries']:.2f}")
