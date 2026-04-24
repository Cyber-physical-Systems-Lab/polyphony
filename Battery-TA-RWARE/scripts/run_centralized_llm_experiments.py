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
    describe_action_id_for_agent,
    get_requested_shelves,
    render_all_agents,
    waiting_agv_support_lines,
)


# A small, explicit scenario catalogue keeps the experiment understandable.
# Balanced scenarios reveal whether the model understands the normal collaboration loop.
# AGV-heavy scenarios reveal picker-support scheduling weakness.
# Picker-heavy scenarios reveal underutilization and role confusion.
# Larger warehouses reveal long-horizon planning weakness.
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


LANGUAGE_PROMPT_TEMPLATE = (
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
    "You are one central LLM.\n"
    "You must choose the next macro action for each agent that needs planning.\n"
    "When an agent status is busy=True, it is already executing a previous instruction and you do not need to plan a new action for it.\n"
    "Choose actions only from the listed candidate actions for each non-busy agent.\n"
    "If an AGV is carrying a requested shelf, prioritize GOAL.\n"
    "If an AGV needs picker support now, align Picker to that same shelf action id.\n"
    "Avoid sending healthy agents to charging unless there is a clear reason.\n"
    "\n"
    "Current global state:\n"
    "All agents:\n{all_agents}\n"
    "\n"
    "Requested shelves:\n{requested_shelves}\n"
    "\n"
    "Charging place occupancy:\n{charging_occupancy}\n"
    "\n"
    "AGVs requesting picker support:\n{agv_support_requests}\n"
    "\n"
    "Candidate actions for agents that are not busy:\n{agent_candidates}\n"
    "\n"
    "Output format:\n"
    "Reasoning: <short overall reason>\n"
    "Actions:\n"
    "agent_0 -> <action_id> | <short reason>\n"
    "agent_1 -> <action_id> | <short reason>\n"
    "...\n"
    "Rules:\n"
    "- Output one line per planned agent only.\n"
    "- For busy agents, omit the line. They will keep executing their previous plan.\n"
    "- Do not output JSON.\n"
    "- Use only candidate action ids listed above.\n"
)


JSON_PROMPT_TEMPLATE = (
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
    "You are one central LLM.\n"
    "You must choose the next macro action for each agent that needs planning.\n"
    "When an agent status is busy=true, it is already executing a previous instruction and you do not need to plan a new action for it.\n"
    "Choose actions only from the listed candidate actions for each non-busy agent.\n"
    "If an AGV is carrying a requested shelf, prioritize GOAL.\n"
    "If an AGV needs picker support now, align Picker to that same shelf action id.\n"
    "Avoid sending healthy agents to charging unless there is a clear reason.\n"
    "\n"
    "Current global state in JSON:\n"
    "{state_json}\n"
    "\n"
    "Return valid JSON only in this exact shape:\n"
    "{{\"reasoning\": \"short overall reason\", \"actions\": [{{\"agent_id\": 0, \"action\": 12, \"reason\": \"short reason\"}}]}}\n"
    "Only include agents that need planning.\n"
    "Use only candidate action ids listed in the JSON state.\n"
)


parser = ArgumentParser(
    description="Run centralized LLM coordination experiments across models and warehouse scenarios",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--seed", type=int, default=0, help="Base episode seed")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
parser.add_argument("--prompt_format", choices=["language", "json"], default="language", help="Prompt/output format")
parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama generate endpoint")
parser.add_argument("--request_timeout_s", type=int, default=60, help="HTTP timeout for Ollama generation")
parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
parser.add_argument("--num_predict", type=int, default=900, help="Generation token budget")
parser.add_argument("--results_dir", type=str, default="results/central_llm_experiments", help="Directory for logs and summaries")
parser.add_argument("--session_dir", type=str, default=None, help="Optional existing session directory path to resume into")
parser.add_argument("--resume_from_model", type=str, default=None, help="Model name (exact string) to resume from")
parser.add_argument("--resume_from_scenario", type=str, default=None, help="Scenario label to resume from when resuming a model")
parser.add_argument("--clean_resumed_model", action="store_true", help="If set, remove existing run results for the resumed model/scenario before restart")
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


def build_agent_candidate_views(env, valid_masks: np.ndarray) -> Tuple[str, Dict[str, List[str]], List[int]]:
    blocks: List[str] = []
    structured: Dict[str, List[str]] = {}
    planned_agent_indices: List[int] = []

    for idx, agent in enumerate(env.agents):
        if bool(agent.busy):
            continue
        candidate_ids = candidate_ids_for_prompt(env, idx, valid_masks, 0)
        candidate_lines = [
            describe_action_id_for_agent(env, agent, action_id)
            for action_id in candidate_ids
            if valid_masks[idx, action_id] > 0
        ]
        if not candidate_lines:
            candidate_lines = ["0:NOOP (distance_steps=0)"]
        structured[f"agent_{idx}"] = candidate_lines
        blocks.append(
            f"agent_{idx} ({agent.type.name}):\n"
            + "\n".join(f"  - {line}" for line in candidate_lines)
        )
        planned_agent_indices.append(idx)

    return ("\n".join(blocks) if blocks else "- none"), structured, planned_agent_indices


def build_central_prompt(
    env,
    valid_masks: np.ndarray,
    scenario: ScenarioSpec,
    prompt_format: str,
) -> Tuple[str, List[int]]:
    del scenario
    agent_candidates_text, agent_candidates_json, planned_agent_indices = build_agent_candidate_views(env, valid_masks)
    state_json = json.dumps(
        {
            "all_agents": render_all_agents(env),
            "requested_shelves": get_requested_shelves(env),
            "charging_place_occupancy": charging_station_occupancy(env),
            "agvs_requesting_picker_support": waiting_agv_support_lines(env),
            "candidate_actions_for_non_busy_agents": agent_candidates_json,
        },
        indent=2,
        ensure_ascii=True,
    )
    fields = {
        "stated_objective": STATED_OBJECTIVE,
        "all_agents": format_bullets(render_all_agents(env)),
        "requested_shelves": format_bullets(get_requested_shelves(env)),
        "charging_occupancy": format_bullets(charging_station_occupancy(env)),
        "agv_support_requests": format_bullets(waiting_agv_support_lines(env)),
        "agent_candidates": agent_candidates_text,
        "state_json": state_json,
    }
    template = JSON_PROMPT_TEMPLATE if prompt_format == "json" else LANGUAGE_PROMPT_TEMPLATE
    return render_template(template, fields), planned_agent_indices


def parse_language_actions(text: str, num_agents: int) -> Dict[int, Dict[str, Any]]:
    parsed: Dict[int, Dict[str, Any]] = {}
    pattern = re.compile(
        r"agent[_\s-]*(\d+)\s*[-:=]>\s*(-?\d+)(?:\s*\|\s*(.+))?",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        agent_idx = int(match.group(1))
        action_id = int(match.group(2))
        reason = (match.group(3) or "").strip()
        if 0 <= agent_idx < num_agents:
            parsed[agent_idx] = {"action": action_id, "reason": reason}
    return parsed


def parse_json_actions(text: str, num_agents: int) -> Dict[int, Dict[str, Any]]:
    parsed: Dict[int, Dict[str, Any]] = {}
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        if start >= 0:
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
                        text = text[start : idx + 1]
                        break
    payload = json.loads(text)
    for item in payload.get("actions", []):
        if not isinstance(item, dict):
            continue
        agent_idx = int(item.get("agent_id", -1))
        action_id = int(item.get("action", 0))
        reason = str(item.get("reason", "")).strip()
        if 0 <= agent_idx < num_agents:
            parsed[agent_idx] = {"action": action_id, "reason": reason}
    return parsed


def parse_actions(text: str, prompt_format: str, num_agents: int) -> Dict[int, Dict[str, Any]]:
    if prompt_format == "json":
        return parse_json_actions(text, num_agents)
    return parse_language_actions(text, num_agents)


def parse_actions_with_metadata(
    text: str,
    prompt_format: str,
    num_agents: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "parsed_llm_output": None,
        "json_generation_failed": False,
        "json_parse_error": None,
    }
    if prompt_format == "json":
        try:
            parsed_actions = parse_json_actions(text, num_agents)
            metadata["parsed_llm_output"] = {"actions": parsed_actions}
            return parsed_actions, metadata
        except Exception as exc:
            metadata["json_generation_failed"] = True
            metadata["json_parse_error"] = f"{type(exc).__name__}: {exc}"
            parsed_actions = parse_language_actions(text, num_agents)
            metadata["parsed_llm_output"] = {
                "actions": parsed_actions,
                "fallback_parser": "language_action_lines",
            }
            return parsed_actions, metadata

    parsed_actions = parse_language_actions(text, num_agents)
    metadata["parsed_llm_output"] = {"actions": parsed_actions}
    return parsed_actions, metadata


def discard_invalid_action(valid_masks: np.ndarray, agent_idx: int) -> int:
    if 0 <= agent_idx < valid_masks.shape[0] and valid_masks[agent_idx, 0] > 0:
        return 0
    valid = np.where(valid_masks[agent_idx] > 0)[0].tolist()
    return int(valid[0]) if valid else 0


def materialize_macro_actions(
    env,
    valid_masks: np.ndarray,
    parsed_actions: Dict[int, Dict[str, Any]],
    planned_agent_indices: List[int],
) -> Tuple[List[int], Dict[str, int], List[Dict[str, Any]]]:
    macro_actions = [0] * env.num_agents
    metrics = {
        "plannable_agents": len(planned_agent_indices),
        "busy_agents": 0,
        "llm_missing_or_invalid_actions": 0,
    }
    resolution_rows: List[Dict[str, Any]] = []

    for idx, agent in enumerate(env.agents):
        if bool(agent.busy):
            metrics["busy_agents"] += 1
            resolution_rows.append(
                {
                    "agent_id": idx,
                    "status": "busy_kept_running",
                    "parsed_action": parsed_actions.get(idx, {}).get("action"),
                    "executed_action": 0,
                }
            )
            continue

        parsed = parsed_actions.get(idx)
        proposed = int(parsed["action"]) if parsed is not None else 0
        if proposed < 0 or proposed >= env.action_size or valid_masks[idx, proposed] <= 0:
            fallback = discard_invalid_action(valid_masks, idx)
            macro_actions[idx] = fallback
            metrics["llm_missing_or_invalid_actions"] += 1
            resolution_rows.append(
                {
                    "agent_id": idx,
                    "status": "discarded_invalid_action_to_noop",
                    "parsed_action": proposed if parsed is not None else None,
                    "executed_action": fallback,
                }
            )
            continue

        macro_actions[idx] = proposed
        resolution_rows.append(
            {
                "agent_id": idx,
                "status": "accepted_llm_action",
                "parsed_action": proposed,
                "executed_action": proposed,
            }
        )

    return macro_actions, metrics, resolution_rows


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
    llm_failures = 0
    total_llm_missing_or_invalid_actions = 0
    total_json_generation_failures = 0
    start_time = time.time()

    for step_idx in range(1, max(1, int(args.max_steps)) + 1):
        base_env = env.unwrapped
        valid_masks = safe_valid_action_masks(base_env)
        prompt, planned_agent_indices = build_central_prompt(base_env, valid_masks, scenario, args.prompt_format)

        if planned_agent_indices:
            total_llm_calls += 1
            llm_raw_text = ""
            parse_metadata: Dict[str, Any] = {
                "parsed_llm_output": None,
                "json_generation_failed": False,
                "json_parse_error": None,
            }
            try:
                llm_raw_text = query_ollama_text(
                    model=model,
                    ollama_url=args.ollama_url,
                    prompt=prompt,
                    timeout_s=args.request_timeout_s,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                )
                llm_text = llm_raw_text
                parsed_actions, parse_metadata = parse_actions_with_metadata(
                    llm_text,
                    args.prompt_format,
                    base_env.num_agents,
                )
            except Exception as exc:
                llm_failures += 1
                llm_text = f"LLM_ERROR: {type(exc).__name__}: {exc}"
                parsed_actions = {}
            else:
                llm_raw_text = llm_text
        else:
            llm_text = "No idle agents required replanning at this step."
            llm_raw_text = llm_text
            parsed_actions = {}
            parse_metadata = {
                "parsed_llm_output": None,
                "json_generation_failed": False,
                "json_parse_error": None,
            }

        if parse_metadata.get("json_generation_failed"):
            total_json_generation_failures += 1

        macro_actions, planner_metrics, resolution_rows = materialize_macro_actions(
            base_env,
            valid_masks,
            parsed_actions,
            planned_agent_indices,
        )
        total_llm_missing_or_invalid_actions += planner_metrics["llm_missing_or_invalid_actions"]

        _, _, terminateds, truncateds, info = env.step(macro_actions)
        if args.render:
            env.render()

        step_deliveries = int(info.get("shelf_deliveries", 0))
        if step_deliveries > 0:
            total_deliveries += step_deliveries
            delivery_timeline.append(
                {
                    "step": step_idx,
                    "total_shelf_deliveries": total_deliveries,
                    "step_shelf_deliveries": step_deliveries,
                }
            )
            log(
                f"Delivery milestone step={step_idx} cumulative_deliveries={total_deliveries}",
                log_path,
            )

        step_records.append(
            {
                "step": step_idx,
                "prompt": prompt,
                "llm_output": llm_text,
                "raw_llm_output": llm_raw_text,
                "parsed_llm_output": parse_metadata.get("parsed_llm_output"),
                "parsed_actions": parsed_actions,
                "json_generation_failed": bool(parse_metadata.get("json_generation_failed")),
                "json_parse_error": parse_metadata.get("json_parse_error"),
                "actions": macro_actions,
                "planner_metrics": planner_metrics,
                "action_resolution": resolution_rows,
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
        "llm_failures": llm_failures,
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
        f"Completed scenario={scenario.label} model={model} deliveries={total_deliveries} steps={len(step_records)} failures={llm_failures}",
        log_path,
    )
    return summary


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
    if args.resume_from_model is not None:
        resume_model_idx = find_model_index(args.resume_from_model)
    if args.resume_from_scenario is not None:
        resume_scenario_idx = find_scenario_index(args.resume_from_scenario)

    for model_idx, (model, model_class) in enumerate(MODELS):
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
                if model_idx == resume_model_idx and scenario_idx < resume_scenario_idx:
                    continue
                run_dir = session_dir / model_safe / scenario.label / args.prompt_format
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.clean_resumed_model and model_idx == resume_model_idx and scenario_idx >= resume_scenario_idx:
                    # remove any previous outputs for resumed scenario(s) to avoid stale data
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
