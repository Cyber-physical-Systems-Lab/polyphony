#!/usr/bin/env python3
"""Extract battery-safety metrics already stored in experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OBJECTIVE_NAMES = {
    "Maximize total shelf deliveries while keeping agents charged enough to avoid critical battery levels.": "Battery + Shelf",
    "Optimize Battery Life: keep agents charged, prioritize safe charging decisions, and never let any agent go below the critical battery threshold.": "Battery",
}

MODEL_NAMES = {
    "llama3.2:1b": "llama 1b (3.2)",
    "llama3.2:3b": "llama 3b (3.2)",
    "mistral:7b": "mistral 7b",
    "gemma3:12b": "gemma 12b",
    "phi4": "phi4",
    "qwen2.5:14b": "qwen 14b (2.5)",
}

FIELDNAMES = [
    "Objective",
    "Arch",
    "Prompt",
    "Model",
    "Scenario",
    "Seed",
    "run_id",
    "steps_executed",
    "n_agents",
    "total_deliveries",
    "charging_utilization_fraction",
    "charging_utilization_percent",
    "critical_battery_threshold",
    "min_battery_observed",
    "agents_ever_below_30",
    "agent_steps_below_30",
    "total_agent_steps",
    "agent_step_fraction_below_30",
    "agents_reaching_zero",
    "run_reached_zero",
    "per_agent_min_battery",
    "summary_json_path",
]


def architecture_from_path(path: Path) -> str:
    lowered_parts = {part.lower() for part in path.parts}
    if "centralized" in lowered_parts or "central_llm_experiments" in lowered_parts:
        return "Centralized"
    if "shared" in lowered_parts or "shared_context" in lowered_parts:
        return "Shared"
    return ""


def run_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("session"):
            return part
    return str(path.parent)


def ordered_values(mapping: Any) -> list[float]:
    if not isinstance(mapping, dict):
        return []

    def agent_number(item: tuple[str, Any]) -> int:
        key = item[0]
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    return [float(value) for _, value in sorted(mapping.items(), key=agent_number)]


def extract_row(summary_path: Path, results_root: Path) -> dict[str, Any] | None:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    objective = OBJECTIVE_NAMES.get(str(summary.get("stated_objective", "")))
    if objective is None or "min_battery_observed" not in summary:
        return None

    per_agent_min = ordered_values(summary.get("per_agent_min_battery"))
    n_agents = len(per_agent_min)
    steps = int(summary.get("steps_executed", 0) or 0)
    agent_steps_below = int(summary.get("below_required_min_charge_events", 0) or 0)
    total_agent_steps = steps * n_agents
    fraction_below = (
        agent_steps_below / total_agent_steps if total_agent_steps > 0 else ""
    )
    agents_reaching_zero = sum(value <= 0 for value in per_agent_min)
    utilization_fraction = float(summary.get("charging_station_utilization_rate", 0) or 0)
    prompt = "JSON" if str(summary.get("prompt_format", "")).upper() == "JSON" else "Natural"
    raw_model = str(summary.get("model", ""))

    return {
        "Objective": objective,
        "Arch": architecture_from_path(summary_path),
        "Prompt": prompt,
        "Model": MODEL_NAMES.get(raw_model, raw_model),
        "Scenario": summary.get("scenario", ""),
        "Seed": summary.get("seed", ""),
        "run_id": run_id_from_path(summary_path),
        "steps_executed": steps,
        "n_agents": n_agents,
        "total_deliveries": int(summary.get("total_shelf_deliveries", 0) or 0),
        "charging_utilization_fraction": utilization_fraction,
        "charging_utilization_percent": utilization_fraction * 100,
        "critical_battery_threshold": float(
            summary.get("critical_battery_threshold", 30) or 30
        ),
        "min_battery_observed": summary.get("min_battery_observed", ""),
        "agents_ever_below_30": len(
            summary.get("agents_ever_below_required_min_charge", []) or []
        ),
        "agent_steps_below_30": agent_steps_below,
        "total_agent_steps": total_agent_steps,
        "agent_step_fraction_below_30": fraction_below,
        "agents_reaching_zero": agents_reaching_zero,
        "run_reached_zero": int(agents_reaching_zero > 0),
        "per_agent_min_battery": ";".join(f"{value:g}" for value in per_agent_min),
        "summary_json_path": str(summary_path.relative_to(results_root.parent)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "battery_safety_metrics_extracted.csv",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    rows = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        row = extract_row(summary_path, results_root)
        if row is not None:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
