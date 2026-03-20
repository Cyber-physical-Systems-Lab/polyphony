import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_summary_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        print(f"[WARN] Failed to load summary: {path} -> {exc}")
        return None


def collect_from_session(session_dir: Path, root_dir: Path, prompt_style: str, objective: str):
    rows = []
    for summary_path in session_dir.rglob("summary.json"):
        rel = summary_path.parent.relative_to(root_dir)
        # rel: something like battery/central_llm_experiments/natural/session_x/model/scenario/language
        # find model, scenario, maybe prompt format from path
        parts = rel.parts
        model = None
        scenario = None
        session = None
        prompt = prompt_style
        # heuristics for path components
        if "session" in parts[0]:
            session = parts[0]
        else:
            # if included objective and structure
            for p in parts:
                if p.startswith("session_"):
                    session = p
                    break
        # model and scenario extraction
        for p in parts:
            if p in {"JSON", "natural", "language", "json"}:
                continue
            if p.startswith("session_"):
                continue
            if p.endswith("_log"):
                continue
        # fallback: path structure near end
        if len(parts) >= 3:
            scenario = parts[-2]
            model = parts[-4] if len(parts) >= 4 else None
        # better parse: find known model names by heuristic
        # We can also get from summary
        summary = load_summary_file(summary_path)
        if summary is None:
            continue
        model = summary.get("model", model)
        scenario = summary.get("scenario", scenario)
        row = {
            "objective": objective,
            "prompt_style": prompt_style,
            "session": session,
            "model": model,
            "scenario": scenario,
            "summary_path": str(summary_path),
            "run_dir": str(summary_path.parent),
        }
        # common metrics
        for key in [
            "total_shelf_deliveries",
            "steps_executed",
            "llm_calls",
            "llm_failures",
            "llm_missing_or_invalid_actions",
            "elapsed_seconds",
            "max_steps",
            "seed",
        ]:
            row[key] = summary.get(key)
        # add any battery metrics
        for bkey in ["battery_usage", "average_battery_level", "battery_degradation"]:
            if bkey in summary:
                row[bkey] = summary.get(bkey)
        rows.append(row)
    return rows


def collect_data(results_root: Path):
    rows = []
    for objective_dir in results_root.iterdir():
        if not objective_dir.is_dir():
            continue
        objective = objective_dir.name
        # Search for session dirs and prompt style dirs
        # expected: objective/central_llm_experiments/{JSON,natural}/session_x/.../summary.json
        for prompt_style_dir in objective_dir.rglob("*"):
            if not prompt_style_dir.is_dir():
                continue
            # find if this is prompt style folder by name
            if prompt_style_dir.name.lower() in {"json", "natural"}:
                prompt_style = prompt_style_dir.name.lower()
                # find all sessions beneath
                for session_dir in prompt_style_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    rows.extend(collect_from_session(session_dir, results_root, prompt_style, objective))
    df = pd.DataFrame(rows)
    return df


def sanitize_df(df: pd.DataFrame):
    # Convert numeric columns
    num_cols = [
        "total_shelf_deliveries",
        "steps_executed",
        "llm_calls",
        "llm_failures",
        "llm_missing_or_invalid_actions",
        "elapsed_seconds",
        "max_steps",
        "seed",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summary_tables(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("No data found to analyze.")
        return

    # Save all rows
    df.to_csv(out / "all_runs.csv", index=False)

    # Shelf performance: each objective+prompt+model+scenario
    pivot = (
        df.groupby(["objective", "prompt_style", "model", "scenario"])
        .agg(
            runs=("summary_path", "count"),
            deliveries=("total_shelf_deliveries", "mean"),
            llm_calls=("llm_calls", "mean"),
            failures=("llm_failures", "mean"),
            elapsed_s=("elapsed_seconds", "mean"),
        )
        .reset_index()
    )
    pivot.to_csv(out / "model_scenario_summary.csv", index=False)

    # Compare JSON vs natural for same model/scenario/objective
    compare = (
        df.pivot_table(
            index=["objective", "model", "scenario"],
            columns="prompt_style",
            values=["total_shelf_deliveries", "llm_calls", "llm_failures", "elapsed_seconds"],
            aggfunc="mean",
        )
        .reset_index()
    )
    compare.columns = ["_" .join(filter(None, map(str, c))).strip("_") for c in compare.columns]
    compare.to_csv(out / "json_vs_natural_comparison.csv", index=False)

    print(f"Saved summary CSVs in {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results and extract core metrics.")
    parser.add_argument("--results_dir", default="results", help="Root results folder")
    parser.add_argument("--output_dir", default="results/analysis", help="Output directory for extracted CSVs")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    df = collect_data(root)
    df = sanitize_df(df)
    summary_tables(df, Path(args.output_dir))

    print("Done. Run analyses on generated CSVs in", args.output_dir)


if __name__ == "__main__":
    main()
