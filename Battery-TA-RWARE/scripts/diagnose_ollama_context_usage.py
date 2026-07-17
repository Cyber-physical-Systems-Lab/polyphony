#!/usr/bin/env python3
"""Replay saved prompts once to diagnose Ollama context usage.

This script is a post-hoc diagnostic, not an experiment runner. It extracts the
first saved prompt from each configured step_records.json file, sends each prompt
once to each selected Ollama model, and records the context metadata and token
counts returned by Ollama.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS: list[tuple[str, str]] = [
    ("llama3.2:1b", "SLM"),
    ("llama3.2:3b", "SLM"),
    ("mistral:7b", "MLM"),
    ("gemma3:12b", "MLM"),
    ("phi4", "LLM"),
    ("qwen2.5:14b", "LLM"),
]

PRIORITY_CONTEXT_KEYS = (
    "llama.context_length",
    "general.context_length",
    "context_length",
    "n_ctx_train",
)


@dataclass(frozen=True)
class PromptSource:
    label: str
    architecture: str
    prompt_format: str
    path: Path


PROMPT_SOURCES: list[PromptSource] = [
    PromptSource(
        label="central_json",
        architecture="centralized",
        prompt_format="json",
        path=REPO_ROOT
        / "results/calls/centralized/json/valid_session1/gemma3_12b/"
        / "medium_balanced_4v4/json/step_records.json",
    ),
    PromptSource(
        label="central_natural",
        architecture="centralized",
        prompt_format="natural",
        path=REPO_ROOT
        / "results/calls/centralized/natural/session_20260423_193725/"
        / "llama3.2_1b/medium_balanced_4v4/language/step_records.json",
    ),
    PromptSource(
        label="shared_natural",
        architecture="shared_context",
        prompt_format="natural",
        path=REPO_ROOT
        / "results/calls/shared_context/natural/session_20260329_094333/"
        / "gemma3_12b/medium_balanced_4v4/language/step_records.json",
    ),
    PromptSource(
        label="shared_json",
        architecture="shared_context",
        prompt_format="json",
        path=REPO_ROOT
        / "results/calls/shared_context/json/session_20260401_121402/"
        / "gemma3_12b/medium_balanced_4v4/json/step_records.json",
    ),
]


CSV_FIELDS = [
    "timestamp_utc",
    "success",
    "model",
    "model_class",
    "prompt_label",
    "architecture",
    "prompt_format",
    "source_path",
    "source_step",
    "source_agent_id",
    "source_agent_type",
    "prompt_chars",
    "prompt_lines",
    "prompt_words",
    "prompt_sha256",
    "selected_context_key",
    "selected_context_window",
    "requested_num_ctx",
    "prompt_eval_count",
    "eval_count",
    "prompt_to_context_ratio",
    "context_fit_status",
    "mechanism_reported_by_ollama",
    "done_reason",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_duration_ns",
    "eval_duration_ns",
    "response_chars",
    "response_sha256",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ollama_url_for(ollama_generate_url: str, api_name: str) -> str:
    parsed = parse.urlsplit(ollama_generate_url)
    path = parsed.path or ""
    if "/api/" in path:
        prefix = path.split("/api/", 1)[0]
    else:
        prefix = path.rstrip("/")
    api_path = f"{prefix}/api/{api_name}" if prefix else f"/api/{api_name}"
    return parse.urlunsplit((parsed.scheme, parsed.netloc, api_path, "", ""))


def post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pull_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def remove_model(model: str) -> None:
    subprocess.run(["ollama", "rm", model], check=True)


def extract_context_from_parameters(parameters: Any) -> int | None:
    if not isinstance(parameters, str):
        return None
    for raw_line in parameters.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 3 and parts[0].upper() == "PARAMETER" and parts[1] == "num_ctx":
            try:
                value = int(parts[2])
            except ValueError:
                continue
            if value > 0:
                return value
    return None


def parse_positive_int(value: Any) -> int | None:
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        return None
    if parsed_value > 0:
        return parsed_value
    return None


def select_context_window(show_body: dict[str, Any]) -> tuple[str | None, int | None, dict[str, Any]]:
    model_info = show_body.get("model_info", {})
    if not isinstance(model_info, dict):
        model_info = {}

    candidates: dict[str, Any] = {}
    for key, value in sorted(model_info.items()):
        if "context_length" in key or key == "n_ctx_train":
            candidates[key] = value

    for key in PRIORITY_CONTEXT_KEYS:
        value = model_info.get(key)
        if value is not None:
            parsed_value = parse_positive_int(value)
            if parsed_value is not None:
                return key, parsed_value, candidates

    family_contexts: list[tuple[str, int]] = []
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            parsed_value = parse_positive_int(value)
            if parsed_value is not None:
                family_contexts.append((key, parsed_value))
    if family_contexts:
        key, value = max(family_contexts, key=lambda item: item[1])
        return key, value, candidates

    fallback_contexts: list[tuple[str, int]] = []
    for key, value in model_info.items():
        if "context_length" in key:
            parsed_value = parse_positive_int(value)
            if parsed_value is not None:
                fallback_contexts.append((key, parsed_value))
    if fallback_contexts:
        key, value = max(fallback_contexts, key=lambda item: item[1])
        return key, value, candidates

    parameter_context = extract_context_from_parameters(show_body.get("parameters"))
    if parameter_context is not None:
        candidates["parameters.num_ctx"] = parameter_context
        return "parameters.num_ctx", parameter_context, candidates

    return None, None, candidates


def load_first_prompt(source: PromptSource) -> dict[str, Any]:
    with source.path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{source.path} does not contain a non-empty list")

    first_record = records[0]
    if not isinstance(first_record, dict):
        raise ValueError(f"first record in {source.path} is not a JSON object")

    if source.architecture == "centralized":
        prompt = first_record.get("central_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{source.path} first record has no central_prompt")
        return {
            "prompt": prompt,
            "source_step": first_record.get("step"),
            "source_agent_id": None,
            "source_agent_type": None,
        }

    query_records = first_record.get("query_records")
    if not isinstance(query_records, list) or not query_records:
        raise ValueError(f"{source.path} first record has no query_records")
    first_query = query_records[0]
    if not isinstance(first_query, dict):
        raise ValueError(f"first query record in {source.path} is not an object")
    prompt = first_query.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"first query record in {source.path} has no prompt")
    return {
        "prompt": prompt,
        "source_step": first_record.get("step"),
        "source_agent_id": first_query.get("agent_id"),
        "source_agent_type": first_query.get("agent_type"),
    }


def prompt_stats(prompt: str) -> dict[str, Any]:
    return {
        "prompt_chars": len(prompt),
        "prompt_lines": prompt.count("\n") + 1,
        "prompt_words": len(prompt.split()),
        "prompt_sha256": sha256_text(prompt),
    }


def classify_context_fit(prompt_eval_count: Any, context_window: int | None) -> tuple[str, float | None]:
    if not isinstance(prompt_eval_count, int) or context_window is None or context_window <= 0:
        return "unknown", None
    ratio = prompt_eval_count / context_window
    if prompt_eval_count > context_window:
        return "reported_prompt_tokens_exceed_context", ratio
    if ratio >= 0.95:
        return "at_or_near_context_limit", ratio
    if ratio >= 0.80:
        return "high_context_pressure", ratio
    return "fits_reported_context", ratio


def generate_once(
    *,
    model: str,
    prompt: str,
    prompt_format: str,
    ollama_url: str,
    timeout_s: int,
    temperature: float,
    num_predict: int,
    context_window: int | None,
    use_json_format: bool,
    use_ollama_raw: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if context_window is not None:
        options["num_ctx"] = context_window

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if use_ollama_raw:
        payload["raw"] = True
    if use_json_format and prompt_format == "json":
        payload["format"] = "json"

    return post_json(ollama_url, payload, timeout_s=timeout_s)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: payload.get(field) for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay four saved medium_balanced_4v4 prompts once per Ollama model and log context usage."
    )
    parser.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    parser.add_argument("--models", nargs="+", default=[model for model, _ in MODELS])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_predict", type=int, default=700)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results/context_diagnostics",
    )
    parser.add_argument("--skip-pull", action="store_true", help="Do not run `ollama pull` before testing a model.")
    parser.add_argument("--delete-after", action="store_true", help="Run `ollama rm` after testing each model.")
    parser.add_argument(
        "--no-json-format",
        action="store_true",
        help="Do not set Ollama format=json for JSON prompts. By default this mirrors the current runners.",
    )
    parser.add_argument(
        "--ollama-raw",
        action="store_true",
        help="Set raw=true in Ollama generate requests. Current experiment runners do not set this.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    jsonl_path = output_dir / "context_usage.jsonl"
    csv_path = output_dir / "context_usage.csv"

    model_class_by_name = dict(MODELS)
    unknown_models = [model for model in args.models if model not in model_class_by_name]
    if unknown_models:
        print(f"Unknown model(s): {unknown_models}", file=sys.stderr)
        return 2

    prompts: list[dict[str, Any]] = []
    for source in PROMPT_SOURCES:
        extracted = load_first_prompt(source)
        prompt = extracted["prompt"]
        prompts.append(
            {
                "label": source.label,
                "architecture": source.architecture,
                "prompt_format": source.prompt_format,
                "source_path": str(source.path),
                **{key: value for key, value in extracted.items() if key != "prompt"},
                "prompt": prompt,
                **prompt_stats(prompt),
            }
        )

    write_json(
        output_dir / "run_config.json",
        {
            "timestamp_utc": utc_now(),
            "models": args.models,
            "ollama_url": args.ollama_url,
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "timeout_s": args.timeout_s,
            "skip_pull": args.skip_pull,
            "delete_after": args.delete_after,
            "use_json_format": not args.no_json_format,
            "ollama_raw": args.ollama_raw,
            "note": "One generate request is made for each model/prompt pair.",
        },
    )
    write_json(
        output_dir / "prompt_sources.json",
        [{key: value for key, value in prompt.items() if key != "prompt"} for prompt in prompts],
    )

    show_url = ollama_url_for(args.ollama_url, "show")
    use_json_format = not args.no_json_format

    for model in args.models:
        model_dir_name = model.replace(":", "_").replace("/", "_")
        show_body: dict[str, Any] = {}
        selected_context_key: str | None = None
        context_window: int | None = None
        context_candidates: dict[str, Any] = {}

        if not args.skip_pull:
            print(f"[{utc_now()}] pulling {model}")
            pull_model(model)

        try:
            print(f"[{utc_now()}] reading Ollama metadata for {model}")
            show_body = post_json(show_url, {"name": model}, timeout_s=args.timeout_s)
            selected_context_key, context_window, context_candidates = select_context_window(show_body)
            write_json(output_dir / "model_show" / f"{model_dir_name}.json", show_body)
        except Exception as exc:
            show_body = {"error": repr(exc)}
            write_json(output_dir / "model_show" / f"{model_dir_name}.json", show_body)
            print(f"[{utc_now()}] warning: failed to read metadata for {model}: {exc}", file=sys.stderr)

        for prompt_info in prompts:
            label = prompt_info["label"]
            print(f"[{utc_now()}] generating once: model={model} prompt={label}")
            base_row: dict[str, Any] = {
                "timestamp_utc": utc_now(),
                "model": model,
                "model_class": model_class_by_name[model],
                "prompt_label": label,
                "architecture": prompt_info["architecture"],
                "prompt_format": prompt_info["prompt_format"],
                "source_path": prompt_info["source_path"],
                "source_step": prompt_info["source_step"],
                "source_agent_id": prompt_info["source_agent_id"],
                "source_agent_type": prompt_info["source_agent_type"],
                "prompt_chars": prompt_info["prompt_chars"],
                "prompt_lines": prompt_info["prompt_lines"],
                "prompt_words": prompt_info["prompt_words"],
                "prompt_sha256": prompt_info["prompt_sha256"],
                "selected_context_key": selected_context_key,
                "selected_context_window": context_window,
                "requested_num_ctx": context_window,
                "mechanism_reported_by_ollama": "not_reported",
                "context_candidates": context_candidates,
            }

            try:
                response_body = generate_once(
                    model=model,
                    prompt=prompt_info["prompt"],
                    prompt_format=prompt_info["prompt_format"],
                    ollama_url=args.ollama_url,
                    timeout_s=args.timeout_s,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    context_window=context_window,
                    use_json_format=use_json_format,
                    use_ollama_raw=args.ollama_raw,
                )
                response_text = str(response_body.get("response", ""))
                response_path = output_dir / "responses" / model_dir_name / f"{label}.txt"
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(response_text, encoding="utf-8")

                prompt_eval_count = response_body.get("prompt_eval_count")
                fit_status, ratio = classify_context_fit(prompt_eval_count, context_window)
                row = {
                    **base_row,
                    "success": True,
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": response_body.get("eval_count"),
                    "prompt_to_context_ratio": ratio,
                    "context_fit_status": fit_status,
                    "done_reason": response_body.get("done_reason"),
                    "total_duration_ns": response_body.get("total_duration"),
                    "load_duration_ns": response_body.get("load_duration"),
                    "prompt_eval_duration_ns": response_body.get("prompt_eval_duration"),
                    "eval_duration_ns": response_body.get("eval_duration"),
                    "response_chars": len(response_text),
                    "response_sha256": sha256_text(response_text),
                    "response_path": str(response_path),
                    "error": None,
                }
                append_jsonl(jsonl_path, {**row, "ollama_response_metadata": response_body})
                append_csv(csv_path, row)
            except (error.HTTPError, error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                row = {
                    **base_row,
                    "success": False,
                    "prompt_eval_count": None,
                    "eval_count": None,
                    "prompt_to_context_ratio": None,
                    "context_fit_status": "request_failed",
                    "done_reason": None,
                    "total_duration_ns": None,
                    "load_duration_ns": None,
                    "prompt_eval_duration_ns": None,
                    "eval_duration_ns": None,
                    "response_chars": None,
                    "response_sha256": None,
                    "error": repr(exc),
                }
                append_jsonl(jsonl_path, row)
                append_csv(csv_path, row)
                print(f"[{utc_now()}] error: {model} {label}: {exc}", file=sys.stderr)

        if args.delete_after:
            print(f"[{utc_now()}] deleting {model}")
            remove_model(model)

    print(f"[{utc_now()}] wrote {jsonl_path}")
    print(f"[{utc_now()}] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
