"""Freeze the accepted configuration and estimate the proposed test run without model calls."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = "results/dev/20261123T084741Z-937c83bd/summary.json"
DEV_STATE = "results/dev/20261123T084741Z-937c83bd/state_diagnostics.json"
DATA_LOCK = "data/split-lock.json"
TEMPLATE = "baselines/MemoryAgentBench/utils/templates.py"
BASELINE = "baseline"
FLASH = "deepseek-flash"
PRO = "deepseek-v4-pro"
PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
VARIANTS = {
    "baseline": {},
    "full": {},
    "no_dual_layer": {"dual_layer": False},
    "no_conflict_unlearning": {"conflict_unlearning": False},
    "no_validation_gate": {"validation_gate": False},
    "threshold_3": {"conflict_min_distinct": 3},
    "threshold_4": {"conflict_min_distinct": 4},
}
RATES = {
    FLASH: {"cache_hit": "0.006", "cache_miss": "0.30", "completion": "1.20"},
    PRO: {"cache_hit": "0.044", "cache_miss": "1.32", "completion": "3.96"},
}
LATENCY_SCENARIOS_S = {
    FLASH: {"32k": [2, 8], "64k": [3, 15], "262k": [8, 60]},
    PRO: {"32k": [3, 15], "64k": [5, 25], "262k": [12, 90]},
}
CONCURRENCY = 8


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_frozen(path, value):
    path = ROOT / path
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise SystemExit(f"Existing freeze or budget artifact differs: {path}")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", required=True, type=int, choices=[5], help="Check the accepted five-case dev run.")
    args = parser.parse_args()
    source = load(DEV)
    state = load(DEV_STATE)
    lock = load(DATA_LOCK)
    upstream = ROOT / "baselines/MemoryAgentBench"
    git = lambda *command: subprocess.run(
        ["git", "-C", str(upstream), *command], text=True, capture_output=True, check=True,
    ).stdout.strip()
    if git("rev-parse", "HEAD") != source["upstream_commit"] or git("status", "--porcelain", "-uno"):
        raise SystemExit("Upstream has drifted from the accepted baseline.")
    if source["complete_pairs"] != args.limit or any(arm["calls_missing_usage"] for arm in source["arms"].values()):
        raise SystemExit("A complete accepted dev smoke is required.")
    files = {**source["code_sha256"], TEMPLATE: source["templates_sha256"], DATA_LOCK: source["split_lock_sha256"]}
    for path, expected in files.items():
        if digest(path) != expected:
            raise SystemExit(f"Accepted input changed: {path}")

    by_length = Counter()
    for name, count in lock["partitions"]["test"]["by_source"].items():
        by_length[name.rsplit("_", 1)[1]] += count
    n = sum(by_length.values())
    schedule = {FLASH: list(VARIANTS), PRO: ["baseline", "full"]}
    freeze = {
        "format": 1, "accepted_dev_source": DEV, "accepted_dev_sha256": digest(DEV),
        "files_sha256": files, "upstream_commit": source["upstream_commit"],
        "test_file_sha256": {name: value for name, value in lock["files"].items() if name.startswith("test/")},
        "test_size": n, "test_contexts": lock["partitions"]["test"]["unique_contexts"],
        "prompt_template_source": TEMPLATE, "timestamp_marker": "2000-01-01 00:00:00",
        "settings": {name: source[name] for name in (
            "metric", "temperature", "max_tokens", "chunk_size", "retrieval_top_k", "tools", "seed", "provider_seed",
        )} | {"thinking": "disabled"},
        "full_method": source["config"], "variants": VARIANTS, "schedule": schedule,
        "execution": {"split": "test", "seeds": [42], "concurrency": CONCURRENCY, "timeout_seconds": 120,
                      "max_retries": 0, "aggregate_budget_cap_usd": "400", "requires_user_confirmation": True},
        "statistics": {
            "main_comparison": "paired Full vs Baseline within each model",
            "interval": "percentile bootstrap, resampling entire context groups",
            "bootstrap_draws": 10000, "bootstrap_seed": 42,
            "case_selection": ["full_only_correct", "baseline_only_correct", "both_wrong", "both_correct"],
            "sensitivity": "Fixed 2/3/4 grid; frozen Full remains 2 regardless of test scores.",
        },
    }
    write_frozen("results/test_plan/freeze.json", freeze)

    nominal = sum(int(length[:-1]) * 1024 * count for length, count in by_length.items())
    base_ratio = Decimal(source["arms"]["baseline"]["prompt_tokens"]) / args.limit / 6144
    short_ratio = Decimal(source["arms"]["method"]["prompt_tokens"]) / source["arms"]["baseline"]["prompt_tokens"]
    dev_profile = {row["min_distinct"]: row["archived"] for row in state["offline_threshold_profile"]}
    shortened = {"full", "no_dual_layer", "no_validation_gate"}
    rows = []
    for model, variants in schedule.items():
        rates = {name: Decimal(value) for name, value in RATES[model].items()}
        for variant in variants:
            ratio = short_ratio if variant in shortened else Decimal(1)
            raw_input = round(nominal * base_ratio * ratio)
            no_savings_input = round(nominal * base_ratio)
            output = n * source["max_tokens"]
            peak = lambda hit, prompt: (
                prompt * (rates["cache_hit"] * hit + rates["cache_miss"] * (1 - hit)) + output * rates["completion"]
            ) / 1_000_000
            wait = [sum(count * LATENCY_SCENARIOS_S[model][length][i] for length, count in by_length.items())
                    for i in (0, 1)]
            rows.append({
                "model": model, "variant": variant, "planned_calls": n, "estimated_prompt_tokens": raw_input,
                "estimated_prompt_tokens_no_savings": no_savings_input, "completion_token_cap": output,
                "peak_usd_no_cache": str(peak(Decimal(0), raw_input)),
                "peak_usd_50_percent_cache": str(peak(Decimal("0.5"), raw_input)),
                "peak_usd_no_savings_no_cache": str(peak(Decimal(0), no_savings_input)),
                "estimated_request_seconds": wait,
            })
    totals = {key: sum(row[key] for row in rows) for key in (
        "planned_calls", "estimated_prompt_tokens", "estimated_prompt_tokens_no_savings", "completion_token_cap",
    )}
    totals.update({key: str(sum(Decimal(row[key]) for row in rows)) for key in (
        "peak_usd_no_cache", "peak_usd_50_percent_cache", "peak_usd_no_savings_no_cache",
    )})
    totals["wall_hours_with_20pct_buffer"] = [
        round(sum(row["estimated_request_seconds"][i] for row in rows) / CONCURRENCY / 3600 * 1.2, 2)
        for i in (0, 1)
    ]
    budget = {
        "kind": "planning_estimate_no_test_inference", "dev_usage_source": DEV, "dev_state_source": DEV_STATE,
        "data_source": DATA_LOCK, "freeze_sha256": digest("results/test_plan/freeze.json"),
        "pricing_source": PRICING_SOURCE, "peak_rates_per_million": RATES,
        "cases_by_nominal_length": dict(by_length), "nominal_length_token_unit": 1024,
        "timing_scenarios_seconds_per_request_are_assumptions": LATENCY_SCENARIOS_S,
        "latency_concurrency": CONCURRENCY, "latency_buffer_fraction": 0.2,
        "dev_archived_by_threshold": dev_profile, "baseline_prompt_to_nominal_multiplier": str(base_ratio),
        "method_to_baseline_prompt_ratio_on_dev": str(short_ratio),
        "cells": rows, "totals": totals,
    }
    write_frozen("results/test_plan/estimate.json", budget)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
