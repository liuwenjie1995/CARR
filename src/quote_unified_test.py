"""Offline formal-test quote calibrated to recorded dev usage; never calls a model."""

import argparse
import json
from decimal import Decimal as D
from pathlib import Path

from src.evaluator import ROOT, dump_json
from src.run_unified_dev import BASE
from src.v2_report import load, write_csv
from src.v2_transport import FLASH, quote

V2 = ROOT / "results/dev/v2-20261123T104058Z-5c9630"
V4 = ROOT / "results/dev/hop-v4-20261123T113918722799Z-paid"
PRICES = "https://api-docs.deepseek.com/quick_start/pricing/"


def relative(path):
    return path.relative_to(ROOT).as_posix()


def estimate(dev, limit):
    dev = dev.resolve()
    if not dev.is_relative_to((ROOT / "results/dev").resolve()):
        raise ValueError("Expected development run logs.")
    manifest = load(dev / "manifest.json")
    v6 = load(dev / "summary.json")
    if manifest["status"] != "complete" or (limit == 25 and len(manifest["cohort"]) != 25):
        raise ValueError("Need a completed v6 dev run.")
    unified = next(row for row in v6["arms"] if row["arm"] == "unified" and row["group"] == "all")
    if unified["questions"] < limit or unified["missing_usage"] or not unified["requests"]:
        if limit != 5:
            raise ValueError("Complete paid development usage is required for a formal quote.")
        # Five-item smoke only validates metadata and rate arithmetic. No projected result is emitted.
        return {"smoke_checked_cases": len(manifest["cohort"]), "wrote_formal_quote": False, "api_calls": 0}

    fc = load(ROOT / "data/split-lock.json")["partitions"]["test"]
    ar = load(ROOT / "data/eventqa_v2/split-lock.json")["partitions"]["test"]
    n, n_event = fc["samples"], ar["samples"]
    mh = sum(n for source, n in fc["by_source"].items() if "_mh_" in source)
    v3 = {row["arm"]: row for row in load(BASE / "summary.json")["arms"]}
    event = next(row for row in load(V2 / "summary.json")["arms"]
                 if row["dataset"] == "eventqa" and row["arm"] == "full_recency")
    plans = [load(path) for path in (V4 / "requests").glob("plan-*.json")]
    plan_prompt = sum(row["usage"]["prompt_tokens"] for row in plans)
    nominal_scale = sum(D(name.rsplit("_", 1)[1][:-1]) * count / D(6) for name, count in fc["by_source"].items())
    cells = []

    def add(arm, count, prompt, output, rate, source):
        prompt, output = round(prompt), round(output)
        cached = round(D(prompt) * rate)
        cells.append({"arm": arm, "max_calls_modeled": count, "estimated_prompt_tokens": prompt,
                      "completion_token_cap": output, "observed_dev_cache_hit_fraction": str(rate),
                      "peak_no_cache_usd": str(quote(FLASH, prompt, output)),
                      "offpeak_at_dev_cache_usd": str(quote(FLASH, prompt, output, cached) / 2),
                      "calibration_source": source})

    def from_v3(name, key, weight):
        row = v3[key]
        add(name, n, D(row["answer_prompt_tokens"]) / row["n"] * weight, n * 10,
            D(row["answer_cache_hit_tokens"]) / row["answer_prompt_tokens"], relative(BASE / "summary.json"))

    compact_prompt = D(unified["prompt_tokens"]) / unified["requests"] * n
    compact_rate = D(unified["cache_hit_tokens"]) / unified["prompt_tokens"]
    from_v3("official_full_context", "full", nominal_scale)
    add("length_matched_recent_tail", n, compact_prompt, n * 10, compact_rate, relative(dev / "summary.json"))
    from_v3("plain_bge_top8", "rag", D(n))
    from_v3("rule_archive_full_context", "rule_full", nominal_scale)
    for name in ("raw_exact_latest_relay", "unified_rule_archive_latest_relay"):
        add(name, n, compact_prompt, n * 10, compact_rate, relative(dev / "summary.json"))
    add("shared_multihop_planner", mh, D(plan_prompt) / len(plans) * mh, 256 * mh, D(0), relative(V4 / "requests"))
    for name in ("eventqa_original", "eventqa_after_rule_preservation_check"):
        add(name, n_event, D(event["answer_prompt_tokens"]) / event["questions"] * n_event, n_event * 40,
            D(event["answer_cache_hit_tokens"]) / event["answer_prompt_tokens"], relative(V2 / "summary.json"))

    total_fields = ("peak_no_cache_usd", "offpeak_at_dev_cache_usd")
    sums = {key: str(sum(D(row[key]) for row in cells)) for key in total_fields}
    with_event_dedup = {key: str(D(sums[key]) - D(cells[-1][key])) for key in total_fields}
    historical_plan = load(ROOT / "results/test_plan/estimate.json")
    long_work = historical_plan["cells"][0]["estimated_request_seconds"]
    # Serial seconds assuming one request per case, divided by planned parallelism.
    event_latency, compact_latency, planning_latency = (3, 15), (0.6, 3), (0.5, 2.5)
    ranges = []
    for i in (0, 1):
        seconds = (2 * long_work[i] + 2 * n_event * event_latency[i] + 4 * n * compact_latency[i]
                   + mh * planning_latency[i])
        ranges.append(round(seconds * 1.2 / 8 / 3600 + (5 if i == 0 else 15) / 60, 2))
    result = {
        "kind": "offline_estimate_not_a_test_result", "api_calls_to_make_quote": 0,
        "generator": "src/quote_unified_test.py", "source": relative(dev),
        "factcon_questions": n, "eventqa_questions": n_event, "planned_shared_decompositions": mh,
        "max_calls_modeled": sum(row["max_calls_modeled"] for row in cells),
        "rates": PRICES, "total": sums, "total_if_eventqa_requests_deduplicate": with_event_dedup,
        "input_tokens_modeled": sum(row["estimated_prompt_tokens"] for row in cells),
        "output_token_cap": sum(row["completion_token_cap"] for row in cells),
        "wall_clock_hours_at_concurrency_8": ranges,
        "assumptions": [
            "FactCon long arms scale by nominal context length vs 6K dev. Retrieval arms extrapolate measured compact inputs.",
            "Latest-tail control uses the unified arm's evidence token budget per question; the quote uses the same mean length.",
            "No model extraction. Rule archive, BGE and splitting run locally.",
            "Official hop types determine the estimated shared planner count, not routing at inference time.",
            "EventQA uses official answer prompting before/after conservative archive. This isolates preservation, not global QA routing.",
            "No request deduplication assumed except the shared planner. Token cache rates use logged dev hit fractions and may not recur.",
            "Answer and planner output charged at configured caps. DeepSeek off-peak prices are half the peak prices.",
            "Timing uses prior long-context latency scenarios, 3–15s for EventQA, 0.6–3s for compact answers, "
            "0.5–2.5s for decomposition, 8 concurrent requests, 20% overhead plus 5–15 minutes local setup.",
        ],
        "cells": cells,
    }
    destination = ROOT / "results/test_plan_v6"
    destination.mkdir(parents=True, exist_ok=True)
    dump_json(destination / "estimate.json", result)
    write_csv(destination / "estimate.csv", cells)
    return {key: value for key, value in result.items() if key != "cells"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5, 25], required=True)
    parser.add_argument("--dev-run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(estimate(args.dev_run, args.limit), indent=2))
