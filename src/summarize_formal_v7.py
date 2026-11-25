"""Aggregate the completed formal run from raw records without model calls."""

import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import mean

from src.evaluator import ROOT, dump_json
from src.formal_v7 import EVENT_ARMS, FACT_ARMS
from src.v2_report import load, write_csv


def accuracy(rows):
    return {"correct": sum(row["correct"] for row in rows), "questions": len(rows),
            "errors": sum(bool(row["error"]) for row in rows)}


def billing(rows):
    charged = [row for row in rows if not row.get("reused", False)]
    inputs = sum(row["usage"]["prompt_tokens"] for row in charged if row["usage"])
    hits = sum(row["usage"]["cache_hit_tokens"] for row in charged if row["usage"])
    outputs = sum(row["usage"]["completion_tokens"] for row in charged if row["usage"])
    return {"paid_requests": len(charged), "reused": len(rows) - len(charged), "prompt_tokens": inputs,
            "completion_tokens": outputs, "cache_hit_tokens": hits, "cache_hit_fraction": hits / inputs if inputs else None,
            "calls_missing_usage": sum(row["usage"] is None for row in charged),
            "usage_usd": str(sum(Decimal(row["usage_usd"] or "0") for row in charged)),
            "reserved_if_usage_missing_usd": str(sum(
                Decimal(row["reserved_if_usage_missing_usd"] or "0") for row in charged)),
            "mean_end_to_end_ms": mean(row.get("end_to_end_ms", row.get("elapsed_ms", 0)) for row in rows) if rows else None}


def summarize(run_dir):
    if not run_dir.is_relative_to((ROOT / "results/test").resolve()):
        raise ValueError("Expected a formal run in results/test.")
    manifest = load(run_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only a completed run can be published as final.")
    cases = [load(path) for path in (run_dir / "cases").glob("*.json")]
    by_arm = {arm: [row for row in cases if row["arm"] == arm] for arm in (*FACT_ARMS, *EVENT_ARMS)}
    expected = {arm: manifest["factcon_questions"] if arm in FACT_ARMS else manifest["eventqa_questions"] for arm in by_arm}
    for arm, rows in by_arm.items():
        if len({row["sample_id"] for row in rows}) != len(rows) or len(rows) != expected[arm]:
            raise ValueError(f"Missing or duplicate cases: {arm}.")
        reference_arm = FACT_ARMS[0] if arm in FACT_ARMS else EVENT_ARMS[0]
        if {row["sample_id"] for row in rows} != {row["sample_id"] for row in by_arm[reference_arm]}:
            raise ValueError(f"Unpaired sample IDs: {arm}.")

    fact_metrics = []
    for arm in FACT_ARMS:
        rows = by_arm[arm]
        groups = {"all": rows}
        for tag in ("sh", "mh"):
            groups[tag] = [row for row in rows if f"_{tag}_" in row["source"]]
        for size in (32, 64, 262):
            groups[f"{size}k"] = [row for row in rows if row["source"].endswith(f"_{size}k")]
            for tag in ("sh", "mh"):
                groups[f"{size}k_{tag}"] = [row for row in groups[f"{size}k"] if f"_{tag}_" in row["source"]]
        for group, subset in groups.items():
            fact_metrics.append({"arm": arm, "group": group, **accuracy(subset), **billing(subset)})

    plans = [load(path) for path in (run_dir / "requests").glob("plan--*.json")]
    costs = [{"arm": arm, **billing(rows)} for arm, rows in by_arm.items()]
    costs.append({"arm": "shared_decomposition", **billing(plans)})
    request_files = sum(1 for _ in (run_dir / "requests").glob("*.json"))
    accounted_calls = sum(row["paid_requests"] for row in costs)
    usage_cost = sum(Decimal(row["usage_usd"]) for row in costs)
    if request_files != accounted_calls or usage_cost != Decimal(manifest["usage_usd"]):
        raise ValueError("Recorded requests or USD totals do not reconcile.")

    before, after = ({row["sample_id"]: row for row in by_arm[arm]} for arm in EVENT_ARMS)
    flips = Counter((before[sid]["correct"], after[sid]["correct"]) for sid in before
                    if not before[sid]["error"] and not after[sid]["error"])
    event_histories = [load(path) for path in (run_dir / "histories").glob("eventqa--*.json")]
    removed, lines = sum(len(row["removed"]) for row in event_histories), sum(row["total_lines"] for row in event_histories)
    event_metrics = {"arms": {arm: accuracy(by_arm[arm]) for arm in EVENT_ARMS},
                     "histories": len(event_histories), "identical_contexts": sum(row["identical_context"] for row in event_histories),
                     "reused_answers": sum(row["reused"] for row in after.values()),
                     "total_history_lines": lines, "removed_history_lines": removed,
                     "removed_line_fraction": removed / lines if lines else None,
                     "paired_without_errors": sum(flips.values()), "correct_to_wrong": flips[True, False],
                     "wrong_to_correct": flips[False, True], "evidence_removed": 0 if removed == 0 else None,
                     "per_arm_mean_recall": {arm: mean(
                         row["metrics"]["eventqa_recall"] for row in rows if not row["error"])
                         for arm, rows in ((arm, by_arm[arm]) for arm in EVENT_ARMS)}}
    summary = {"generator": "src/summarize_formal_v7.py", "factcon": fact_metrics, "billing": costs,
               "eventqa": event_metrics, "paid_requests": accounted_calls, "usage_usd": str(usage_cost),
               "factcon_error_count": sum(bool(row["error"]) for row in cases if row["dataset"] == "factcon"),
               "plan_error_count": sum(bool(row["error"]) for row in plans)}
    write_csv(run_dir / "factcon_summary.csv", fact_metrics)
    write_csv(run_dir / "billing.csv", costs)
    dump_json(run_dir / "summary.json", summary)
    print(run_dir / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, choices=[5])
    args = parser.parse_args()
    if args.limit == 5:
        dev = args.run_dir.resolve()
        if not dev.is_relative_to((ROOT / "results/dev").resolve()):
            parser.error("The offline smoke requires development logs.")
        filenames = sorted((dev / "requests").glob("answer-*.json"))[:5]
        rows = [load(path) for path in filenames]
        if len(rows) != 5 or any(row["usage"] is None for row in rows):
            raise ValueError("Expected five development requests with logged usage.")
        for row in rows:
            row["usage_usd"] = row["budget_accounted_usd"]
            row["reserved_if_usage_missing_usd"] = None
        print(billing(rows))
    else:
        summarize(args.run_dir.resolve())
