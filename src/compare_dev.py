"""Run a paired DeepSeek experiment on the frozen development split."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from statistics import mean
from uuid import uuid4

from evaluator import (
    CHUNK_SIZE, MAX_RETRIES, MAX_TOKENS, MODEL, PRICING_SOURCE, RATES, ROOT, TIMEOUT_SECONDS,
    UPSTREAM, UPSTREAM_COMMIT, LongContext, dump_json, estimated_reservation, load_dev, load_key,
    run_sample, token_cost, utc_now,
)
from src.agent_memory import MemoryConfig, prepare_factconsolidation
from src.memory_state import NUMBERED

SOURCE = "factconsolidation_sh_6k"
BUDGETS = {5: Decimal("0.10"), 20: Decimal("0.40")}
ARMS = ("baseline", "method")


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["dev"])
    parser.add_argument("--limit", required=True, type=int, choices=[5, 20])
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--approved-limit-20", action="store_true")
    parser.add_argument("--smoke-summary", type=Path)
    for feature in ("dual-layer", "conflict-unlearning", "validation-gate"):
        parser.add_argument(f"--use-{feature}", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conflict-min-distinct", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--gate-min-gains", type=int, default=1)
    parser.add_argument("--gate-max-regressions", type=int, default=0)
    args = parser.parse_args()
    if args.limit == 20 and (not args.approved_limit_20 or args.smoke_summary is None):
        parser.error("A 20-case run requires user approval and a completed smoke summary.")
    if args.gate_min_gains < 1 or args.gate_max_regressions < 0:
        parser.error("Invalid gate threshold.")
    return args


def check_upstream():
    for command, expected in [(["rev-parse", "HEAD"], UPSTREAM_COMMIT), (["status", "--porcelain", "-uno"], "")]:
        actual = subprocess.run(["git", "-C", str(UPSTREAM), *command], check=True,
                                text=True, capture_output=True).stdout.strip()
        if actual != expected:
            raise SystemExit("Pinned upstream checkout does not match its clean commit.")


def code_hashes():
    paths = [
        "src/compare_dev.py", "src/evaluator.py", "src/memory_state.py", "src/replay_gate.py",
        "src/agent_memory.py", "baselines/long_context.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def verify_approval_artifact(args, metadata):
    if args.limit == 5:
        return
    previous = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
    fields = ("code_sha256", "split_lock_sha256", "config", "seed", "model", "templates_sha256")
    if (previous.get("limit") != 5 or previous.get("complete_pairs") != 5
            or previous.get("sample_ids") != metadata["sample_ids"][:5]
            or any(previous.get(field) != metadata[field] for field in fields)):
        raise SystemExit("20-case run requires a successful smoke with matching inputs and configuration.")


def arm_metrics(rows, planned):
    complete = [row for row in rows if row["status"] == "ok"]
    usage = [row["usage"] for row in rows if row["usage"] is not None]
    prompt = sum(row["prompt_tokens"] for row in usage)
    completion = sum(row["completion_tokens"] for row in usage)
    hit = sum(row["cache_hit_tokens"] for row in usage)
    successes = sum(row["correct"] for row in complete)
    return {
        "planned": planned, "attempted": len(rows), "completed": len(complete), "successes": successes,
        "errors": len(rows) - len(complete), "success_rate": successes / len(rows) if rows else None,
        "prompt_tokens": prompt, "completion_tokens": completion, "cache_hit_tokens": hit,
        "api_calls": sum(len(row["trajectory"]) for row in rows), "calls_missing_usage": len(rows) - len(usage),
        "mean_end_to_end_ms": mean(row["end_to_end_ms"] for row in rows) if rows else None,
        "estimated_usd": {period: str(token_cost(hit, prompt - hit, completion, period)) for period in RATES},
    }


def evaluate_pairs(rows, samples):
    by_id = {(row["sample_id"], row["arm"]): row for row in rows}
    counts = dict.fromkeys(("both_correct", "baseline_only", "method_only", "both_wrong", "incomplete"), 0)
    failures = []
    for sample in samples:
        baseline, method = (by_id.get((sample["id"], arm)) for arm in ARMS)
        if baseline is None or method is None or any(row["status"] != "ok" for row in (baseline, method)):
            outcome = "incomplete"
        elif baseline["correct"] and method["correct"]:
            outcome = "both_correct"
        elif baseline["correct"]:
            outcome = "baseline_only"
        elif method["correct"]:
            outcome = "method_only"
        else:
            outcome = "both_wrong"
        counts[outcome] += 1
        if outcome != "both_correct":
            failures.append({
                "sample_id": sample["id"], "question": sample["question"], "reference_answers": sample["answers"],
                "paired_outcome": outcome,
                "arms": {arm: {key: row[key] for key in (
                    "prediction", "status", "correct", "alias_presence", "trajectory_file"
                )} for arm, row in zip(ARMS, (baseline, method), strict=True) if row is not None},
            })
    return counts, failures


def alias_presence(history, answers, baseline):
    # Post-answer diagnostic. Matching words alone do not establish support.
    normalize = baseline.scoring["normalize_answer"]
    aliases = tuple(normalize(alias) for alias in answers if normalize(alias))
    hits = []
    for line in history.splitlines():
        if match := NUMBERED.fullmatch(line):
            statement = f" {normalize(match['statement'])} "
            if any(f" {alias} " in statement for alias in aliases):
                hits.append(int(match["id"]))
    return hits


def main():
    args = parse_cli()
    check_upstream()
    samples, contexts, lock_hash = load_dev()
    eligible = sorted((sample for sample in samples if sample["source"] == SOURCE), key=lambda row: row["id"])
    random.Random(args.seed).shuffle(eligible)
    picked = eligible[:args.limit]
    if len(picked) != args.limit:
        raise SystemExit("Not enough eligible development cases.")
    context_ids = sorted({sample["context_id"] for sample in picked})
    config = MemoryConfig(
        args.use_dual_layer, args.use_conflict_unlearning, args.use_validation_gate,
        args.conflict_min_distinct, args.gate_min_gains, args.gate_max_regressions,
    )
    baseline = LongContext(ROOT / ".venv" / "nltk_data")
    metadata = {
        "split": "dev", "limit": args.limit, "source": SOURCE, "seed": args.seed, "provider_seed": None,
        "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS, "chunk_size": CHUNK_SIZE,
        "retrieval_top_k": None, "tools": [], "upstream_commit": UPSTREAM_COMMIT,
        "metric": "substring_exact_match", "split_lock_sha256": lock_hash, "config": asdict(config),
        "templates_sha256": baseline.templates_sha256, "code_sha256": code_hashes(),
        "sample_ids": [sample["id"] for sample in picked], "history_count": len(context_ids),
        "timeout_seconds": TIMEOUT_SECONDS, "max_retries": MAX_RETRIES, "max_requests": 2 * args.limit,
        "budget_cap_usd": str(BUDGETS[args.limit]), "python": sys.version.split()[0],
        "pricing_source": PRICING_SOURCE,
        "pricing_usd_per_million": {period: {key: str(rate) for key, rate in rates.items()} for period, rates in RATES.items()},
    }
    verify_approval_artifact(args, metadata)

    start = time.perf_counter()
    memory = {context_id: prepare_factconsolidation(contexts[context_id], config) for context_id in context_ids}
    preprocessing_ms = round((time.perf_counter() - start) * 1000, 3)
    sensitivity = []
    for context_id in context_ids:
        for threshold in (2, 3, 4):
            _, trace = prepare_factconsolidation(contexts[context_id], replace(
                config, conflict_unlearning=True, conflict_min_distinct=threshold,
            ))
            sensitivity.append({"context_id": context_id, "min_distinct": threshold,
                                "conflicts": len(trace["conflicts"]), "archived": len(trace["archived_ids"]),
                                "parsed_facts": trace["parsed_facts"]})
    jobs = []
    for index, sample in enumerate(picked):
        # Alternate order to reduce the effect of one arm always running first.
        for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
            context_id = sample["context_id"]
            history = contexts[context_id] if arm == "baseline" else memory[context_id][0]
            start = time.perf_counter()
            messages = baseline.messages(sample, history)
            prepare_ms = (time.perf_counter() - start) * 1000
            reserve, ceil = estimated_reservation(messages)
            jobs.append((arm, sample, history, messages, reserve, ceil, prepare_ms))
    reserved = sum(job[4] for job in jobs)
    if reserved > BUDGETS[args.limit]:
        raise SystemExit(f"Budget guard rejected reserved maximum USD {reserved}.")
    metadata.update({"estimated_reservation_peak_usd": str(reserved), "state_preprocessing_ms": preprocessing_ms})
    if args.preflight:
        print(json.dumps({**metadata, "offline_threshold_profile": sensitivity, "api_calls": 0}, indent=2))
        return

    key = load_key(args.key_file)
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = ROOT / "results" / "dev" / run_id
    folder.mkdir(parents=True)
    metadata["run_id"] = run_id
    metadata["created_at_utc"] = utc_now().isoformat()
    dump_json(folder / "state_diagnostics.json", {
        "memory_traces": {context_id: item[1] for context_id, item in memory.items()},
        "offline_threshold_profile": sensitivity,
    })
    rows = []
    for index, (arm, sample, history, messages, reserve, ceil, prepare_ms) in enumerate(jobs, 1):
        row = run_sample(baseline, sample, messages, key, reserve, ceil, time.perf_counter())
        path = f"{index:02d}-{arm}-trajectory.json"
        row.update({
            "arm": arm, "trajectory_file": path, "message_preparation_ms": round(prepare_ms, 3),
            "end_to_end_ms": round(row["end_to_end_ms"] + prepare_ms, 3),
            "alias_presence": alias_presence(history, sample["answers"], baseline),
            "state_diagnostics": "state_diagnostics.json" if arm == "method" else None,
        })
        rows.append(row)
        dump_json(folder / path, row)
        usage = row["usage"]
        if usage is not None and (usage["prompt_tokens"] > ceil or usage["completion_tokens"] > MAX_TOKENS):
            metadata["stopped_reason"] = "observed_usage_exceeded_reservation"
            break
        if row["status"] in {"http_400", "http_401", "http_402", "http_403", "http_404", "http_429"}:
            metadata["stopped_reason"] = row["status"]
            break
    metrics = {arm: arm_metrics([row for row in rows if row["arm"] == arm], args.limit) for arm in ARMS}
    outcomes, failures = evaluate_pairs(rows, picked)
    complete_pairs = args.limit - outcomes["incomplete"]
    projection = None
    if args.limit == 5 and complete_pairs == args.limit and all(
        arm["calls_missing_usage"] == 0 for arm in metrics.values()
    ):
        prompt = 4 * sum(arm["prompt_tokens"] for arm in metrics.values())
        completion = 4 * sum(arm["completion_tokens"] for arm in metrics.values())
        projection = {
            "cases": 20, "calls": 40, "prompt_tokens": prompt, "completion_tokens": completion,
            "peak_no_cache_usd": str(token_cost(0, prompt, completion, "peak")),
            "estimated_reservation_peak_usd": str(reserved * 4), "cap_usd": str(BUDGETS[20]),
        }
    summary = {**metadata, "complete_pairs": complete_pairs, "paired_outcomes": outcomes, "arms": metrics,
               "paired_gain_percentage_points": 100 * (outcomes["method_only"] - outcomes["baseline_only"]) / complete_pairs
               if complete_pairs else None, "projection_limit_20": projection}
    dump_json(folder / "failure_cases.json", {
        "notes": ["A matching answer alias does not establish that the referenced fact supports the query."],
        "cases": failures,
    })
    dump_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if complete_pairs != args.limit or "stopped_reason" in metadata:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
