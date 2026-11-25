"""Summarize recorded follow-up trajectories and extrapolate a separately approvable quote."""

import csv
import json
import random
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from statistics import mean

from src.evaluator import dump_json
from src.v2_data import DATA, EVENT_DIR
from src.v2_transport import FLASH, GPT, quote


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def usage(tasks):
    reported = [task["usage"] for task in tasks if task["usage"] is not None]
    return {
        "requests": len(tasks), "missing_usage": len(tasks) - len(reported),
        **{field: sum(row[field] for row in reported) for field in (
            "prompt_tokens", "completion_tokens", "cache_hit_tokens",
        )},
        "estimated_usd": str(sum((Decimal(task["budget_accounted_usd"]) for task in tasks), Decimal(0))),
    }


def evidence_flags(by_case, projections):
    cases = []
    for (dataset, model, sample_id), variants in sorted(by_case.items()):
        base, method = (variants.get(arm) for arm in ("full_recency", "hybrid"))
        if not base or not method:
            continue
        outcome = ("gain" if method["correct"] else "loss") if base["correct"] != method["correct"] else (
            "both_correct" if method["correct"] else "both_wrong"
        )
        if outcome == "both_correct":
            continue
        item = projections[method["projection_log"]]
        moved = item["modes"]["hybrid"]["archive"]
        history = item["raw_history"]
        spans = []
        references = [value.casefold() for value in method["reference_answers"]]
        for move in moved:
            s, e = move["old_span"]
            if any(value in history[s:e].casefold() for value in references):
                start, end = move["new_span"]
                spans.append({**move, "removed_quote": history[s:e], "kept_quote": history[start:end]})
        cases.append({
            "dataset": dataset, "model": model, "sample_id": sample_id, "outcome": outcome,
            "question": method["question"], "reference_answers": method["reference_answers"],
            "baseline": {field: base[field] for field in ("prediction", "status", "physical_request")},
            "hybrid": {field: method[field] for field in ("prediction", "status", "physical_request")},
            "projection_log": method["projection_log"], "removed_alias_candidates": spans,
        })
    return cases


def paired_rows(by_case):
    histories = defaultdict(list)
    for (dataset, model, _), arms in by_case.items():
        if "full_recency" not in arms:
            continue
        base = arms["full_recency"]
        for name, row in arms.items():
            if name != "full_recency":
                histories[dataset, model, name, row["source"], row["context_id"]].append(
                    (int(base["correct"]), int(row["correct"])),
                )
    rows, groups = [], defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (dataset, model, arm, source, context), pairs in sorted(histories.items()):
        n = len(pairs)
        b, v = (sum(x) for x in zip(*pairs, strict=True))
        rows.append({"dataset": dataset, "model": model, "arm": arm, "source": source, "context_id": context,
                     "questions": n, "baseline_correct": b, "variant_correct": v, "gain_pp": 100 * (v - b) / n,
                     "variant_only": sum(x < y for x, y in pairs), "baseline_only": sum(x > y for x, y in pairs)})
        group = groups[dataset, model, arm][context]
        group[0] += v - b
        group[1] += n
    intervals = []
    for key, histories in sorted(groups.items()):
        values = list(histories.values())
        lo = hi = None
        if len(values) > 1:
            rng = random.Random(42)
            resamples = [rng.choices(values, k=len(values)) for _ in range(10000)]
            scores = sorted(100 * sum(a for a, _ in sample) / sum(b for _, b in sample) for sample in resamples)
            lo, hi = scores[249], scores[9749]
        intervals.append(dict(zip(("dataset", "model", "arm"), key, strict=True)) | {
            "independent_histories": len(values), "gain_pp": 100 * sum(x[0] for x in values) / sum(x[1] for x in values),
            "descriptive_ci95_low_pp": lo, "descriptive_ci95_high_pp": hi,
        })
    return rows, intervals


def cost_projection(manifest, arms, totals):
    fact = load(DATA / "split-lock.json")["partitions"]["test"]
    events = load(EVENT_DIR / "split-lock.json")["partitions"]["test"]
    facts_by_length = defaultdict(int)
    for source, count in fact["by_source"].items():
        facts_by_length[int(source.rsplit("_", 1)[-1][:-1])] += count
    fact_scale = sum(length / 6 * n for length, n in facts_by_length.items())
    forecast = []
    item_count = 0
    gpt_ok = manifest["cross_model_enabled"] and any(row["model"] == GPT for row in arms) and not any(
        row["errors"] or row["answer_missing_usage"] for row in arms if row["model"] == GPT
    )
    for row in arms:
        dataset, model, arm = (row[field] for field in ("dataset", "model", "arm"))
        if model == GPT and not gpt_ok:
            continue
        if dataset == "factcon" and arm == "hybrid_no_archive":
            continue  # Requests are structurally identical to the full recency arm.
        n = 20 if model == GPT else (fact if dataset == "factcon" else events)["samples"]
        weight = n if dataset == "eventqa" or arm == "dense_top8" else (20 * 32 / 6 if model == GPT else fact_scale)
        scale = weight / row["questions"]
        prompt = round(row["answer_prompt_tokens"] * scale)
        completion = n * (10 if dataset == "factcon" else 40)
        forecast.append({"dataset": dataset, "model": model, "arm": arm, "requests_max": n,
                         "estimated_prompt_tokens": prompt, "completion_token_cap": completion,
                         "estimated_usd_no_cache": str(quote(model, prompt, completion))})
        item_count += n
    fc_history_scale = sum(length / 6 for length in facts_by_length)
    ext = totals["extract"]
    # EventQA dev contains no date-eligible lines. Reserve for this changing on heldout histories.
    assume_extra_event = events["unique_contexts"] * 64 / 6
    ext_scale = fc_history_scale + assume_extra_event
    ext_prompt = round(ext["prompt_tokens"] * ext_scale)
    ext_completion = round(ext["completion_tokens"] * ext_scale)
    ext_requests = round(ext["requests"] * ext_scale)
    costs = sum((Decimal(row["estimated_usd_no_cache"]) for row in forecast), Decimal(0))
    estimate = costs + quote(FLASH, ext_prompt, ext_completion)
    proposed = (estimate * Decimal("1.25")).quantize(Decimal("1"), rounding=ROUND_CEILING)
    errors = sum(row["errors"] + row["answer_missing_usage"] for row in arms if row["model"] == FLASH)
    return {
        "kind": "dev_extrapolation_not_test_result", "configuration": manifest["configuration"],
        "dev_smoke_complete": manifest["status"] == "completed" and errors == 0 and ext["missing_usage"] == 0,
        "cross_model_enabled": gpt_ok, "answer_cells": forecast,
        "answer_calls_upper_before_additional_deduplication": item_count,
        "estimated_prompt_tokens": sum(row["estimated_prompt_tokens"] for row in forecast) + ext_prompt,
        "completion_tokens_answer_cap_plus_estimated_extraction": sum(row["completion_token_cap"] for row in forecast) + ext_completion,
        "estimated_extraction_calls_with_eventqa_contingency": ext_requests,
        "estimated_usd_no_cache": str(estimate), "proposed_budget_ceiling_usd": str(proposed),
        "assumptions": [
            "Longer FactCon histories scale by the published nominal context lengths relative to dev 6K.",
            "Dense Top-8 answer tokens are scaled by sample count, not the full history length.",
            "EventQA answer quote includes both arms even if actual requests deduplicate.",
            "Extraction is charged per history and includes a contingency for eligible dated EventQA records.",
            "All input is priced at uncached rates. Runtime stops at the approved budget; actual tokens can differ.",
        ],
    }


def report_run(out: Path):
    manifest = load(out / "manifest.json")
    requests = {}
    for path in sorted((out / "requests").glob("*.json")):
        task = load(path)
        requests[path.relative_to(out).as_posix()] = {
            key: task.get(key) for key in ("id", "phase", "usage", "budget_accounted_usd", "extraction_status")
        }
    projections = {p.relative_to(out).as_posix(): load(p) for p in sorted((out / "histories").rglob("projection.json"))}
    grouped, by_case = defaultdict(list), defaultdict(dict)
    for path in sorted((out / "samples").glob("*.json")):
        row = load(path)
        grouped[row["dataset"], row["model"], row["arm"]].append(row)
        by_case[row["dataset"], row["model"], row["sample_id"]][row["arm"]] = row
    by_request_id = {task["id"]: task for task in requests.values()}
    arms = []
    for (dataset, model, arm), rows in sorted(grouped.items()):
        answer = usage([requests[row["physical_request"]] for row in rows])
        paths = {row["projection_log"] for row in rows}
        mode = "hybrid" if arm == "hybrid_no_archive" else arm
        reports = [projections[path]["modes"][mode] for path in sorted(paths)
                   if mode in projections[path]["modes"]]
        ext = usage([by_request_id[key] for report in reports for key in report["calls"]])
        by_history = {path: projections[path]["local_preparation"] for path in paths}
        eligible = sum(report["eligible_lines"] for report in reports)
        coverage = sum(report["rule_extracted"] + report["model_extracted"] for report in reports)
        arms.append({
            "dataset": dataset, "model": model, "arm": arm, "questions": len(rows),
            "correct": sum(row["correct"] for row in rows), "accuracy": mean(row["correct"] for row in rows),
            "errors": sum(row["status"] != "ok" for row in rows),
            "answer_mean_ms": mean(row["milliseconds"] for row in rows),
            "preparation_ms": round(sum(report["preparation_ms"] for report in reports)
                                    + sum(item.get("dense_index_preparation_ms" if arm == "dense_top8"
                                                   else "matched_recent_preparation_ms" if arm == "matched_recent"
                                                   else "", 0) for item in by_history.values()), 3),
            "deduplicated_answers": sum(row["shared_answer"] for row in rows),
            "archived_lines": sum(len(report["archive"]) for report in reports) if arm != "hybrid_no_archive" else 0,
            "eligible_lines": eligible, "covered_lines": coverage, "coverage": coverage / eligible if eligible else None,
            **{f"answer_{key}": item for key, item in answer.items()},
            **{f"extract_{key}": item for key, item in ext.items()},
            "isolated_arm_usd": str(Decimal(answer["estimated_usd"]) + Decimal(ext["estimated_usd"])),
        })
    baseline = {(row["dataset"], row["model"]): row for row in arms if row["arm"] == "full_recency"}
    for row in arms:
        base = baseline[row["dataset"], row["model"]]
        base_tokens = base["answer_prompt_tokens"] + base["answer_completion_tokens"]
        total = sum(row[f"{phase}_{field}"] for phase in ("answer", "extract")
                    for field in ("prompt_tokens", "completion_tokens"))
        row["amortized_total_token_reduction_pct"] = (
            100 * (1 - total / base_tokens) if base_tokens and not (
                row["answer_missing_usage"] or row["extract_missing_usage"] or base["answer_missing_usage"]
            ) else None
        )
    totals = {phase: usage([task for task in requests.values() if task["phase"] == phase]) for phase in ("answer", "extract")}
    details, intervals = paired_rows(by_case)
    write_csv(out / "summary.csv", arms)
    write_csv(out / "paired_by_history.csv", details)
    write_csv(out / "paired_intervals.csv", intervals)
    cases = evidence_flags(by_case, projections)
    dump_json(out / "cases.json", {"cases": cases, "note": "Alias matches are audit candidates only, not verified false archives."})
    aggregate = {
        "actual_physical_requests": totals, "arms": arms, "retrospective_error_cases": len(cases),
        "extraction_failed_batches": sum(task["extraction_status"] not in (None, "ok") for task in requests.values()),
        "scorers": {"factcon": "official_SubEM", "eventqa": "official_all_answer_element_recall"},
        "generator": "src/v2_report.py",
    }
    dump_json(out / "summary.json", aggregate)
    if manifest["split"] == "dev":
        estimate = cost_projection(manifest, arms, totals)
        dump_json(out / "test_quote.json", estimate)
    return aggregate
