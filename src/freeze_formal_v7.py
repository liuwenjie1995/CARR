"""Validate A on saved development traces and write an offline formal freeze and quote."""

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from src.evaluator import LongContext, MAX_TOKENS, ROOT, SEED, dump_json
from src.formal_v7 import CONCURRENCY, EVENT_ARMS, FACT_ARMS, WARMUP, FactHistory, archive, event_prompts, history_batches, prefix_id
from src.run_hop_dev import MAX_PLANNER_OUTPUT
from src.run_unified_prompt_dev import BASE, V6
from src.run_v2 import BGE_FOLDER, SOURCE_PATHS
from src.run_v3_dev import token_proxy
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE, PASSAGE_TOKENS, TOP_K
from src.v2_transport import FLASH, TIMEOUT, quote

D = Decimal
DEST = ROOT / "results/test_plan_v7"
DEV_A = ROOT / "results/dev/unified-prompt-v7-20261123T133145804847Z-paid"
V2 = ROOT / "results/dev/v2-20261123T104058Z-5c9630"
V4 = ROOT / "results/dev/hop-v4-20261123T113918722799Z-paid"
LOCKS = ("data/split-lock.json", "data/eventqa_v2/split-lock.json")
PRICING = "https://api-docs.deepseek.com/quick_start/pricing/"
CACHE_DOC = "https://api-docs.deepseek.com/guides/kv_cache"
EVENT_MAX_TOKENS = 40


def relative(path):
    return path.relative_to(ROOT).as_posix()


def hashes():
    files = {*SOURCE_PATHS, relative(Path(__file__).resolve()), relative(DEST / "protocol.md"),
             "src/formal_v7.py", "src/run_unified_dev.py", "src/run_unified_prompt_dev.py", "src/run_hop_dev.py",
             "src/run_hop_recency_dev.py", "src/hop_retrieval.py", "src/factual_v3.py", "src/run_v3_dev.py",
             "src/prepare_data.py", relative(DEV_A / "summary.json")}
    files.update(relative(path) for path in BGE_FOLDER.rglob("*") if path.is_file())
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(files)}


def smoke(limit):
    from fastembed import TextEmbedding

    baseline = LongContext(ROOT / ".venv/nltk_data")
    embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                             local_files_only=True, specific_model_path=str(BGE_FOLDER))
    manifest = load(V6 / "manifest.json")
    rows, contexts = read_locked("factcon", "dev")
    by_id = {row["id"]: row for row in rows}
    subjects = [by_id[sid] for sid in manifest["cohort"][:limit]]
    history = FactHistory(next(iter(contexts.values())), embedder)
    archived = load(BASE / "projection.json")["rule_archive"]
    if history.archive != archived:
        # JSON round-trips tuples to lists. Compare operative archive decisions directly.
        if {d["removed_line"]: d["winner_line"] for d in history.archive} != {
                d["removed_line"]: d["winner_line"] for d in archived}:
            raise ValueError("Archive differs from the development run.")
    prefix_ids = set()
    for i, row in enumerate(subjects, 1):
        old = load(V6 / "cases" / f"{i:02d}.json")
        a = load(DEV_A / "cases" / f"{i:02d}.json")["variants"]["official_evidence"]
        decision = history.route(row["question"])
        if decision != old["routing"]:
            raise ValueError(f"Route replay differs for {row['id']}.")
        plan = old["saved_plan"]["plan"] if old["saved_plan"] else None
        prompts, trace = history.prompts(row, baseline, decision, plan)
        if prompts["unified_a"] != a["messages"] or trace["evidence"] != a["evidence"]:
            raise ValueError(f"A prompt replay differs for {row['id']}.")
        if token_proxy(prompts["length_matched_recent_tail"]) > token_proxy(prompts["unified_a"]):
            raise ValueError(f"Recent tail is over budget for {row['id']}.")
        prefix_ids.add(prefix_id(prompts["official_full_context"], row["question"]))
    if len(prefix_ids) != 1:
        raise ValueError("Questions from one history did not share a stable full-context prefix.")
    rows, contexts = read_locked("eventqa", "dev")
    first, = contexts.values()
    _, clean, decisions = archive(first, "dated")
    exact = sum(event_prompts(r, baseline, first, clean)[1] for r in rows[:limit])
    batches = list(history_batches(subjects))
    if len(batches) != 1 or len(batches[0][1]) != WARMUP:
        raise ValueError("History scheduling smoke failed.")
    return {"kind": "offline_development_replay", "api_calls": 0, "factcon_a_exact_matches": len(subjects),
            "eventqa_identical_request_pairs": exact, "eventqa_cases": limit,
            "eventqa_removed_lines": len(decisions), "common_full_context_prefixes": len(prefix_ids),
            "source_sha256": hashes()}


def build_quote():
    fc, event = [load(ROOT / lock)["partitions"]["test"] for lock in LOCKS]
    n, m = fc["samples"], event["samples"]
    counts = Counter()
    for name, size in fc["by_source"].items():
        counts[int(name.rsplit("_", 1)[-1][:-1])] += size
    planned_hops = sum(size for name, size in fc["by_source"].items() if "_mh_" in name)
    v3_path = BASE / "summary.json"
    v3 = {row["arm"]: row for row in load(v3_path)["arms"]}
    a = next(row for row in load(DEV_A / "summary.json")["arms"]
             if row["arm"] == "official_evidence" and row["group"] == "all")
    event_dev = next(row for row in load(V2 / "summary.json")["arms"]
                     if row["dataset"] == "eventqa" and row["arm"] == "full_recency")
    plan_calls = [load(path) for path in (V4 / "requests").glob("plan-*.json")]
    plan_mean = D(sum(row["usage"]["prompt_tokens"] for row in plan_calls)) / len(plan_calls)
    compact_prompt = D(a["prompt_tokens"]) / a["questions"] * n
    cells = []

    def add(arm, requests, prompts, completion, hits, source):
        prompts = round(prompts)
        cached = round(D(prompts) * hits)
        cells.append({"arm": arm, "planned_calls": requests, "input_tokens": prompts,
                      "output_token_cap": completion, "dev_cache_fraction": str(hits),
                      "peak_no_cache_usd": str(quote(FLASH, prompts, completion)),
                      "offpeak_with_dev_cache_usd": str(quote(FLASH, prompts, completion, cached) / 2),
                      "source": source})

    def v3_cell(arm, name, weight):
        row = v3[name]
        p = row["answer_prompt_tokens"]
        add(arm, n, D(p) / row["n"] * weight, n * MAX_TOKENS,
            D(row["answer_cache_hit_tokens"]) / p, relative(v3_path))

    long_weight = sum(D(size) * questions / 6 for size, questions in counts.items())
    for arm in FACT_ARMS:
        if arm == FACT_ARMS[0]:
            v3_cell(arm, "full", long_weight)
        elif arm == FACT_ARMS[2]:
            v3_cell(arm, "rag", n)
        elif arm == FACT_ARMS[3]:
            # The rule view and the hybrid view removed the same dev lines.
            v3_cell(arm, "conflict_pruned_rag", n)
        else:
            add(arm, n, compact_prompt, n * MAX_TOKENS, D(0), relative(DEV_A / "summary.json"))

    add("shared_decomposition", planned_hops, plan_mean * planned_hops, planned_hops * MAX_PLANNER_OUTPUT,
        D(0), relative(V4 / "requests"))
    for arm in EVENT_ARMS:
        p = event_dev["answer_prompt_tokens"]
        add(arm, m, D(p) / event_dev["questions"] * m, m * EVENT_MAX_TOKENS,
            D(event_dev["answer_cache_hit_tokens"]) / p, relative(V2 / "summary.json"))

    fields = ("planned_calls", "input_tokens", "output_token_cap", "peak_no_cache_usd", "offpeak_with_dev_cache_usd")
    old_timing = load(ROOT / "results/test_plan/estimate.json")["timing_scenarios_seconds_per_request_are_assumptions"][FLASH]
    total_long_seconds = []
    for i in (0, 1):
        event_latency = (3, 15)[i]
        long_baseline = sum(questions * old_timing[f"{size}k"][i] for size, questions in counts.items())
        warm = WARMUP * sum(old_timing[f"{size}k"][i] for size in counts)
        total_long_seconds.append((long_baseline, warm, event_latency))

    def total(rows, event_arms):
        totals = {key: (sum(row[key] for row in rows) if key in fields[:3]
                        else str(sum(D(row[key]) for row in rows))) for key in fields}
        totals["offpeak_no_cache_usd"] = str(D(totals["peak_no_cache_usd"]) / 2)
        minutes = []
        for i, (full, warm, event_latency) in enumerate(total_long_seconds):
            serial = full + 5 * n * (0.6, 3)[i] + planned_hops * (0.5, 2.5)[i] + event_arms * m * event_latency
            barriers = warm + event_arms * event["unique_contexts"] * WARMUP * event_latency
            minutes.append(round(((serial - barriers) / CONCURRENCY + barriers) * 1.2 / 60 + (5, 15)[i]))
        return {**totals, "wall_minutes_estimate": minutes}

    return {"kind": "offline_estimate_not_a_test_result", "api_calls": 0, "factcon_questions": n,
            "factcon_questions_by_k": counts, "eventqa_questions": m, "shared_plans_modeled": planned_hops,
            "extra_peak_usd_if_every_factcon_question_routes_to_planner": str(
                quote(FLASH, round(plan_mean * (n - planned_hops)), (n - planned_hops) * MAX_PLANNER_OUTPUT)),
            "with_two_eventqa_requests_per_question": total(cells, 2),
            "if_all_eventqa_requests_are_identical": total(cells[:-1], 1), "pricing": PRICING, "cells": cells,
            "assumptions": {
                "long_inputs": "Scale 6K development usage by locked history-size metadata.",
                "compact_inputs": "Use A-group measured lengths for matched truncation and both relay variants. Exact BGE fallback frequency can differ on test.",
                "planner": "Locked multi-hop label totals are a quote proxy only. Live routing uses question cues; one plan per routed question across both relay arms.",
                "rag": "Group 4 uses v3 pruned-RAG answer usage as a proxy; no extraction fees. Local rule and hybrid archive agreed on this development history.",
                "eventqa": "Use historical official-prompt usage. Actual paid calls depend on exact request equality.",
                "billing": "Output priced at cap, without retries. Cache-hit rates are measured on dev and not guaranteed on test.",
                "timing": "Eight concurrent after two sequential real questions per long history, prior long-context latency bounds, 20% overhead, 5–15 minutes local setup.",
            }}


def freeze():
    current = hashes()
    smoke_report = load(DEST / "offline_smoke.json")
    if smoke_report["limit"] != 5 or smoke_report["source_sha256"] != current:
        raise ValueError("Run the five-item offline smoke on this exact source first.")
    quote_data = build_quote()
    freeze_data = {
        "status": "frozen_pending_explicit_test_approval", "test_api_calls": 0,
        "runner_core": "src/formal_v7.py",
        "protocol": "results/test_plan_v7/protocol.md", "offline_smoke": "results/test_plan_v7/offline_smoke.json",
        "development_a_summary": relative(DEV_A / "summary.json"), "source_sha256": current,
        "test_split_locks": {name: load(ROOT / name)["files"] for name in LOCKS},
        "factcon_arms": FACT_ARMS, "eventqa_arms": EVENT_ARMS,
        "model": FLASH, "temperature": 0, "thinking": "disabled", "tools": [], "local_seed": SEED,
        "request_timeout_seconds": TIMEOUT, "retries": 0,
        "answer_template": "pinned official MemoryAgentBench LongContext.messages",
        "answer_max_tokens": {"factcon": MAX_TOKENS, "eventqa": EVENT_MAX_TOKENS},
        "planner": {"prompt": "src/hop_retrieval.py:model_messages", "max_tokens": MAX_PLANNER_OUTPUT,
                    "shared_across_relay_arms": True, "format": "json_object"},
        "retrieval": {"model": BGE, "passage_tokens_proxy": PASSAGE_TOKENS, "top_k": TOP_K,
                      "max_hops": 4, "archive_lookup_supplement": False, "llm_fact_extraction": False},
        "cache": {"group_by": "context_id", "history_and_question_order": "stable_id",
                  "sequential_first_real_questions": WARMUP, "subsequent_concurrency": CONCURRENCY,
                  "eventqa_exact_request_reuse": True, "docs": CACHE_DOC},
        "quote": "results/test_plan_v7/estimate.json", "pricing": PRICING,
    }
    dump_json(DEST / "freeze.json", freeze_data)
    dump_json(DEST / "estimate.json", quote_data)
    write_csv(DEST / "estimate.csv", quote_data["cells"])
    print(quote_data["with_two_eventqa_requests_per_question"])
    print(quote_data["if_all_eventqa_requests_are_identical"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5, 25], required=True,
                        help="5 replays development smoke; 25 freezes and quotes from metadata only.")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.limit == 5:
        report = smoke(args.limit)
        dump_json(DEST / "offline_smoke.json", {**report, "limit": args.limit})
        print({key: value for key, value in report.items() if key != "source_sha256"})
    else:
        freeze()
