"""Quote or run the approved FactConsolidation-only v3 development comparison."""

import argparse
import csv
import json
import time
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from itertools import zip_longest
from math import ceil
from pathlib import Path
from statistics import mean
from unittest.mock import patch
from uuid import uuid4

from src.evaluator import ROOT, LongContext, dump_json, utc_now
from src.factual_v3 import batches, plan_extraction, prepare
from src.run_v2 import BGE_FOLDER, choose, snapshot
from src.v2_data import DATA, read_locked
from src.v2_extract import OUTPUT_CAP, messages
from src.v2_report import load, usage
from src.v2_retrieval import BGE, PASSAGE_TOKENS, TOP_K, DenseIndex, proxy_tokens
from src.v2_transport import FLASH, Meter, credentials, quote

ARMS = ("full", "rule_full", "rag", "gated_full", "conflict_pruned_rag")
GATED_ARMS = {"gated_full", "conflict_pruned_rag"}
RAG_ARMS = {"rag", "conflict_pruned_rag"}
MAX_OUTPUT = 10
CAP = Decimal("0.50")
QUOTES = ROOT / "results" / "dev_plan_v3"
CALIBRATION = ROOT / "results/dev/v2-20261123T104058Z-5c9630/samples"


def cohort(rows, limit):
    sh = choose([x for x in rows if "_sh_" in x["source"]], (limit + 1) // 2)
    mh = choose([x for x in rows if "_mh_" in x["source"]], limit // 2)
    result = [row for pair in zip_longest(sh, mh) for row in pair if row]
    if len(result) != limit:
        raise ValueError("Missing items in the fixed development cohort.")
    return result


def request_fingerprint(messages):
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def bge_indexes(embedder, original, pruned):
    raw = DenseIndex(embedder, original)
    return raw, raw if original == pruned else DenseIndex(embedder, pruned)


def inputs_for(arm, question, baseline, history, rule, hybrid, raw_index, pruned_index):
    selected = None
    if arm == "rag":
        visible, selected = raw_index.search(question["question"])
    elif arm == "conflict_pruned_rag":
        visible, selected = pruned_index.search(question["question"])
    else:
        visible = {"full": history, "rule_full": rule, "gated_full": hybrid}[arm]
    return baseline.messages(question, visible), selected


def corpus_audit():
    import pyarrow.parquet as pq

    corpus = DATA / "source" / "Conflict_Resolution.parquet"
    split = load(DATA / "split-lock.json")
    if sha256(corpus.read_bytes()).hexdigest() != split["source_sha256"]:
        raise ValueError("Official parquet differs from the locked revision.")
    groups = defaultdict(list)
    for row in pq.read_table(corpus, columns=["metadata", "context", "questions"]).to_pylist():
        digest = sha256(row["context"].encode()).hexdigest()
        groups[digest].append({"source": row["metadata"]["source"], "questions": len(row["questions"])})
    return {"source_sha256": split["source_sha256"], "source_revision": split["revision"],
            "official_distinct_histories": len(groups), "contexts_by_sha256": dict(groups),
            "local_dev_histories": split["partitions"]["dev"]["unique_contexts"],
            "local_test_histories": split["partitions"]["test"]["unique_contexts"],
            "local_test_questions": split["partitions"]["test"]["samples"],
            "excluded_from_test": split["excluded_test_samples"]}


def token_proxy(request):
    return sum(proxy_tokens(message["content"]) for message in request)


def make_quote(items, history, layout, baseline, embedder, config):
    rule, preview, trace = prepare(history, layout)
    raw, pruned = bge_indexes(embedder, history, preview)
    raw_payloads = defaultdict(list)
    for item in items:
        for arm in ARMS:
            payload, _ = inputs_for(arm, item, baseline, history, rule, preview, raw, pruned)
            raw_payloads[arm].append(token_proxy(payload))
    ratios = []
    for path in CALIBRATION.glob("*.json"):
        log = load(path)
        if log["dataset"] == "factcon" and log["model"] == FLASH and log["usage"]:
            ratios.append(log["usage"]["prompt_tokens"] / log["token_proxy_o200k_unframed"])
    if not ratios:
        raise ValueError("Missing DeepSeek development token calibration.")
    multiplier = max(1.05, max(ratios))
    quote_rows = []
    for arm in ARMS:
        tokens = ceil(sum(raw_payloads[arm]) * multiplier)
        maximum_line = max(proxy_tokens(line) for line in history.splitlines())
        if arm == "gated_full":
            upper = ceil((sum(raw_payloads["full"]) + 128 * len(items)) * multiplier)
        elif arm == "conflict_pruned_rag":
            upper = ceil(sum(max(observed, TOP_K * max(PASSAGE_TOKENS, maximum_line)
                                 + token_proxy(baseline.messages(question, "")) + 128)
                             for observed, question in zip(raw_payloads[arm], items, strict=True)) * multiplier)
        else:
            upper = tokens
        output = len(items) * MAX_OUTPUT
        quote_rows.append({"arm": arm, "max_calls": len(items), "prompt_token_estimate": tokens,
                           "planning_prompt_tokens": upper, "completion_cap": output,
                           "peak_uncached_planning_usd": str(quote(FLASH, upper, output))})
    batches_list = list(batches(layout[3]))
    ext_prompt = ceil(sum(token_proxy(messages(batch)) + 32 for batch in batches_list) * multiplier)
    ext_out = len(batches_list) * OUTPUT_CAP
    estimate = sum((quote(FLASH, row["prompt_token_estimate"], row["completion_cap"])
                    for row in quote_rows), quote(FLASH, ext_prompt, ext_out))
    planning = sum((Decimal(row["peak_uncached_planning_usd"]) for row in quote_rows),
                   quote(FLASH, ext_prompt, ext_out))
    return {
        "kind": "offline_quote_not_an_experiment", "api_calls": 0, "n": len(items),
        "sample_ids": [row["id"] for row in items], "configuration": config, "arms": quote_rows,
        "fact_coverage": trace, "context_count_audit": corpus_audit(), "token_input_multiplier": multiplier,
        "extraction": {"max_calls": len(batches_list), "prompt_token_estimate": ext_prompt,
                       "completion_token_cap": ext_out, "peak_uncached_usd": str(quote(FLASH, ext_prompt, ext_out))},
        "peak_uncached_estimated_usd": str(estimate), "peak_uncached_planning_usd": str(planning),
        "hard_cap_usd": str(CAP), "generator": "src/run_v3_dev.py",
        "token_estimate_basis": "Actual local payloads on locked dev, calibrated to v2 Flash usage. Dynamic hybrid "
        "estimated from the offline rule projection. Planning case gives hybrid full the unpruned input and gives "
        "pruned RAG a generous Top-8 allowance. Every output charged at its cap; no cache or dedup savings assumed.",
    }


def run_paid(items, history, layout, baseline, embedder, key, approved, gate):
    out = ROOT / "results/dev" / ("v3-" + utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6])
    for name in ("requests", "samples"):
        (out / name).mkdir(parents=True)
    meter = Meter({FLASH: key}, CAP)
    save = lambda row: dump_json(out / "requests" / (row["id"] + ".json"), row)
    physical, logs = {}, []
    manifest = {"status": "running", "split": "dev", "quote_sha256": sha256(approved.read_bytes()).hexdigest(),
                "budget_usd": str(CAP), "sample_ids": [row["id"] for row in items]}
    dump_json(out / "manifest.json", manifest)
    try:
        start = time.perf_counter()
        rule, hybrid, trace = prepare(history, layout, meter.chat, save, entity_gate=gate)
        preparation_ms = round((time.perf_counter() - start) * 1000, 3)
        raw, pruned = bge_indexes(embedder, history, hybrid)
        dump_json(out / "projection.json", {"raw_history": history, **trace,
                  "preparation_ms": preparation_ms, "rag_index_ms": raw.build_ms,
                  "pruned_index_ms": 0 if raw is pruned else pruned.build_ms})
        extraction = [load(out / "requests" / (key + ".json")) for key in trace["extraction_calls"]]
        for index, item in enumerate(items):
            seen = {}
            order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
            for arm in order:
                started = time.perf_counter()
                payload, passages = inputs_for(arm, item, baseline, history, rule, hybrid, raw, pruned)
                sig = request_fingerprint(payload)
                reused = sig in seen
                if not reused:
                    seen[sig] = call = meter.chat(FLASH, payload, MAX_OUTPUT, "answer")
                    save(call)
                    physical[call["id"]] = call
                call = seen[sig]
                correct = not call["error"] and baseline.score(call["answer"], item["answers"])
                row = {"sample_id": item["id"], "source": item["source"], "arm": arm, "correct": bool(correct),
                       "question": item["question"], "reference_answers": item["answers"], "prediction": call["answer"],
                       "status": call["error"] or "ok", "answer_request": call["id"], "reused_response": reused,
                       "selected_passages": passages, "usage": call["usage"],
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
                dump_json(out / "samples" / f"{index+1:02d}-{arm}.json", row)
                logs.append(row)
                if call["error"] in {"http_400", "http_401", "http_402", "http_403", "http_404", "http_429"}:
                    raise RuntimeError(call["error"])
        summary = []
        for arm in ARMS:
            chosen = [row for row in logs if row["arm"] == arm]
            answer = usage([physical[row["answer_request"]] for row in chosen])
            extract = usage(extraction if arm in GATED_ARMS else [])
            summary.append({"arm": arm, "n": len(chosen), "correct": sum(row["correct"] for row in chosen),
                            "accuracy": mean(row["correct"] for row in chosen),
                            "errors": sum(row["status"] != "ok" for row in chosen),
                            "runtime_ms_per_case": mean(row["elapsed_ms"] for row in chosen),
                            "unique_paid_answers_for_arm": sum(not row["reused_response"] for row in chosen),
                            **{f"answer_{name}": value for name, value in answer.items()},
                            **{f"extract_{name}": value for name, value in extract.items()},
                            "standalone_total_tokens": sum(phase[key] for phase in (answer, extract)
                                                           for key in ("prompt_tokens", "completion_tokens")),
                            "standalone_estimated_usd": str(Decimal(answer["estimated_usd"]) + Decimal(extract["estimated_usd"]))})
        with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        dump_json(out / "summary.json", {"arms": summary, "actual_shared_answer_requests": usage(physical.values()),
                  "actual_shared_extraction_requests": usage(extraction), "generator": "src/run_v3_dev.py"})
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest["accounted_peak_usd"] = str(meter.spent)
        manifest["physical_calls"] = dict(meter.calls)
        dump_json(out / "manifest.json", manifest)
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", choices=[5, 25], required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument("--approved-quote", type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--entity-gate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.execute and (args.limit != 25 or args.confirmation != "确认运行" or not args.approved_quote):
        parser.error("Paid development requires --limit 25, --confirmation 确认运行 and an accepted quote.")
    if not args.execute and (args.confirmation or args.approved_quote or args.key_file):
        parser.error("Offline mode accepts no confirmation, quote or credentials.")
    seed_hashes = snapshot()
    config = {"v2_dependencies": seed_hashes, "source_hashes": {
        file: sha256((ROOT / file).read_bytes()).hexdigest()
        for file in ("src/factual_v3.py", "src/run_v3_dev.py")}, "entity_gate": args.entity_gate, "model": FLASH}
    samples, contexts = read_locked("factcon", "dev")
    items = cohort(samples, args.limit)
    if len(contexts) != 1 or {item["context_id"] for item in items} != set(contexts):
        raise ValueError("This v3 development cohort must use the single locked development history.")
    history, = contexts.values()
    layout = plan_extraction(history, entity_gate=args.entity_gate)
    from fastembed import TextEmbedding

    with patch("nltk.download", return_value=True):
        baseline = LongContext(ROOT / ".venv/nltk_data")
        embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                                 local_files_only=True, specific_model_path=str(BGE_FOLDER))
        if args.execute:
            expected = load(args.approved_quote)
            if (expected.get("kind") != "offline_quote_not_an_experiment" or expected["configuration"] != config
                    or expected["sample_ids"] != [row["id"] for row in items]
                    or Decimal(expected["peak_uncached_planning_usd"]) > CAP):
                raise SystemExit("Quote does not match this exact configuration, cohort or budget.")
            if not (key := credentials(FLASH, args.key_file)):
                raise SystemExit("DeepSeek credential unavailable.")
            run_paid(items, history, layout, baseline, embedder, key, args.approved_quote, args.entity_gate)
        else:
            result = make_quote(items, history, layout, baseline, embedder, config)
            QUOTES.mkdir(parents=True, exist_ok=True)
            target = QUOTES / f"preflight-{args.limit}.json"
            dump_json(target, result)
            print(json.dumps({key: result[key] for key in (
                "n", "api_calls", "peak_uncached_estimated_usd", "peak_uncached_planning_usd", "hard_cap_usd",
            )}, indent=2))
            print(target)


if __name__ == "__main__":
    main()
