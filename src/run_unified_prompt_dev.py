"""Replay unified dev retrieval under the official answer template with optional BGE context."""

import argparse
import time
from decimal import Decimal
from math import ceil
from pathlib import Path
from statistics import mean

from src.evaluator import MAX_TOKENS, ROOT, LongContext, dump_json, utc_now
from src.factual_v2 import records_for
from src.run_unified_dev import BASE
from src.run_v2 import BGE_FOLDER
from src.run_v3_dev import token_proxy
from src.v2_data import read_locked
from src.v2_report import load, usage, write_csv
from src.v2_retrieval import BGE, QUERY_PREFIX, DenseIndex, normalize_rows
from src.v2_transport import FLASH, Meter, credentials, quote

V6 = ROOT / "results/dev/unified-v6-20261123T132344809711Z-paid"
CAP = Decimal("0.05")
ARMS = ("official_evidence", "official_evidence_bge")
MAX_PASSAGES = 6


def supplement(sample, evidence, baseline, dense, token_limit):
    """Pin the original evidence, then append clean passages in BGE rank order."""
    vector = normalize_rows(list(dense.model.embed([QUERY_PREFIX + sample["question"]])))[0]
    scores = dense.vectors @ vector
    ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:MAX_PASSAGES]
    selected, context = [], list(evidence)
    for index in ranked:
        additions = list(dict.fromkeys(line for line in dense.texts[index].splitlines() if line not in context))
        if not additions:
            continue
        trial = context + additions
        if token_proxy(baseline.messages(sample, "\n".join(trial))) <= token_limit:
            context = trial
            selected.append(index)
    return context, selected


def prepare(limit):
    from fastembed import TextEmbedding

    manifest = load(V6 / "manifest.json")
    samples, contexts = read_locked("factcon", "dev")
    by_id = {sample["id"]: sample for sample in samples}
    history, = contexts.values()
    projection = load(BASE / "projection.json")
    records, duplicates = records_for(history, "numbered")
    if duplicates or projection["raw_history"] != history or manifest["status"] != "complete":
        raise ValueError("Expected the locked unified development history.")
    archived = {row["removed_line"] for row in projection["rule_archive"]}
    lines = history.splitlines()
    clean_lines = [lines[index] for index in records if index not in archived]
    clean_set = set(clean_lines)
    embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                             local_files_only=True, specific_model_path=str(BGE_FOLDER))
    dense = DenseIndex(embedder, "\n".join(clean_lines))
    baseline = LongContext(ROOT / ".venv/nltk_data")
    cases = []
    for i, sid in enumerate(manifest["cohort"][:limit], 1):
        sample = by_id[sid]
        old = load(V6 / "cases" / f"{i:02d}.json")
        previous = old["variants"]["unified"]
        evidence = previous["evidence"]
        if old["sample_id"] != sid or old["question"] != sample["question"] or not set(evidence) <= clean_set:
            raise ValueError(f"Saved evidence does not match the locked development item {sid}.")
        full = load(BASE / "samples" / f"{i:02d}-full.json")
        full_call = load(BASE / "requests" / f"{full['answer_request']}.json")
        full_messages = baseline.messages(sample, history)
        if full["sample_id"] != sid or full_messages != full_call["request"]["messages"]:
            raise ValueError(f"Official prompt differs from the full-context baseline for {sid}.")
        token_limit = token_proxy(full_messages) // 4
        a_messages = baseline.messages(sample, "\n".join(evidence))
        if token_proxy(a_messages) > token_limit:
            raise ValueError(f"Original evidence already exceeds the context limit for {sid}.")
        retrieval_start = time.perf_counter()
        enriched, selected = supplement(sample, evidence, baseline, dense, token_limit)
        b_messages = baseline.messages(sample, "\n".join(enriched))
        cases.append({"sample_id": sid, "question": sample["question"], "reference": sample["answers"],
                      "type": previous["type"], "v6_trace": previous["trace"], "v6_prediction": previous["prediction"],
                      "v6_correct": previous["correct"], "v6_case": (V6 / "cases" / f"{i:02d}.json").relative_to(ROOT).as_posix(),
                      "v3_full_correct": full["correct"], "v3_full_prompt_tokens": full_call["usage"]["prompt_tokens"],
                      "full_proxy_tokens": token_proxy(full_messages),
                      "variants": {
                          ARMS[0]: {"evidence": evidence, "bge_passages": [], "messages": a_messages, "extra_retrieval_ms": 0},
                          ARMS[1]: {"evidence": enriched, "bge_passages": selected, "messages": b_messages,
                                    "extra_retrieval_ms": round((time.perf_counter() - retrieval_start) * 1000, 3)},
                      }})
    return baseline, cases


def quote_cases(cases):
    rows = []
    for arm in ARMS:
        messages = [case["variants"][arm]["messages"] for case in cases]
        # Allow tokenizer/framing differences. The meter guards actual requests separately.
        prompt = sum(ceil(token_proxy(item) * 1.15) + 32 for item in messages)
        output = len(cases) * MAX_TOKENS
        rows.append({"arm": arm, "requests": len(cases), "estimated_prompt_tokens": prompt,
                     "completion_cap": output, "peak_no_cache_estimate_usd": str(quote(FLASH, prompt, output))})
    total = sum(Decimal(row["peak_no_cache_estimate_usd"]) for row in rows)
    if total > CAP:
        raise ValueError("Estimated cost exceeds the authorized cap.")
    return {"arms": rows, "peak_no_cache_estimate_usd": str(total), "hard_cap_usd": str(CAP)}


def run(limit, execute, key_file):
    baseline, cases = prepare(limit)
    price = quote_cases(cases)
    mode = "paid" if execute else "offline"
    out = ROOT / "results/dev" / f"unified-prompt-v7-{utc_now():%Y%m%dT%H%M%S%fZ}-{mode}"
    (out / "cases").mkdir(parents=True)
    (out / "requests").mkdir()
    dump_json(out / "quote.json", price)
    meter = None
    if execute:
        key = credentials(FLASH, key_file)
        if not key:
            raise SystemExit("Missing DeepSeek key.")
        meter = Meter({FLASH: key}, CAP)
    manifest = {"split": "dev", "status": "running", "generator": "src/run_unified_prompt_dev.py",
                "cohort": [case["sample_id"] for case in cases], "model": FLASH,
                "templates_sha256": baseline.templates_sha256, "max_bge_passages": MAX_PASSAGES,
                "max_full_prompt_proxy_fraction": 0.25, "new_planner_calls": 0,
                "historical_evidence_source": V6.relative_to(ROOT).as_posix()}
    calls = {}
    dump_json(out / "manifest.json", manifest)
    try:
        for i, case in enumerate(cases, 1):
            answers = {}
            order = ARMS if i % 2 else ARMS[::-1]
            for arm in order:
                row = case["variants"][arm]
                messages = row["messages"]
                row["prompt_tokens_proxy"] = token_proxy(messages)
                if not meter:
                    continue
                signature = tuple((item["role"], item["content"]) for item in messages)
                if signature not in answers:
                    call = meter.chat(FLASH, messages, MAX_TOKENS, "answer")
                    answers[signature] = calls[call["id"]] = call
                    dump_json(out / "requests" / f"{call['id']}.json", call)
                call = answers[signature]
                row.update({"answer_request": call["id"], "prediction": call["answer"], "error": call["error"],
                            "answer_ms": call["elapsed_ms"],
                            "correct": bool(not call["error"] and baseline.score(call["answer"], case["reference"]))})
            dump_json(out / "cases" / f"{i:02d}.json", case)
        if meter:
            summary = []
            for arm in ARMS:
                for group in ("all", "single", "multi"):
                    subset = [case for case in cases if group == "all" or case["type"] == group]
                    results = [case["variants"][arm] for case in subset]
                    tally = usage([calls[row["answer_request"]] for row in results])
                    summary.append({"arm": arm, "group": group, "questions": len(subset),
                                    "correct": sum(row["correct"] for row in results),
                                    "unknown": sum((row["prediction"] or "").strip().upper() == "UNKNOWN" for row in results),
                                    "api_errors": sum(bool(row["error"]) for row in results),
                                    "mean_answer_plus_extra_retrieval_ms": mean(
                                        row["answer_ms"] + row["extra_retrieval_ms"] for row in results),
                                    "v3_full_prompt_tokens": sum(case["v3_full_prompt_tokens"] for case in subset),
                                    **tally})
            write_csv(out / "summary.csv", summary)
            dump_json(out / "summary.json", {"arms": summary, "physical": usage(calls.values()),
                                             "generator": manifest["generator"]})
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        dump_json(out / "manifest.json", {**manifest, "budget_cap_usd": str(CAP), "api_calls": len(calls)})
    print(out)
    print(price)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(5, 25), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if (args.execute and args.limit != 25) or (args.key_file and not args.execute):
        parser.error("Use --limit 5 for offline smoke, or --limit 25 for a quote or an authorized run.")
    run(args.limit, args.execute, args.key_file)
