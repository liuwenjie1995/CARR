"""Five-case offline check and approved twelve-case hop retrieval comparison."""

import argparse
import json
import time
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import mean

from src.evaluator import ROOT, LongContext, dump_json, utc_now
from src.factual_v2 import records_for
from src.hop_retrieval import Store, local_plan, model_messages, model_plan, retrieve
from src.run_v2 import BGE_FOLDER
from src.v2_data import read_locked
from src.v2_report import load, usage, write_csv
from src.v2_retrieval import BGE
from src.v2_transport import FLASH, Meter, credentials, quote

BASE = ROOT / "results/dev/v3-20261123T111806Z-586a10"
GOLD = ROOT / "data/dev/hop_v4_reference.json"
CAP = Decimal("0.10")
MAX_PLANNER_OUTPUT, MAX_ANSWER_OUTPUT = 256, 10
PLANNER_INPUT_LIMIT, ANSWER_INPUT_LIMIT = 1500, 1800


def quote_amount():
    return quote(FLASH, PLANNER_INPUT_LIMIT + 1024, MAX_PLANNER_OUTPUT) * 12 + quote(
        FLASH, ANSWER_INPUT_LIMIT + 1024, MAX_ANSWER_OUTPUT,
    ) * 48


def bounded(messages, limit):
    if sum(len(msg["content"].encode("utf-8")) for msg in messages) > limit:
        raise ValueError("Request exceeds the approved text byte limit.")
    return messages


def answer_messages(question, evidence):
    return [{"role": "system", "content": "Answer only from the given numbered facts. A higher number "
             "is newer for conflicting facts. Follow the evidence chain. Return only the concise answer "
             "or UNKNOWN if the evidence does not support it."},
            {"role": "user", "content": "\n".join(evidence) + "\n\nQuestion: " + question}]


def assess(trace, refs, archived):
    chosen = trace["fact_ids"]
    mismatch = next((i for i, fact in enumerate(refs) if i >= len(chosen) or chosen[i] != fact), None)
    full = mismatch is None and len(chosen) == len(refs)
    first_wrong = chosen[mismatch] if mismatch is not None and mismatch < len(chosen) else None
    return {"exact_reference_chain": full, "first_reference_failure": None if full else (mismatch + 1)
            if mismatch is not None else len(refs) + 1,
            "first_divergence_selected_archive": first_wrong is not None and first_wrong in archived,
            "selected_known_old_facts": [fact for fact in chosen if fact in archived],
            "first_divergence_fact": first_wrong}


def inputs(limit):
    rows, contexts = read_locked("factcon", "dev")
    previous = load(BASE / "manifest.json")
    question_ids = [key for key in previous["sample_ids"] if "_mh_" in key]
    by_id = {row["id"]: row for row in rows}
    questions = [by_id[key] for key in question_ids[:limit]]
    references = load(GOLD)["paths"]
    if len(question_ids) != 12 or set(question_ids) != references.keys():
        raise ValueError("Reference file does not match the exact previous development cohort.")
    history, = contexts.values()
    projection = load(BASE / "projection.json")
    if history != projection["raw_history"]:
        raise ValueError("Previous development history differs from the locked data.")
    records, duplicate = records_for(history, "numbered")
    if duplicate:
        raise ValueError("Duplicated fact numbers in dev history.")
    lines = history.splitlines()
    by_number = {rec.order: rec for rec in records.values()}
    for rid in question_ids:
        reference = references[rid]["ids"]
        evidence = [by_number[number] for number in reference]
        if not any(answer.casefold() in evidence[-1].text.casefold() for answer in by_id[rid]["answers"]):
            raise ValueError(f"Manually annotated path has no answer support: {rid}")
    removed = {item["removed_line"] for item in projection["hybrid_archive"]}
    archived = {rec.order for rec in records.values() if rec.index in removed}
    old = {}
    for path in (BASE / "samples").glob("*-conflict_pruned_rag.json"):
        row = load(path)
        if row["sample_id"] in question_ids:
            old[row["sample_id"]] = row
    if set(old) != set(question_ids):
        raise ValueError("Previous baseline scores missing.")
    return questions, references, list(records.values()), lines, archived, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5, 12], required=True)
    parser.add_argument("--run-approved", action="store_true")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.run_approved != (args.limit == 12) or (args.key_file and not args.run_approved):
        parser.error("Choose --limit 5 offline or --limit 12 --run-approved after explicit authorization.")
    questions, refs, records, lines, old_facts, old_baseline = inputs(args.limit)
    from fastembed import TextEmbedding

    embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                             local_files_only=True, specific_model_path=str(BGE_FOLDER))
    stores = {"raw": Store(records, lines, embedder), "clean": Store(
        [record for record in records if record.order not in old_facts], lines, embedder,
    )}
    baseline = LongContext(ROOT / ".venv/nltk_data")
    suffix = "paid" if args.run_approved else "offline-smoke"
    out = ROOT / "results/dev" / f"hop-v4-{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}-{suffix}"
    (out / "requests").mkdir(parents=True)
    (out / "cases").mkdir()
    manifest = {"split": "dev", "status": "running", "count": len(questions),
                "generator": "src/run_hop_dev.py", "reference_sha256": sha256(GOLD.read_bytes()).hexdigest(),
                "retriever_sha256": sha256((ROOT / "src/hop_retrieval.py").read_bytes()).hexdigest(),
                "cohort": [sample["id"] for sample in questions], "old_baseline": BASE.relative_to(ROOT).as_posix(),
                "budget_cap_usd": str(CAP), "quote_usd": str(quote_amount())}
    dump_json(out / "manifest.json", manifest)
    meter = None
    if args.run_approved:
        if not (key := credentials(FLASH, args.key_file)):
            raise SystemExit("DeepSeek credential missing.")
        if quote_amount() > CAP:
            raise SystemExit("Preflight estimate exceeds the allowed budget.")
        meter = Meter({FLASH: key}, CAP)
    requests = []

    def call(messages, phase, max_tokens, json_mode=False):
        result = meter.chat(FLASH, messages, max_tokens, phase, json_output=json_mode)
        requests.append(result)
        dump_json(out / "requests" / f"{result['id']}.json", result)
        return result

    rows = []
    try:
        entities = {fact.entity for fact in stores["raw"].facts.values()}
        for i, sample in enumerate(questions, 1):
            question = sample["question"]
            expected = refs[sample["id"]]
            lm_request = bounded(model_messages(question), PLANNER_INPUT_LIMIT)
            plans = {"local": local_plan(question, entities)}
            planner = call(lm_request, "plan", MAX_PLANNER_OUTPUT, True) if meter else None
            if planner:
                plans["model"] = model_plan(planner, question)
            case = {"sample_id": sample["id"], "question": question, "reference_answer": sample["answers"],
                    "manual_reference": expected, "previous_rag_correct": old_baseline[sample["id"]]["correct"],
                    "planner_request": planner["id"] if planner else None, "variants": []}
            seen = {}
            for mode, plan in plans.items():
                for name, store in stores.items():
                    start = time.perf_counter()
                    trace = retrieve(store, plan)
                    evidence = [store.lines[fid] for fid in trace["fact_ids"]]
                    qa = bounded(answer_messages(question, evidence), ANSWER_INPUT_LIMIT)
                    signature = json.dumps(qa, ensure_ascii=False)
                    if meter and signature not in seen:
                        seen[signature] = call(qa, "answer", MAX_ANSWER_OUTPUT)
                    answer = seen.get(signature)
                    verdict = assess(trace, expected["ids"], old_facts)
                    row = {"method": mode, "library": name, "plan": plan, "trace": trace, "evidence": evidence,
                           **verdict, "answer_request": answer["id"] if answer else None,
                           "prediction": answer["answer"] if answer else None,
                           "answer_status": (answer["error"] or "ok") if answer else None,
                           "correct": bool(not answer["error"] and baseline.score(answer["answer"], sample["answers"]))
                           if answer else None, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
                           "planner_request": planner["id"] if planner and mode == "model" else None}
                    rows.append(row)
                    case["variants"].append(row)
            dump_json(out / "cases" / f"{i:02d}.json", case)
        summary = []
        by_request = {task["id"]: task for task in requests}
        for mode in sorted({row["method"] for row in rows}):
            for name in stores:
                data = [row for row in rows if row["library"] == name and row["method"] == mode]
                ids = {row[key] for row in data for key in ("answer_request", "planner_request") if row[key]}
                stats = usage([by_request[idx] for idx in sorted(ids)])
                summary.append({"decomposition": mode, "library": name, "questions": len(data),
                                "complete_reference_chains": sum(row["exact_reference_chain"] for row in data),
                                "first_failure_hop": json.dumps(dict(Counter(
                                    row["first_reference_failure"] for row in data if row["first_reference_failure"]
                                )), sort_keys=True),
                                "known_old_selections": sum(len(row["selected_known_old_facts"]) for row in data),
                                "questions_first_derailed_by_known_old": sum(row["first_divergence_selected_archive"] for row in data),
                                "answers_correct": sum(row["correct"] is True for row in data) if meter else None,
                                "answers_with_api_error": sum(row["answer_status"] not in (None, "ok") for row in data),
                                "avg_wall_ms_including_answer_and_reuse": mean(row["elapsed_ms"] for row in data),
                                **stats})
        write_csv(out / "summary.csv", summary)
        dump_json(out / "summary.json", {
            "generator": "src/run_hop_dev.py", "actual_shared_requests": usage(requests), "arms": summary,
            "previous_rag_correct": sum(old_baseline[sample["id"]]["correct"] for sample in questions),
            "manual_reference_caveats": {sample["id"]: refs[sample["id"]]["note"] for sample in questions
                                          if "note" in refs[sample["id"]]},
        })
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest.update({"api_calls": len(requests), "spent_peak_usd": str(meter.spent if meter else 0)})
        dump_json(out / "manifest.json", manifest)
    print(out)


if __name__ == "__main__":
    main()
