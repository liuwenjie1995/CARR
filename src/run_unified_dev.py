"""Unified one-hop/multi-hop dev run, with a clean BGE-only multi-hop control."""

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import mean

from src.evaluator import ROOT, LongContext, dump_json, utc_now
from src.factual_v2 import Observation, norm, records_for
from src.hop_retrieval import CUES, anchor_from, retrieve
from src.run_hop_dev import ANSWER_INPUT_LIMIT, MAX_ANSWER_OUTPUT, answer_messages, assess, bounded
from src.run_hop_recency_dev import RecencyStore, saved_plans
from src.run_v2 import BGE_FOLDER
from src.v2_data import read_locked
from src.v2_report import load, usage, write_csv
from src.v2_retrieval import BGE
from src.v2_transport import FLASH, Meter, credentials, quote

BASE = ROOT / "results/dev/v3-20261123T111806Z-586a10"
GOLD = ROOT / "data/dev/hop_v4_reference.json"
CAP = Decimal("0.10")
DIRECT = {**CUES,
          "developer": r"\bdeveloper\b|\bdeveloped by\b",
          "official_language": r"\bofficial language\b",
          "music_type": r"\b(?:type|kind) of music\b|\bmusic genre\b",
          "origin_country": CUES["origin_country"] + r"|\bcreated in\b"}
PINNED_FACT_PATTERNS = (
    (r"(.+?) was developed by (.+)\.", "developer"),
    (r"The type of music that (.+?) plays is (.+)\.", "music_type"),
)


def route(question, known_entities):
    cues_text = re.sub(r'"[^"]*"|(?<!\w)\'[^\']*\'(?!\w)', "", question)
    targets = [rel for rel, regex in DIRECT.items() if re.search(regex, cues_text, re.I)]
    start = anchor_from(question, known_entities)
    if len(targets) >= 2:
        return {"route": "multi", "cue_relations": targets}
    if start is not None and len(targets) == 1:
        return {"route": "single", "start": start, "relations": targets, "cue_relations": targets}
    return {"route": "direct_bge", "cue_relations": targets}


def augment_single_index(store):
    # These two fact formats extend the single-hop index only. The v5 multi-hop
    # parser, plans, bridges, and archive stay exactly as before.
    for number, record in store.records.items():
        if number in store.facts:
            continue
        for regex, relation in PINNED_FACT_PATTERNS:
            match = re.fullmatch(regex, record.text)
            if match:
                entity, value = match.groups()
                fact = Observation(record.index, record.order, entity, relation, value, "assert",
                                   (record.start, record.start + len(record.text)), "local")
                store.index[(norm(entity), relation)].append(fact)
                store.facts[number] = fact
                break


def compact_dense_evidence(store, question):
    """Use the retrieved passages directly if single-hop grounding has no fact."""
    visible, passage_ids = store.dense.search(question)
    rows = [line for line in visible.splitlines() if re.match(r"^\d+\.", line)]
    entity = anchor_from(question, [fact.entity for fact in store.facts.values()])
    if entity:
        needle = r"(?<!\w)" + re.escape(norm(entity)) + r"(?!\w)"
        rows.sort(key=lambda line: not bool(re.search(needle, norm(line))))
    evidence = []
    for line in rows:
        trial = answer_messages(question, evidence + [line])
        if sum(len(msg["content"].encode("utf-8")) for msg in trial) <= ANSWER_INPUT_LIMIT:
            evidence.append(line)
        if len(evidence) == 8:
            break
    return evidence, {"status": "single_bge_text", "passages": passage_ids,
                      "fact_ids": [int(line.split(".", 1)[0]) for line in evidence], "events": []}


def unified_evidence(store, question, decision, known_plan):
    if decision["route"] == "multi":
        if known_plan is None:
            raise ValueError("Multi-hop dev replay has no saved plan for the routed question.")
        trace = retrieve(store, known_plan["plan"])
        return [store.lines[num] for num in trace["fact_ids"]], trace
    if decision["route"] == "single":
        fact, step = store.step(decision["start"], decision["relations"][0])
        if fact:
            return [store.lines[fact.order]], {"status": "single_direct_fact", "fact_ids": [fact.order],
                                              "events": [{**step, "chosen": fact.order}]}
    return compact_dense_evidence(store, question)


def run(limit, approved, key_file):
    manifest_v3 = load(BASE / "manifest.json")
    baseline_rows = {r["sample_id"]: r for p in (BASE / "samples").glob("*-full.json") if (r := load(p))}
    expected = manifest_v3["sample_ids"][:limit]
    rows, contexts = read_locked("factcon", "dev")
    samples = {row["id"]: row for row in rows}
    questions = [samples[key] for key in expected]
    history, = contexts.values()
    prior = load(BASE / "projection.json")
    if history != prior["raw_history"]:
        raise ValueError("Archived history does not match locked development input.")
    archived_positions = {item["removed_line"] for item in prior["rule_archive"]}
    if archived_positions != {item["removed_line"] for item in prior["hybrid_archive"]}:
        raise ValueError("Rule archive differs from frozen v5 archive.")
    records, duplicate = records_for(history, "numbered")
    if duplicate:
        raise ValueError("Duplicate fact numbers.")
    selected = [rec for rec in records.values() if rec.index not in archived_positions]
    archived = {rec.order for rec in records.values() if rec.index in archived_positions}
    from fastembed import TextEmbedding

    embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2, local_files_only=True,
                             specific_model_path=str(BGE_FOLDER))
    main_index = RecencyStore(selected, history.splitlines(), embedder, use_exact=True)
    bge_index = RecencyStore(selected, history.splitlines(), embedder, use_exact=False)
    bge_index.dense = main_index.dense
    augment_single_index(main_index)
    entities = [fact.entity for fact in main_index.facts.values()]
    plans = saved_plans()
    refs = load(GOLD)["paths"]
    scorer = LongContext(ROOT / ".venv/nltk_data")
    projected = (25 + 12) * quote(FLASH, ANSWER_INPUT_LIMIT + 1024, MAX_ANSWER_OUTPUT)
    if projected > CAP:
        raise ValueError("Planned response requests exceed the approved ceiling.")
    meter = None
    if approved:
        if not (key := credentials(FLASH, key_file)):
            raise SystemExit("Missing DeepSeek key.")
        meter = Meter({FLASH: key}, CAP)
    out = ROOT / "results/dev" / f"unified-v6-{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}-{'paid' if approved else 'offline-smoke'}"
    (out / "cases").mkdir(parents=True)
    (out / "requests").mkdir()
    metadata = {"split": "dev", "status": "running", "cohort": expected, "budget_cap_usd": str(CAP),
                "max_projected_dev_usd": str(projected), "new_planner_calls": 0,
                "archive": (BASE / "projection.json").relative_to(ROOT).as_posix(),
                "answer_prompt_source": "src/run_hop_dev.py:answer_messages",
                "planner_source": "results/dev/hop-v4-20261123T113918722799Z-paid",
                "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "model": FLASH}
    dump_json(out / "manifest.json", metadata)
    traces, all_calls = [], {}
    try:
        for i, sample in enumerate(questions, 1):
            sid, question = sample["id"], sample["question"]
            decide = route(question, entities)
            if (decide["route"] == "multi") != (sid in plans):
                raise ValueError(f"Automatic routing and saved-plan coverage differ on {sid}.")
            case = {"sample_id": sid, "source": sample["source"], "question": question, "reference": sample["answers"],
                    "routing": decide, "saved_plan": plans.get(sid), "variants": {}}
            variants = [("unified", *unified_evidence(main_index, question, decide, plans.get(sid)))]
            if decide["route"] == "multi":
                control = retrieve(bge_index, plans[sid]["plan"])
                variants.append(("clean_bge_hop", [bge_index.lines[num] for num in control["fact_ids"]], control))
            seen = {}
            for arm, evidence, trace in variants:
                start = time.perf_counter()
                request = bounded(answer_messages(question, evidence), ANSWER_INPUT_LIMIT)
                signature = json.dumps(request, ensure_ascii=False)
                response = None
                if meter and signature not in seen:
                    response = meter.chat(FLASH, request, MAX_ANSWER_OUTPUT, "answer")
                    seen[signature] = all_calls[response["id"]] = response
                    dump_json(out / "requests" / f"{response['id']}.json", response)
                if meter:
                    response = seen[signature]
                row = {"arm": arm, "type": "multi" if decide["route"] == "multi" else "single",
                       "evidence": evidence, "trace": trace, "answer_request": response["id"] if response else None,
                       "prediction": response["answer"] if response else None,
                       "correct": bool(not response["error"] and scorer.score(response["answer"], sample["answers"]))
                       if response else None,
                       "error": response["error"] if response else None,
                       "answer_ms_including_reuse": round((time.perf_counter() - start) * 1000, 3)}
                if sid in refs:
                    row.update(assess(trace, refs[sid]["ids"], archived))
                traces.append(row)
                case["variants"][arm] = row
            dump_json(out / "cases" / f"{i:02d}.json", case)
        cells = []
        for arm, types in (("unified", ("all", "single", "multi")), ("clean_bge_hop", ("multi",))):
            for kind in types:
                selected_rows = [r for r in traces if r["arm"] == arm and (kind == "all" or r["type"] == kind)]
                if not selected_rows:
                    continue
                cells.append({"arm": arm, "group": kind, "questions": len(selected_rows),
                              "correct": sum(r["correct"] is True for r in selected_rows) if meter else None,
                              "api_errors": sum(bool(r["error"]) for r in selected_rows),
                              "complete_reference_chains": sum(r.get("exact_reference_chain", False) for r in selected_rows)
                              if kind == "multi" else None,
                              "first_reference_failure": json.dumps(dict(Counter(
                                  r["first_reference_failure"] for r in selected_rows if r.get("first_reference_failure")
                              ))) if kind == "multi" else None,
                              "mean_answer_ms": mean(r["answer_ms_including_reuse"] for r in selected_rows),
                              **usage([all_calls[r["answer_request"]] for r in selected_rows if r["answer_request"]])})
        write_csv(out / "summary.csv", cells)
        dump_json(out / "summary.json", {"generator": "src/run_unified_dev.py", "arms": cells,
                  "physical": usage(all_calls.values()), "routing": dict(Counter(
                      route(sample["question"], entities)["route"] for sample in questions)),
                  "v3_full_correct": sum(baseline_rows[sid]["correct"] for sid in expected),
                  "v3_full_source": BASE.relative_to(ROOT).as_posix()})
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "stopped"
        raise
    finally:
        metadata.update({"spent_peak_usd": str(meter.spent if meter else 0), "api_calls": len(all_calls)})
        dump_json(out / "manifest.json", metadata)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", choices=[5, 25], type=int, required=True)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.approved != (args.limit == 25) or (args.key_file and not args.approved):
        parser.error("Use --limit 5 offline, or --limit 25 --approved for the authorized development run.")
    run(args.limit, args.approved, args.key_file)
