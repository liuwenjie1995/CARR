"""Replay frozen v4 plans with BGE-only and exact-latest hop retrieval."""

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
from src.factual_v2 import norm
from src.hop_retrieval import Store, retrieve
from src.run_hop_dev import ANSWER_INPUT_LIMIT, MAX_ANSWER_OUTPUT, answer_messages, assess, bounded, inputs
from src.run_v2 import BGE_FOLDER
from src.v2_report import load, usage, write_csv
from src.v2_retrieval import BGE, QUERY_PREFIX, normalize_rows
from src.v2_transport import FLASH, Meter, credentials, quote

V4 = ROOT / "results/dev/hop-v4-20261123T113918722799Z-paid"
CAP = Decimal("0.05")
ARMS = ("raw_bge", "clean_latest", "raw_latest_offline")


class RecencyStore(Store):
    def __init__(self, records, lines, embedder, *, use_exact):
        super().__init__(records, lines, embedder)
        self.use_exact = use_exact

    def step(self, entity, relation):
        key = norm(entity), relation
        if key in self.multiple:
            return self.multiple[key]
        # BGE-only arms never use the global entity/relation index for candidate generation.
        exact = self.index.get(key, []) if self.use_exact else []
        passages = None
        if exact:
            channel = "exact_latest"
            candidates = sorted(exact, key=lambda fact: -fact.order)
        else:
            channel = "bge_top8"
            query = f"{entity} {relation.replace('_', ' ')}"
            visible, passages = self.dense.search(query)
            candidates = []
            for line in visible.splitlines():
                match = re.match(r"(\d+)\.", line)
                fact = self.facts.get(int(match[1])) if match else None
                # Ground the next entity in a matching record within the retrieved passages.
                if fact and fact.relation == relation and fact.key[0].removeprefix("the ") == key[0].removeprefix("the "):
                    candidates.append(fact)
            if len(candidates) > 1:
                vectors = normalize_rows(list(self.model.embed(
                    [QUERY_PREFIX + query] + [self.lines[fact.order] for fact in candidates],
                )))
                scores = vectors[1:] @ vectors[0]
                candidates = [candidates[i] for i in sorted(
                    range(len(candidates)), key=lambda i: (-float(scores[i]), -candidates[i].order),
                )]
        result = (candidates[0] if candidates else None), {
            "entity": entity, "relation": relation, "channel": channel, "passages": passages,
            "candidates": [asdict(fact) for fact in candidates],
        }
        self.multiple[key] = result
        return result


def saved_plans():
    prior = load(V4 / "manifest.json")
    results = {}
    for path in sorted((V4 / "cases").glob("*.json")):
        row = load(path)
        plans = [variant["plan"] for variant in row["variants"] if variant["method"] == "model"]
        assert len(plans) == 2 and plans[0] == plans[1]
        results[row["sample_id"]] = {
            "plan": plans[0], "planner_request": row["planner_request"],
            "old_model_clean_correct": next(variant["correct"] for variant in row["variants"]
                                            if variant["method"] == "model" and variant["library"] == "clean"),
        }
    if prior["status"] != "complete" or list(results) != prior["cohort"]:
        raise ValueError("Missing frozen v4 plans.")
    return results


def cost_ceiling():
    return 24 * quote(FLASH, ANSWER_INPUT_LIMIT + 1024, MAX_ANSWER_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5, 12], required=True)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.approved != (args.limit == 12) or (args.key_file and not args.approved):
        parser.error("Use --limit 5 offline; the approved run requires --limit 12 --approved.")
    plans = saved_plans()
    questions, refs, records, lines, archived, old_rag = inputs(args.limit)
    if cost_ceiling() > CAP:
        raise SystemExit("Quote exceeds the approved budget cap.")
    meter = None
    if args.approved:
        if not (key := credentials(FLASH, args.key_file)):
            raise SystemExit("DeepSeek credentials unavailable.")
        meter = Meter({FLASH: key}, CAP)
    from fastembed import TextEmbedding

    model = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2, local_files_only=True,
                          specific_model_path=str(BGE_FOLDER))
    raw = RecencyStore(records, lines, model, use_exact=False)
    cleaned = RecencyStore([record for record in records if record.order not in archived], lines, model, use_exact=True)
    raw_latest = RecencyStore(records, lines, model, use_exact=True)
    # Same raw passages in both raw arms. The third arm only produces offline retrieval traces.
    raw_latest.dense = raw.dense
    stores = dict(zip(ARMS, (raw, cleaned, raw_latest), strict=True))
    baseline = LongContext(ROOT / ".venv/nltk_data")
    tag = "paid" if args.approved else "offline-smoke"
    out = ROOT / "results/dev" / f"hop-v5-{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}-{tag}"
    (out / "cases").mkdir(parents=True)
    (out / "requests").mkdir()
    metadata = {"status": "running", "split": "dev", "sample_ids": [q["id"] for q in questions],
                "v4_plans": V4.relative_to(ROOT).as_posix(), "new_planner_calls": 0,
                "reference_sha256": sha256((ROOT / "data/dev/hop_v4_reference.json").read_bytes()).hexdigest(),
                "budget_cap_usd": str(CAP), "projected_peak_usd": str(cost_ceiling()),
                "model": FLASH, "seed": 42, "temperature": 0, "max_answer_tokens": MAX_ANSWER_OUTPUT}
    dump_json(out / "manifest.json", metadata)
    trajectories, rows = {}, []
    try:
        for i, sample in enumerate(questions, 1):
            sid = sample["id"]
            frozen = plans[sid]
            ref = refs[sid]
            case = {"sample_id": sid, "question": sample["question"], "reference_answers": sample["answers"],
                    "manual_reference": ref, **frozen, "previous_rag_correct": old_rag[sid]["correct"], "variants": {}}
            seen = {}
            for arm, store in stores.items():
                start = time.perf_counter()
                trace = retrieve(store, frozen["plan"])
                evidence = [store.lines[num] for num in trace["fact_ids"]]
                messages = bounded(answer_messages(sample["question"], evidence), ANSWER_INPUT_LIMIT)
                request = None
                if meter and arm != "raw_latest_offline":
                    signature = json.dumps(messages, ensure_ascii=False)
                    if signature not in seen:
                        response = meter.chat(FLASH, messages, MAX_ANSWER_OUTPUT, "answer")
                        request_id = response["id"]
                        dump_json(out / "requests" / f"{request_id}.json", response)
                        trajectories[request_id] = seen[signature] = response
                    request = seen[signature]
                row = {"arm": arm, "trace": trace, "evidence": evidence, **assess(trace, ref["ids"], archived),
                       "answer_request": request["id"] if request else None,
                       "prediction": request["answer"] if request else None,
                       "correct": (not request["error"] and baseline.score(request["answer"], sample["answers"]))
                       if request else None,
                       "wall_ms": round((time.perf_counter() - start) * 1000, 3)}
                rows.append(row)
                case["variants"][arm] = row
            dump_json(out / "cases" / f"{i:02d}.json", case)
        cells = []
        for arm in ARMS:
            variants = [row for row in rows if row["arm"] == arm]
            calls = [trajectories[row["answer_request"]] for row in variants if row["answer_request"]]
            cells.append({"arm": arm, "questions": len(variants),
                          "complete_reference_chains": sum(row["exact_reference_chain"] for row in variants),
                          "first_reference_failure": json.dumps(dict(Counter(
                              row["first_reference_failure"] for row in variants if row["first_reference_failure"]
                          )), sort_keys=True),
                          "known_old_selections": sum(len(row["selected_known_old_facts"]) for row in variants),
                          "first_divergences_via_archived_fact": sum(row["first_divergence_selected_archive"] for row in variants),
                          "correct": sum(row["correct"] is True for row in variants) if calls else None,
                          "api_errors": sum(bool(row["error"]) for row in calls), "mean_wall_ms": mean(row["wall_ms"] for row in variants),
                          **usage(calls)})
        write_csv(out / "summary.csv", cells)
        dump_json(out / "summary.json", {"generator": "src/run_hop_recency_dev.py", "arms": cells,
                  "physical_requests": usage(list(trajectories.values())), "new_planner_calls": 0,
                  "v4_pruned_rag_correct": sum(old_rag[q["id"]]["correct"] for q in questions),
                  "v4_hop_clean_correct": sum(plans[q["id"]]["old_model_clean_correct"] for q in questions)})
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "stopped"
        raise
    finally:
        metadata.update({"spent_peak_usd": str(meter.spent if meter else 0), "physical_requests": len(trajectories)})
        dump_json(out / "manifest.json", metadata)
    print(out)


if __name__ == "__main__":
    main()
