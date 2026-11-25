"""A-G controlled ablation matrix on FactConsolidation 562Q (reusing frozen v2 parses & plans).

Groups:
  A_fact_hybrid          : Fact-level BM25+BGE RRF top-8 over all parsed facts (no recency, no planner)
  B_latest_dict_hybrid   : Latest-Fact Dictionary (max order per key) + BM25+BGE RRF top-8 (no planner)
  C_planner_no_recency   : Cached Planner v2 + exact-key lookup selecting earliest order (no recency, Ch1 only)
  D_planner_exact_recency: Cached Planner v2 + exact-key lookup selecting latest order (Ch1 only)
  E_plus_semantic_rel    : D + Channel 2 entity semantic relation match (reuses D answer if prompt identical)
  F_plus_hybrid_fallback : Loaded from carr_v2_planner_only (E + Channel 3 hybrid fallback, no tombstone)
  G_full_carr_v2         : Loaded from carr_v2 (Full CARR v2 with tombstone)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.long_context import MAX_TOKENS
from src.carr_v2.open_parser import parse_numbered_v2
from src.carr_v2.semantic_router import CANONICAL_SET, _entity_variants
from src.carr_v2.tombstone_index import BM25Index
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import norm, records_for
from src.hop_retrieval import BRIDGES, MAX_FACTS
from src.run_carr_v2 import SafeMeter
from src.run_v2 import BGE_FOLDER
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE, QUERY_PREFIX, normalize_rows
from src.v2_transport import FLASH, credentials

RUN_FC_V2 = ROOT / "results" / "test" / "carr-v2-formal-20261124"
DEFAULT_OUT = ROOT / "results" / "test" / "ablation-matrix-v2"

NEW_ARMS = (
    "A_fact_hybrid",
    "B_latest_dict_hybrid",
    "C_planner_no_recency",
    "D_planner_exact_recency",
    "E_plus_semantic_rel",
)
ALL_MATRIX_ARMS = (*NEW_ARMS, "F_plus_hybrid_fallback", "G_full_carr_v2")


def _norm_ans(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _matches_ref(val: str, refs: list[str]) -> bool:
    nv = _norm_ans(val)
    if not nv:
        return False
    for r in refs:
        nr = _norm_ans(r)
        if not nr:
            continue
        if nv == nr or (len(nr) >= 3 and re.search(r"(?<!\w)" + re.escape(nr) + r"(?!\w)", nv)):
            return True
    return False


def _exact_mcnemar(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


class FactRRFIndex:
    """BM25 + BGE RRF top-k search directly at individual fact-line granularity."""

    def __init__(self, embedder, lines_with_order: list[tuple[int, str]]):
        self.items = sorted(lines_with_order, key=lambda x: x[0])
        self.texts = [txt for _, txt in self.items]
        self.orders = [ord_ for ord_, _ in self.items]
        self.bm25 = BM25Index(self.texts)
        self.embedder = embedder
        self.vecs = normalize_rows(list(embedder.embed(self.texts, batch_size=64)))

    def search(self, query: str, top_k: int = 8, rrf_k: int = 60) -> tuple[str, list[int]]:
        if not self.items:
            return "", []
        lex_scores = self.bm25.scores(query)
        q_vec = normalize_rows(list(self.embedder.embed([QUERY_PREFIX + query])))[0]
        dense_scores = self.vecs @ q_vec

        lex_order = np.argsort(-lex_scores, kind="mergesort")
        dense_order = np.argsort(-dense_scores, kind="mergesort")
        fused = np.zeros(len(self.items), dtype=np.float64)
        pool = min(len(self.items), max(top_k * 10, 64))
        for r, idx in enumerate(lex_order[:pool]):
            fused[idx] += 1.0 / (rrf_k + r + 1)
        for r, idx in enumerate(dense_order[:pool]):
            fused[idx] += 1.0 / (rrf_k + r + 1)

        chosen_idx = sorted(np.argsort(-fused, kind="mergesort")[:top_k].tolist())
        chosen_lines = [self.texts[i] for i in chosen_idx]
        chosen_orders = [self.orders[i] for i in chosen_idx]
        return "\n".join(chosen_lines), chosen_orders


class StructuredStoreLite:
    """In-memory structured ledger supporting C (no recency), D (exact recency), and E (+semantic)."""

    def __init__(self, raw_text: str, embedder):
        raw_lines = raw_text.splitlines()
        records, _ = records_for(raw_text, "numbered")
        self.by_key: dict[tuple[str, str], list] = defaultdict(list)
        self.by_ent: dict[str, list] = defaultdict(list)
        self.facts: dict[int, object] = {}
        self.raw_line_by_order: dict[int, str] = {}
        self.all_parsed_lines: list[tuple[int, str]] = []
        for rec in sorted(records.values(), key=lambda r: r.order):
            self.raw_line_by_order[rec.order] = raw_lines[rec.index]
            if fact := parse_numbered_v2(rec):
                self.by_key[fact.key].append(fact)
                self.by_ent[norm(fact.entity)].append(fact)
                self.facts[fact.order] = fact
                self.all_parsed_lines.append((fact.order, raw_lines[rec.index]))
        self.embedder = embedder

    def latest_dict_lines(self) -> list[tuple[int, str]]:
        out = []
        for _key, flist in self.by_key.items():
            best = max(flist, key=lambda f: f.order)
            out.append((best.order, self.raw_line_by_order[best.order]))
        return out

    def _lookup_exact(self, entity: str, relation: str) -> list:
        for evar in _entity_variants(entity):
            cands = self.by_key.get((evar, relation), [])
            if cands:
                return cands
        return []

    def _lookup_entity_facts(self, entity: str) -> list:
        out = []
        for evar in _entity_variants(entity):
            out.extend(self.by_ent.get(evar, []))
        return out

    def step(self, entity: str, relation: str, sub_query: str, *, recency: bool, allow_semantic: bool, embed_lock) -> tuple[object | None, dict]:
        exact = self._lookup_exact(entity, relation)
        if exact:
            cands = sorted(exact, key=lambda f: (-f.order if recency else f.order))
            chosen = cands[0]
            return chosen, {"channel": "exact_latest" if recency else "exact_earliest", "chosen": chosen.order}

        if allow_semantic and (
            relation not in CANONICAL_SET
            or relation in {"creator", "author", "manufacturer", "developer", "original_language", "official_language"}
        ):
            ent_facts = self._lookup_entity_facts(entity)
            if ent_facts:
                qtext = f"{entity} {sub_query or relation.replace('_', ' ')}"
                with embed_lock:
                    vecs = normalize_rows(list(self.embedder.embed(
                        [QUERY_PREFIX + qtext] + [self.raw_line_by_order[f.order] for f in ent_facts]
                    )))
                sims = vecs[1:] @ vecs[0]
                best_idx = max(range(len(ent_facts)), key=lambda i: (float(sims[i]), ent_facts[i].order if recency else -ent_facts[i].order))
                min_sim = 0.56 if relation not in CANONICAL_SET else 0.62
                if float(sims[best_idx]) >= min_sim:
                    best_rel = ent_facts[best_idx].relation
                    matched = sorted(
                        [f for f in ent_facts if f.relation == best_rel],
                        key=lambda f: (-f.order if recency else f.order),
                    )
                    chosen = matched[0]
                    return chosen, {"channel": "entity_semantic_recency", "chosen": chosen.order}

        return None, {"channel": "miss", "chosen": None}

    def retrieve_chain(self, plan: dict, *, recency: bool, allow_semantic: bool, embed_lock) -> tuple[str, dict]:
        entity = plan.get("start") or ""
        steps = plan.get("steps") or []
        if not entity or not steps:
            return "", {"status": "empty_plan", "fact_ids": [], "events": [], "last_value": ""}

        queue: list[tuple[str, str, bool]] = [(s["relation"], s["query"], True) for s in steps]
        events, visited, proof = [], set(), []
        status = "completed_route"
        while queue:
            rel, sub_q, allow_bridge = queue.pop(0)
            can_bridge = allow_bridge and rel in BRIDGES and len(proof) + len(queue) + len(BRIDGES[rel]) <= MAX_FACTS
            fact, ev = self.step(
                entity,
                rel,
                sub_q,
                recency=recency,
                allow_semantic=(allow_semantic and not can_bridge),
                embed_lock=embed_lock,
            )
            events.append(ev)
            if not fact and can_bridge:
                queue = [(part, part.replace("_", " "), False) for part in BRIDGES[rel]] + queue
                continue
            if not fact:
                status = "no_matching_fact"
                break
            if len(proof) >= MAX_FACTS or fact.order in visited:
                status = "path_limit_or_cycle"
                break
            visited.add(fact.order)
            proof.append(fact.order)
            entity = fact.value

        ctx = "\n".join(self.raw_line_by_order[fid] for fid in proof)
        return ctx, {"status": status, "fact_ids": proof, "events": events, "last_value": entity}


def run_ablation(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    samples, contexts = read_locked("factcon", "test")
    n = len(samples)
    if not execute:
        # A, B, C, D = 4 * 562 calls; E mostly reuses D (~30 diff calls)
        est_calls = n * 4 + 40
        est_usd = token_cost(0, est_calls * 350, est_calls * MAX_TOKENS, "peak")
        q = {"questions": n, "arms": list(ALL_MATRIX_ARMS), "est_new_calls": est_calls, "est_peak_usd": str(est_usd)}
        print(json.dumps(q, indent=2))
        return q

    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DeepSeek API key.")

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "requests", "traces"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    from threading import Lock
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
        local_files_only=True, specific_model_path=str(BGE_FOLDER),
    )
    embed_lock = Lock()

    # Load cached v2 plans from carr-v2-formal-20261124/traces
    plans: dict[str, dict] = {}
    for s in samples:
        tr = load(RUN_FC_V2 / "traces" / f"{s['id']}.json")
        plans[s["id"]] = tr.get("plan") or {}

    by_ctx = defaultdict(list)
    for s in samples:
        by_ctx[s["context_id"]].append(s)

    for cid, ctx_rows in sorted(by_ctx.items()):
        store = StructuredStoreLite(contexts[cid], embedder)
        idx_a = FactRRFIndex(embedder, store.all_parsed_lines)
        idx_b = FactRRFIndex(embedder, store.latest_dict_lines())

        def process_one(samp: dict) -> None:
            sid = samp["id"]
            q = samp["question"]
            plan = plans[sid]

            ctx_map: dict[str, tuple[str, dict]] = {}
            with embed_lock:
                ctx_a, ids_a = idx_a.search(q, top_k=8)
                ctx_b, ids_b = idx_b.search(q, top_k=8)
            ctx_map["A_fact_hybrid"] = (ctx_a, {"fact_ids": ids_a, "status": "hybrid_top8"})
            ctx_map["B_latest_dict_hybrid"] = (ctx_b, {"fact_ids": ids_b, "status": "latest_dict_top8"})

            ctx_c, tr_c = store.retrieve_chain(plan, recency=False, allow_semantic=False, embed_lock=embed_lock)
            ctx_d, tr_d = store.retrieve_chain(plan, recency=True, allow_semantic=False, embed_lock=embed_lock)
            ctx_e, tr_e = store.retrieve_chain(plan, recency=True, allow_semantic=True, embed_lock=embed_lock)
            ctx_map["C_planner_no_recency"] = (ctx_c, tr_c)
            ctx_map["D_planner_exact_recency"] = (ctx_d, tr_d)
            ctx_map["E_plus_semantic_rel"] = (ctx_e, tr_e)

            trace_out = {"sample_id": sid, "context_id": cid}
            d_case_cache: dict | None = None

            for arm in NEW_ARMS:
                cpath = out / "cases" / f"{arm}--{sid}.json"
                ctx_str, arm_tr = ctx_map[arm]
                trace_out[arm] = arm_tr

                # Full-chain and gold-alias diagnostics
                retrieved_text = "\n".join(store.raw_line_by_order[fid] for fid in arm_tr["fact_ids"] if fid in store.raw_line_by_order)
                gold_alias_hit = _matches_ref(retrieved_text, samp["answers"])
                full_chain_hit = False
                if arm in ("C_planner_no_recency", "D_planner_exact_recency", "E_plus_semantic_rel"):
                    # Strict full-chain check requires completed route + latest valid hops + final value match
                    all_latest = True
                    for fid in arm_tr["fact_ids"]:
                        fobj = store.facts.get(fid)
                        if not fobj or fid != max(x.order for x in store.by_key[fobj.key]):
                            all_latest = False
                            break
                    full_chain_hit = (
                        arm_tr["status"] == "completed_route"
                        and bool(arm_tr["fact_ids"])
                        and all_latest
                        and _matches_ref(arm_tr.get("last_value") or "", samp["answers"])
                    )

                if cpath.exists():
                    existing = load(cpath)
                    if arm == "D_planner_exact_recency":
                        d_case_cache = existing
                    continue

                if arm == "E_plus_semantic_rel" and ctx_str == ctx_d and d_case_cache is not None:
                    row = {
                        **d_case_cache,
                        "arm": arm,
                        "reused_from": "D_planner_exact_recency",
                        "gold_alias_hit": gold_alias_hit,
                        "full_chain_hit": full_chain_hit,
                    }
                    dump_json(cpath, row)
                    continue

                msgs = scorer.messages(samp, ctx_str)
                ans_call = meter.call(f"{arm}--{sid}", msgs, MAX_TOKENS, json_output=False)
                correct = bool(not ans_call["error"] and scorer.score(ans_call["answer"], samp["answers"]))
                row = {
                    "dataset": "factcon",
                    "arm": arm,
                    "sample_id": sid,
                    "source": samp["source"],
                    "context_id": cid,
                    "question": q,
                    "reference": samp["answers"],
                    "prediction": ans_call["answer"],
                    "correct": correct,
                    "error": ans_call["error"],
                    "request_id": ans_call["id"],
                    "usage": ans_call["usage"],
                    "usage_usd": ans_call["usage_usd"],
                    "end_to_end_ms": ans_call["elapsed_ms"],
                    "gold_alias_hit": gold_alias_hit,
                    "full_chain_hit": full_chain_hit,
                    "reused_from": None,
                }
                if arm == "D_planner_exact_recency":
                    d_case_cache = row
                dump_json(cpath, row)

            dump_json(out / "traces" / f"{sid}.json", trace_out)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(process_one, s) for s in ctx_rows]
            for fut in as_completed(futs):
                fut.result()

    # Aggregate A-E + F (carr_v2_planner_only) + G (carr_v2) using offline audit for F/G full-chain
    off_aud = load(ROOT / "results" / "analysis" / "revision_offline_audit.json")
    g_mech = off_aud["mechanistic_full_chain"]["mechanistic"]

    rows_by_arm: dict[str, list[dict]] = {}
    for arm in NEW_ARMS:
        rows_by_arm[arm] = [load(p) for p in sorted((out / "cases").glob(f"{arm}--*.json"))]

    # Load F & G directly from carr-v2-formal-20261124
    fg_map = {"F_plus_hybrid_fallback": "carr_v2_planner_only", "G_full_carr_v2": "carr_v2"}
    for matrix_arm, src_arm in fg_map.items():
        loaded_rows = []
        for s in samples:
            c = load(RUN_FC_V2 / "cases" / f"{src_arm}--{s['id']}.json")
            c_copy = dict(c)
            c_copy["arm"] = matrix_arm
            c_copy["reused_from"] = f"carr-v2-formal-20261124:{src_arm}"
            loaded_rows.append(c_copy)
        rows_by_arm[matrix_arm] = loaded_rows

    summary_rows = []
    for arm in ALL_MATRIX_ARMS:
        arm_cases = rows_by_arm[arm]
        for grp in ("all", "sh", "mh", "32k", "64k", "262k"):
            sub = [c for c in arm_cases if grp == "all" or f"_{grp}_" in c["sample_id"] or c["source"].endswith(f"_{grp}")]
            if not sub:
                continue
            # For prompt tokens / USD: deduplicate E's reused calls or report both effective & incremental
            ans_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
            ans_usd = sum(Decimal(str(c.get("usage_usd") or "0")) for c in sub)
            corr = sum(int(bool(c["correct"])) for c in sub)
            errs = sum(int(bool(c.get("error"))) for c in sub)
            ga_hits = sum(int(bool(c.get("gold_alias_hit"))) for c in sub) if arm in NEW_ARMS else None
            fc_hits = sum(int(bool(c.get("full_chain_hit"))) for c in sub) if arm in NEW_ARMS else None
            if arm == "G_full_carr_v2" and grp in g_mech:
                ga_hits = g_mech[grp]["gold_alias_hit"]
                fc_hits = g_mech[grp]["full_chain_hit"]
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "correct": corr,
                "acc_pct": round(100.0 * corr / len(sub), 2),
                "errors": errs,
                "ans_prompt_tokens": ans_in,
                "ans_input_per_q": round(ans_in / len(sub), 1),
                "ans_usd": str(ans_usd),
                "gold_alias_hits": ga_hits,
                "full_chain_hits": fc_hits,
            })

    # Paired McNemar comparisons
    by_sid = {arm: {c["sample_id"]: c for c in rows_by_arm[arm]} for arm in ALL_MATRIX_ARMS}
    pair_specs = [
        ("B_latest_dict_hybrid", "A_fact_hybrid"),
        ("D_planner_exact_recency", "B_latest_dict_hybrid"),
        ("D_planner_exact_recency", "C_planner_no_recency"),
        ("E_plus_semantic_rel", "D_planner_exact_recency"),
        ("F_plus_hybrid_fallback", "E_plus_semantic_rel"),
        ("F_plus_hybrid_fallback", "D_planner_exact_recency"),
        ("G_full_carr_v2", "D_planner_exact_recency"),
        ("G_full_carr_v2", "F_plus_hybrid_fallback"),
    ]
    paired = {}
    for a, b in pair_specs:
        ma, mb = by_sid[a], by_sid[b]
        w = sum(1 for sid in ma if ma[sid]["correct"] and not mb[sid]["correct"])
        l = sum(1 for sid in ma if not ma[sid]["correct"] and mb[sid]["correct"])
        paired[f"{a}_vs_{b}"] = {"wins": w, "losses": l, "p": _exact_mcnemar(w, l)}

    report = {
        "generator": "src/run_ablation_matrix_v2.py",
        "generated_utc": utc_now().isoformat(),
        "incremental_api_usd": str(meter.usage_cost),
        "rows": summary_rows,
        "paired_mcnemar": paired,
    }
    dump_json(out / "summary.json", report)
    write_csv(out / "summary.csv", summary_rows)

    print("=== Phase 2B A-G Ablation Matrix Summary ===")
    for arm in ALL_MATRIX_ARMS:
        r_all = next(x for x in summary_rows if x["arm"] == arm and x["group"] == "all")
        r_sh = next(x for x in summary_rows if x["arm"] == arm and x["group"] == "sh")
        r_mh = next(x for x in summary_rows if x["arm"] == arm and x["group"] == "mh")
        print(
            f"{arm:24s} | All={r_all['correct']}/562 ({r_all['acc_pct']:5.2f}%) "
            f"| SH={r_sh['correct']}/286 ({r_sh['acc_pct']:5.2f}%) "
            f"| MH={r_mh['correct']}/276 ({r_mh['acc_pct']:5.2f}%) "
            f"| FullChain={r_all['full_chain_hits']} | Err={r_all['errors']}"
        )
    print("Incremental API USD:", meter.usage_cost)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("5.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run_ablation(args.execute, args.budget, args.key_file, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
