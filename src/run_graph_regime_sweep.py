"""Single-variable Graph-Density & Hop-Depth Phase-Transition Experiment (Concern #3).

Holds the FactConsolidation 262K questions (197Q), parser, and answering model constant
while sweeping distractor graph size across 4 scales:
  - 500 facts (sparse, comparable to LongMemEval-S* ~588 facts)
  - 2,000 facts (medium-sparse)
  - 6,000 facts (medium-dense)
  - full (~16,450 active / 18,290 parsed facts, full dense 262K)

Measures entity degree topology, offline Gold-Alias Recall (overall, SH, MH, 2-hop, 3+ hop),
and end-to-end DeepSeek Flash accuracy for HippoRAG Tombstone+PPR vs. CARR v2 Directed Relay.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.long_context import MAX_TOKENS
from src.baselines_v2.hipporag_ppr import GraphFact, HippoGraphIndex
from src.carr_v2.open_parser import archive_v2, parse_numbered_v2
from src.carr_v2.semantic_router import RecencyHybridStore, retrieve_v2
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import norm
from src.run_carr_v2 import SafeMeter
from src.run_hipporag_v2 import _load_embedder
from src.v2_data import read_locked
from src.v2_report import load
from src.v2_transport import FLASH, credentials

RUN_FC_V2 = ROOT / "results" / "test" / "carr-v2-formal-20261124"
RUN_HIPPO_FC = ROOT / "results" / "test" / "hipporag-fc-formal-20261124"
DEFAULT_OUT = ROOT / "results" / "test" / "graph-regime-sweep"
OUT_AUDIT = ROOT / "results" / "analysis" / "graph_regime_sweep.json"

SCALES = (500, 2000, 6000, -1)  # -1 means full active graph


def _norm_ans(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _has_gold_alias(ctx: str, refs: list[str]) -> bool:
    nc = _norm_ans(ctx)
    if not nc:
        return False
    for r in refs:
        nr = _norm_ans(r)
        if not nr:
            continue
        if len(nr) < 3:
            if nr in nc.split():
                return True
        elif re.search(r"(?<!\w)" + re.escape(nr) + r"(?!\w)", nc):
            return True
    return False


def _prepare_262k(embedder):
    samples, contexts = read_locked("factcon", "test")
    rows_262 = [s for s in samples if "_262k_" in s["id"] or s["source"].endswith("_262k")]
    cid = rows_262[0]["context_id"]
    raw_text = contexts[cid]
    raw_lines = raw_text.splitlines()
    records, _clean, decisions = archive_v2(raw_text, "numbered")
    tomb_line_ids = {d["removed_line"] for d in decisions}

    parsed_by_order = {}
    all_records_sorted = sorted(records.values(), key=lambda r: r.order)
    for rec in all_records_sorted:
        obs = parse_numbered_v2(rec)
        if obs is not None:
            parsed_by_order[rec.order] = (rec, obs)

    # Identify essential fact orders from CARR v2 traces + all matching answer/start facts
    essential_orders: set[int] = set()
    plans: dict[str, dict] = {}
    hop_depth: dict[str, int] = {}
    full_v2_cases: dict[str, dict] = {}
    full_hip_cases: dict[str, dict] = {}

    for s in rows_262:
        sid = s["id"]
        tr = load(RUN_FC_V2 / "traces" / f"{sid}.json")
        plans[sid] = tr.get("plan") or {}
        hop_depth[sid] = len((plans[sid].get("steps") or [])) if "_mh_" in sid else 1
        cv2 = tr.get("carr_v2") or {}
        for fid in cv2.get("fact_ids") or []:
            essential_orders.add(fid)
        for ev in cv2.get("events") or []:
            if ev.get("chosen") is not None:
                essential_orders.add(ev["chosen"])
        full_v2_cases[sid] = load(RUN_FC_V2 / "cases" / f"carr_v2--{sid}.json")
        full_hip_cases[sid] = load(RUN_HIPPO_FC / "cases" / f"hipporag_tombstone_ppr--{sid}.json")

    # Active non-tombstone parsed orders
    active_orders = [ord_ for ord_, (rec, _obs) in parsed_by_order.items() if rec.index not in tomb_line_ids]
    essential_active = [o for o in active_orders if o in essential_orders]
    distractor_active = [o for o in active_orders if o not in essential_orders]

    return {
        "rows": rows_262,
        "cid": cid,
        "raw_lines": raw_lines,
        "parsed_by_order": parsed_by_order,
        "tomb_line_ids": tomb_line_ids,
        "active_orders": active_orders,
        "essential_active": essential_active,
        "distractor_active": distractor_active,
        "plans": plans,
        "hop_depth": hop_depth,
        "full_v2_cases": full_v2_cases,
        "full_hip_cases": full_hip_cases,
    }


def run_sweep(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    embedder = _load_embedder()
    prep = _prepare_262k(embedder)
    rows = prep["rows"]
    n_ess = len(prep["essential_active"])
    n_full = len(prep["active_orders"])

    if not execute:
        # 3 new PPR scales (500, 2000, 6000) x 197Q; CARR v2 reuses identical context prompts
        est_calls = len(rows) * 3
        est_usd = token_cost(0, est_calls * 380, est_calls * MAX_TOKENS, "peak")
        q = {
            "questions_262k": len(rows),
            "essential_active_facts": n_ess,
            "full_active_facts": n_full,
            "scales": [s if s > 0 else n_full for s in SCALES],
            "est_new_calls": est_calls,
            "est_peak_usd": str(est_usd),
        }
        print(json.dumps(q, indent=2))
        return q

    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DeepSeek API key.")

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "requests"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    rng = random.Random(42)
    shuffled_distractors = list(prep["distractor_active"])
    rng.shuffle(shuffled_distractors)

    scale_results = []
    parsed_by_order = prep["parsed_by_order"]
    raw_lines = prep["raw_lines"]

    for target in SCALES:
        actual_k = n_full if target < 0 or target >= n_full else target
        scale_tag = "full" if target < 0 else str(actual_k)
        if actual_k == n_full:
            chosen_orders = sorted(prep["active_orders"])
        else:
            need_dist = max(0, actual_k - n_ess)
            chosen_orders = sorted(prep["essential_active"] + shuffled_distractors[:need_dist])

        # Build subset text (preserving original chronological order numbers) & GraphFact list
        subset_recs = [parsed_by_order[o][0] for o in chosen_orders]
        subset_text = "\n".join(f"{o}. {raw_lines[parsed_by_order[o][0].index]}" for o in chosen_orders)
        subset_raw_lines = subset_text.splitlines()
        # Map original order -> subset line for RecencyHybridStore (using original raw_lines works cleanly since rec.index is preserved!)
        graph_facts = [
            GraphFact(
                fact_id=parsed_by_order[o][0].index,
                entity=parsed_by_order[o][1].entity,
                relation=parsed_by_order[o][1].relation,
                value=parsed_by_order[o][1].value,
                order=o,
                text=raw_lines[parsed_by_order[o][0].index],
            )
            for o in chosen_orders
        ]
        g_idx = HippoGraphIndex(graph_facts, embedder, damping=0.5, num_iters=25)

        # Build subset RecencyHybridStore for CARR v2 on this exact scale
        sub_hybrid = HybridLexicalDenseIndex(embedder, subset_text, tombstones=None)
        carr_store = RecencyHybridStore(subset_recs, raw_lines, embedder, sub_hybrid, use_exact=True)

        topo = {
            "scale_tag": scale_tag,
            "active_facts": len(graph_facts),
            "num_entities": g_idx.num_entities,
            "mean_entity_degree": round(float(g_idx.ent_deg.mean()), 3) if g_idx.num_entities else 0.0,
            "max_entity_degree": int(g_idx.ent_deg.max()) if g_idx.num_entities else 0,
            "p95_entity_degree": round(float(sorted(g_idx.ent_deg.tolist())[int(0.95 * (g_idx.num_entities - 1))]), 1) if g_idx.num_entities else 0.0,
        }

        def eval_sample(s: dict) -> dict:
            sid = s["id"]
            q = s["question"]
            refs = s["answers"]
            hop_grp = "sh" if "_sh_" in sid else "mh"
            hd = prep["hop_depth"][sid]

            # 1. HippoRAG Tombstone+PPR
            hip_ctx, _ = g_idx.search(q, top_k=8, max_bytes=3600, chronological_output=True)
            hip_hit = _has_gold_alias(hip_ctx, refs)
            if scale_tag == "full":
                hip_case = prep["full_hip_cases"][sid]
                hip_ok = bool(hip_case["correct"])
                hip_err = hip_case.get("error")
            else:
                cp = out / "cases" / f"hip_{scale_tag}--{sid}.json"
                if cp.exists():
                    hc = load(cp)
                else:
                    msgs = scorer.messages(s, hip_ctx)
                    call = meter.call(f"hip_{scale_tag}--{sid}", msgs, MAX_TOKENS, json_output=False)
                    hc = {
                        "sample_id": sid,
                        "correct": bool(not call["error"] and scorer.score(call["answer"], refs)),
                        "error": call["error"],
                        "usage_usd": call["usage_usd"],
                    }
                    dump_json(cp, hc)
                hip_ok = bool(hc["correct"])
                hip_err = hc.get("error")

            # 2. CARR v2 Directed Relay on exact same scale
            ev_list, tr_v2 = retrieve_v2(carr_store, prep["plans"][sid], q)
            v2_ctx = "\n".join(ev_list)
            v2_hit = _has_gold_alias(v2_ctx, refs)
            if tr_v2.get("status") == "completed_route" and scale_tag != "full":
                # Check if exact same context as full; if not completed_route or context differs, call LLM
                full_case = prep["full_v2_cases"][sid]
                # Compare chosen fact_ids against full trace
                full_tr = load(RUN_FC_V2 / "traces" / f"{sid}.json")["carr_v2"]
                if tr_v2.get("fact_ids") == full_tr.get("fact_ids"):
                    v2_ok = bool(full_case["correct"])
                    v2_err = full_case.get("error")
                else:
                    cp2 = out / "cases" / f"v2_{scale_tag}--{sid}.json"
                    if cp2.exists():
                        vc = load(cp2)
                    else:
                        msgs2 = scorer.messages(s, v2_ctx)
                        call2 = meter.call(f"v2_{scale_tag}--{sid}", msgs2, MAX_TOKENS, json_output=False)
                        vc = {
                            "sample_id": sid,
                            "correct": bool(not call2["error"] and scorer.score(call2["answer"], refs)),
                            "error": call2["error"],
                        }
                        dump_json(cp2, vc)
                    v2_ok = bool(vc["correct"])
                    v2_err = vc.get("error")
            elif scale_tag == "full":
                full_case = prep["full_v2_cases"][sid]
                v2_ok = bool(full_case["correct"])
                v2_err = full_case.get("error")
            else:
                cp2 = out / "cases" / f"v2_{scale_tag}--{sid}.json"
                if cp2.exists():
                    vc = load(cp2)
                else:
                    msgs2 = scorer.messages(s, v2_ctx)
                    call2 = meter.call(f"v2_{scale_tag}--{sid}", msgs2, MAX_TOKENS, json_output=False)
                    vc = {
                        "sample_id": sid,
                        "correct": bool(not call2["error"] and scorer.score(call2["answer"], refs)),
                        "error": call2["error"],
                    }
                    dump_json(cp2, vc)
                v2_ok = bool(vc["correct"])
                v2_err = vc.get("error")

            return {
                "sample_id": sid,
                "hop_grp": hop_grp,
                "hop_depth": hd,
                "hip_hit": hip_hit,
                "hip_ok": hip_ok,
                "hip_err": hip_err,
                "v2_hit": v2_hit,
                "v2_ok": v2_ok,
                "v2_err": v2_err,
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_q = list(pool.map(eval_sample, rows))

        def agg(sub: list[dict], prefix: str) -> dict:
            n = len(sub)
            hits = sum(int(x[f"{prefix}_hit"]) for x in sub)
            corr = sum(int(x[f"{prefix}_ok"]) for x in sub)
            errs = sum(int(bool(x[f"{prefix}_err"])) for x in sub)
            return {
                "questions": n,
                "recall_hits": hits,
                "recall_pct": round(100.0 * hits / n, 2) if n else 0.0,
                "correct": corr,
                "acc_pct": round(100.0 * corr / n, 2) if n else 0.0,
                "errors": errs,
            }

        groups = {
            "all": per_q,
            "sh": [x for x in per_q if x["hop_grp"] == "sh"],
            "mh": [x for x in per_q if x["hop_grp"] == "mh"],
            "mh_2hop": [x for x in per_q if x["hop_grp"] == "mh" and x["hop_depth"] <= 2],
            "mh_3plus_hop": [x for x in per_q if x["hop_grp"] == "mh" and x["hop_depth"] >= 3],
        }
        scale_results.append({
            "topology": topo,
            "hipporag_tombstone_ppr": {g: agg(sub, "hip") for g, sub in groups.items()},
            "carr_v2": {g: agg(sub, "v2") for g, sub in groups.items()},
        })

    report = {
        "generator": "src/run_graph_regime_sweep.py",
        "generated_utc": utc_now().isoformat(),
        "dataset": "factconsolidation_262k_197Q",
        "essential_active_facts": n_ess,
        "incremental_api_usd": str(meter.usage_cost),
        "scales": scale_results,
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    dump_json(OUT_AUDIT, report)

    print("=== Step 1: Single-Variable Graph-Density & Hop-Depth Phase Transition (262K 197Q) ===")
    for sr in scale_results:
        t = sr["topology"]
        hp = sr["hipporag_tombstone_ppr"]
        cv = sr["carr_v2"]
        print(
            f"Facts={t['active_facts']:5d} (deg={t['mean_entity_degree']:4.2f}, max={t['max_entity_degree']:3d}) | "
            f"PPR All={hp['all']['acc_pct']:5.2f}% SH={hp['sh']['acc_pct']:5.1f}% MH={hp['mh']['acc_pct']:5.1f}% "
            f"(2h={hp['mh_2hop']['acc_pct']:5.1f}%, 3+h={hp['mh_3plus_hop']['acc_pct']:5.1f}%) | "
            f"CARR v2 All={cv['all']['acc_pct']:5.2f}% MH={cv['mh']['acc_pct']:5.1f}%"
        )
    print("Incremental API USD:", meter.usage_cost)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("2.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run_sweep(args.execute, args.budget, args.key_file, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
