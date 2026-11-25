"""Controlled HippoRAG-style PPR formal runner for FactConsolidation (562Q) and LongMemEval-S* (240Q).

Reuses cached/parsed entity-relation edges (0 write-LLM calls) and runs two arms:
  - hipporag_ppr: full raw entity-relation graph + PPR diffusion
  - hipporag_tombstone_ppr: tombstone-pruned active graph + PPR diffusion
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.long_context import MAX_TOKENS
from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.hipporag_ppr import GraphFact, HippoGraphIndex
from src.carr_v2.open_parser import (
    archive_v2, format_longmemeval_transcript, parse_longmemeval_sessions, parse_numbered_v2,
)
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import norm
from src.longmemeval_data import read_longmemeval_locked
from src.run_carr_v2 import SafeMeter, load_factcon_split
from src.run_longmemeval import LME_MAX_TOKENS, longmemeval_messages, score_longmemeval
from src.run_v2 import BGE_FOLDER
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE
from src.v2_transport import FLASH, credentials

HIPPO_ARMS = ("hipporag_ppr", "hipporag_tombstone_ppr")
LME_CACHE_DIR = ROOT / "results/test/longmemeval-v2-formal-20261124/cache"


def _load_embedder():
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE,
        cache_dir=str(ROOT / ".venv/bge_cache"),
        threads=2,
        local_files_only=True,
        specific_model_path=str(BGE_FOLDER),
    )
    lock = Lock()

    class LockedEmbedder:
        def embed(self, texts, batch_size=32):
            with lock:
                return list(embedder.embed(texts, batch_size=batch_size))

    return LockedEmbedder()


def _fc_facts(raw_text: str) -> tuple[list[GraphFact], list[GraphFact]]:
    raw_lines = raw_text.splitlines()
    records, _, decisions = archive_v2(raw_text, "numbered")
    tombstones = {d["removed_line"] for d in decisions}
    raw_facts: list[GraphFact] = []
    active_facts: list[GraphFact] = []
    for rec in sorted(records.values(), key=lambda r: r.order):
        obs = parse_numbered_v2(rec)
        if obs is None:
            continue
        gf = GraphFact(
            fact_id=rec.index,
            entity=obs.entity,
            relation=obs.relation,
            value=obs.value,
            order=rec.order,
            text=raw_lines[rec.index],
        )
        raw_facts.append(gf)
        if rec.index not in tombstones:
            active_facts.append(gf)
    return raw_facts, active_facts


def _lme_facts_from_cache(cid: str) -> tuple[list[GraphFact], list[GraphFact], int]:
    cache_path = LME_CACHE_DIR / f"carr_lme_facts_{cid[:12]}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached extracted dialogue facts: {cache_path}")
    raw_list = load(cache_path)["facts"]

    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for idx, f in enumerate(raw_list):
        groups[(norm(f["entity"]), f["attribute"])].append((idx, f))
    superseded: set[int] = set()
    for _, items in groups.items():
        if len(items) < 2:
            continue
        max_sess = max(f["session_idx"] for _, f in items)
        latest_vals = {norm(f["value"]) for _, f in items if f["session_idx"] == max_sess}
        has_replace = any(f["operation"] == "replace" and f["session_idx"] == max_sess for _, f in items)
        if len(latest_vals) == 1 and has_replace:
            w = next(iter(latest_vals))
            for idx, f in items:
                if f["session_idx"] < max_sess and norm(f["value"]) != w:
                    superseded.add(idx)

    raw_facts: list[GraphFact] = []
    active_facts: list[GraphFact] = []
    for idx, f in enumerate(raw_list):
        line = f"[S{f['session_idx']:03d} | {f['chat_time']}] {f['entity']} - {f['attribute']}: {f['value']} (evidence: \"{f['quote']}\")"
        gf = GraphFact(
            fact_id=idx,
            entity=f["entity"],
            relation=f["attribute"],
            value=f["value"],
            order=f["session_idx"] * 10000 + idx,
            text=line,
        )
        raw_facts.append(gf)
        if idx not in superseded:
            active_facts.append(gf)
    return raw_facts, active_facts, len(superseded)


def run_factcon(out: Path, budget: Decimal, limit: int | None, execute: bool, max_workers: int = 10) -> dict:
    rows, contexts = load_factcon_split("test", limit)
    n = len(rows)
    est_usd = token_cost(0, n * 450 * 2, n * MAX_TOKENS * 2, "peak")
    if not execute:
        print(f"Dry run (factcon HippoRAG PPR): {n}Q x 2 arms = {n*2} calls, est peak ~${est_usd:.4f}")
        return {"suite": "factcon", "questions": n, "est_peak_usd": str(est_usd)}

    key = credentials(FLASH, None)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY.")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "traces", "requests"):
        (out / sub).mkdir(exist_ok=True)
    dump_json(out / "manifest.json", {
        "generator": "src/run_hipporag_v2.py",
        "suite": "factcon",
        "questions": n,
        "arms": list(HIPPO_ARMS),
        "budget_cap_usd": str(budget),
        "status": "running",
        "started_utc": utc_now().isoformat(),
    })

    meter = SafeMeter(out, key, budget)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    embedder = _load_embedder()

    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r)

    for cid, ctx_rows in sorted(by_ctx.items()):
        raw_facts, active_facts = _fc_facts(contexts[cid])
        idx_raw = HippoGraphIndex(raw_facts, embedder, damping=0.5, num_iters=25)
        idx_tomb = HippoGraphIndex(active_facts, embedder, damping=0.5, num_iters=25)

        def one(sample: dict) -> None:
            sid = sample["id"]
            q = sample["question"]
            arm_traces = {}
            for arm, g_idx in (("hipporag_ppr", idx_raw), ("hipporag_tombstone_ppr", idx_tomb)):
                cp = out / "cases" / f"{arm}--{sid}.json"
                if cp.exists():
                    continue
                ctx_str, tr = g_idx.search(q, top_k=8, max_bytes=3600, chronological_output=True)
                arm_traces[arm] = tr
                msgs = scorer.messages(sample, ctx_str)
                call = meter.call(f"{arm}--{sid}", msgs, MAX_TOKENS, json_output=False)
                corr = bool(not call["error"] and scorer.score(call["answer"], sample["answers"]))
                dump_json(cp, {
                    "dataset": "factcon",
                    "arm": arm,
                    "sample_id": sid,
                    "source": sample["source"],
                    "context_id": cid,
                    "question": q,
                    "reference": sample["answers"],
                    "prediction": call["answer"],
                    "correct": corr,
                    "error": call["error"],
                    "request_id": call["id"],
                    "usage": call["usage"],
                    "usage_usd": call["usage_usd"],
                    "end_to_end_ms": call["elapsed_ms"],
                })
            if arm_traces:
                dump_json(out / "traces" / f"{sid}.json", arm_traces)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(one, ctx_rows))

    all_cases = [load(p) for p in sorted((out / "cases").glob("*.json"))]
    summary_rows = []
    for arm in HIPPO_ARMS:
        sub_all = [c for c in all_cases if c["arm"] == arm]
        for grp in ("all", "sh", "mh", "32k", "64k", "262k"):
            sub = sub_all if grp == "all" else [c for c in sub_all if f"_{grp}_" in c["sample_id"]]
            if not sub:
                continue
            pt = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
            usd = sum(Decimal(c.get("usage_usd") or "0") for c in sub)
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "correct": sum(1 for c in sub if c["correct"]),
                "acc_pct": round(100.0 * sum(1 for c in sub if c["correct"]) / len(sub), 2),
                "errors": sum(1 for c in sub if c.get("error")),
                "ans_prompt_tokens": pt,
                "ans_input_per_q": round(pt / len(sub), 1),
                "ans_usd": str(usd),
                "mean_api_ms": round(mean(c["end_to_end_ms"] for c in sub), 2),
            })
    summary = {"generator": "src/run_hipporag_v2.py", "suite": "factcon", "rows": summary_rows}
    dump_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary_rows)
    man = load(out / "manifest.json")
    man["status"] = "complete"
    man["ended_utc"] = utc_now().isoformat()
    man["usage_usd"] = str(meter.usage_cost)
    dump_json(out / "manifest.json", man)
    print(json.dumps({"out": str(out), "usage_usd": str(meter.usage_cost), "rows": [r for r in summary_rows if r["group"] in ("all", "sh", "mh")]}, indent=2))
    return summary


def run_lme(out: Path, budget: Decimal, limit: int | None, execute: bool, max_workers: int = 8) -> dict:
    rows, contexts = read_longmemeval_locked("test")
    if limit is not None:
        rows = rows[:limit]
    n = len(rows)
    est_usd = token_cost(0, n * 1900 * 2, n * LME_MAX_TOKENS * 2, "peak")
    if not execute:
        print(f"Dry run (longmemeval HippoRAG PPR): {n}Q x 2 arms = {n*2} calls, 0 extraction calls, est peak ~${est_usd:.4f}")
        return {"suite": "longmemeval", "questions": n, "est_peak_usd": str(est_usd)}

    key = credentials(FLASH, None)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY.")
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "traces", "requests"):
        (out / sub).mkdir(exist_ok=True)
    dump_json(out / "manifest.json", {
        "generator": "src/run_hipporag_v2.py",
        "suite": "longmemeval",
        "questions": n,
        "arms": list(HIPPO_ARMS),
        "budget_cap_usd": str(budget),
        "status": "running",
        "started_utc": utc_now().isoformat(),
    })

    meter = SafeMeter(out, key, budget)
    embedder = _load_embedder()
    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r)

    for cid, ctx_rows in sorted(by_ctx.items()):
        sessions = parse_longmemeval_sessions(contexts[cid])
        transcript = format_longmemeval_transcript(sessions)
        hybrid_turns = HybridLexicalDenseIndex(embedder, transcript, tombstones=None)
        raw_facts, active_facts, superseded_cnt = _lme_facts_from_cache(cid)
        idx_raw = HippoGraphIndex(raw_facts, embedder, damping=0.5, num_iters=25)
        idx_tomb = HippoGraphIndex(active_facts, embedder, damping=0.5, num_iters=25)

        def one(sample: dict) -> None:
            sid = sample["id"]
            q = sample["question"]
            turn_ctx, turn_meta = retrieve_bm25_recency(hybrid_turns, q, top_k=8, max_lines=14, max_bytes=4800)
            for arm, g_idx in (("hipporag_ppr", idx_raw), ("hipporag_tombstone_ppr", idx_tomb)):
                cp = out / "cases" / f"{arm}--{sid}.json"
                if cp.exists():
                    continue
                fact_ctx, f_tr = g_idx.search(q, top_k=12, max_bytes=3600, chronological_output=True)
                parts = []
                if fact_ctx:
                    parts.append("Verified Chronological Facts:\n" + fact_ctx)
                if turn_ctx:
                    parts.append("Relevant Chronological Dialogue Turns:\n" + turn_ctx)
                ctx_str = "\n\n".join(parts)
                msgs = longmemeval_messages(ctx_str, q)
                call = meter.call(f"{arm}--{sid}", msgs, LME_MAX_TOKENS, json_output=False)
                metrics = (
                    score_longmemeval(call["answer"], sample["answers"])
                    if not call["error"]
                    else {"substring_exact_match": False, "exact_match": False, "f1": 0.0, "rougeL_f1": 0.0, "rougeL_recall": 0.0}
                )
                dump_json(cp, {
                    "dataset": "longmemeval_s",
                    "arm": arm,
                    "sample_id": sid,
                    "question_type": sample["question_type"],
                    "context_id": cid,
                    "question": q,
                    "reference": sample["answers"],
                    "prediction": call["answer"],
                    "correct": metrics["substring_exact_match"],
                    "metrics": metrics,
                    "error": call["error"],
                    "request_id": call["id"],
                    "usage": call["usage"],
                    "usage_usd": call["usage_usd"],
                    "end_to_end_ms": call["elapsed_ms"],
                    "superseded_facts_masked": superseded_cnt,
                    "trace": {"fact_ppr": f_tr, "turns": turn_meta["chunk_ids"]},
                })

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(one, ctx_rows))

    all_cases = [load(p) for p in sorted((out / "cases").glob("*.json"))]
    summary_rows = []
    cats = sorted({c["question_type"] for c in all_cases})
    for arm in HIPPO_ARMS:
        sub_all = [c for c in all_cases if c["arm"] == arm]
        for grp in ["all", "factual_5cat"] + cats:
            if grp == "all":
                sub = sub_all
            elif grp == "factual_5cat":
                sub = [c for c in sub_all if c["question_type"] != "single-session-preference"]
            else:
                sub = [c for c in sub_all if c["question_type"] == grp]
            if not sub:
                continue
            pt = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
            usd = sum(Decimal(c.get("usage_usd") or "0") for c in sub)
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "subem_correct": sum(1 for c in sub if c["correct"]),
                "subem_pct": round(100.0 * sum(1 for c in sub if c["correct"]) / len(sub), 2),
                "mean_f1": round(100.0 * mean(float(c["metrics"]["f1"]) for c in sub), 2),
                "mean_rougeL_f1": round(100.0 * mean(float(c["metrics"]["rougeL_f1"]) for c in sub), 2),
                "errors": sum(1 for c in sub if c.get("error")),
                "ans_prompt_tokens": pt,
                "ans_input_per_q": round(pt / len(sub), 1),
                "ans_usd": str(usd),
                "mean_api_ms": round(mean(c["end_to_end_ms"] for c in sub), 2),
            })
    summary = {"generator": "src/run_hipporag_v2.py", "suite": "longmemeval", "rows": summary_rows}
    dump_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary_rows)
    man = load(out / "manifest.json")
    man["status"] = "complete"
    man["ended_utc"] = utc_now().isoformat()
    man["usage_usd"] = str(meter.usage_cost)
    dump_json(out / "manifest.json", man)
    print(json.dumps({"out": str(out), "usage_usd": str(meter.usage_cost), "rows": [r for r in summary_rows if r["group"] in ("all", "factual_5cat", "knowledge-update", "multi-session")]}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("factcon", "longmemeval"), required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--budget", type=Decimal, default=Decimal("5.0"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if args.suite == "factcon":
        run_factcon(args.run_dir, args.budget, args.limit, args.execute, max_workers=args.workers)
    else:
        run_lme(args.run_dir, args.budget, args.limit, args.execute, max_workers=args.workers)


if __name__ == "__main__":
    main()
