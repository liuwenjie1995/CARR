"""Metered, checkpoint-resumable LongMemEval-S* evaluation runner across 6 arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
from pathlib import Path
import re
from statistics import mean
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MemoryAgentBench.utils.eval_other_utils import post_process
from baselines.MemoryAgentBench.utils.templates import get_template
from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.ircot_baseline import run_ircot
from src.baselines_v2.mem0_style import consolidate_mem0_history
from src.carr_v2.open_parser import (
    extract_dialogue_facts_batch,
    format_longmemeval_transcript,
    parse_longmemeval_sessions,
)
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex, TombstoneDenseIndex, tokenize_lexical
from src.evaluator import dump_json, token_cost, utc_now
from src.factual_v2 import norm
from src.longmemeval_data import read_longmemeval_locked
from src.run_carr_v2 import SafeMeter
from src.run_v2 import BGE_FOLDER
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE, QUERY_PREFIX, normalize_rows
from src.v2_transport import FLASH, credentials

LME_ARMS = (
    "official_full_context",
    "naive_bge_top8",
    "bm25_bge_recency",
    "ircot_bge",
    "mem0_style",
    "carr_v2",
)
LME_MAX_TOKENS = 64


def longmemeval_messages(context: str, question: str) -> list[dict]:
    sys_msg = get_template("longmemeval_", "system", "Long_context_agent")
    mem_tmpl = get_template("longmemeval_", "memorize", "Long_context_agent")
    qry_tmpl = get_template("longmemeval_", "query", "Long_context_agent")
    user_msg = mem_tmpl.format(context=context) + "\n\n" + qry_tmpl.format(question=question)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def score_longmemeval(prediction: str, references: list[str]) -> dict:
    metrics, _ = post_process({"output": prediction}, references, {"sub_dataset": "longmemeval_s"})
    return {
        "substring_exact_match": bool(metrics["substring_exact_match"]),
        "exact_match": bool(metrics["exact_match"]),
        "f1": float(metrics["f1"]),
        "rougeL_f1": float(metrics["rougeL_f1"]),
        "rougeL_recall": float(metrics["rougeL_recall"]),
    }


def _build_lme_fact_index(facts: list[dict], embedder):
    """Build conservative recency-resolved fact store over span-verified dialogue facts."""
    # Group by (norm(entity), attribute) to detect explicit replace updates
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for idx, f in enumerate(facts):
        groups[(norm(f["entity"]), f["attribute"])].append((idx, f))

    superseded_ids: set[int] = set()
    for key, items in groups.items():
        if len(items) < 2:
            continue
        max_sess = max(f["session_idx"] for _, f in items)
        latest_vals = {norm(f["value"]) for _, f in items if f["session_idx"] == max_sess}
        has_replace = any(f["operation"] == "replace" and f["session_idx"] == max_sess for _, f in items)
        if len(latest_vals) == 1 and has_replace:
            winner_val = next(iter(latest_vals))
            for idx, f in items:
                if f["session_idx"] < max_sess and norm(f["value"]) != winner_val:
                    superseded_ids.add(idx)

    active = [f for idx, f in enumerate(facts) if idx not in superseded_ids]
    # Sort chronologically by session_idx
    active.sort(key=lambda f: f["session_idx"])
    fact_lines = [
        f"[S{f['session_idx']:03d} | {f['chat_time']}] {f['entity']} - {f['attribute']}: {f['value']} (evidence: \"{f['quote']}\")"
        for f in active
    ]
    fact_vecs = normalize_rows(list(embedder.embed(fact_lines, batch_size=32))) if fact_lines else None
    return active, fact_lines, fact_vecs, len(superseded_ids)


def retrieve_carr_v2_lme(
    question: str,
    fact_lines: list[str],
    fact_vecs,
    active_facts: list[dict],
    hybrid_turns: HybridLexicalDenseIndex,
    embedder,
) -> tuple[str, dict]:
    """Combine recency-resolved extracted dialogue facts with hybrid chronological turn fallback."""
    selected_facts: list[tuple[int, str]] = []
    if fact_lines and fact_vecs is not None:
        qvec = normalize_rows(list(embedder.embed([QUERY_PREFIX + question])))[0]
        dense_s = fact_vecs @ qvec
        qtoks = set(tokenize_lexical(question))
        scored = []
        for idx, (line, f) in enumerate(zip(fact_lines, active_facts)):
            lex = len(qtoks & set(tokenize_lexical(line)))
            # Boost exact lexical/entity hits + semantic match
            comb = float(dense_s[idx]) + 0.05 * min(lex, 4)
            scored.append((comb, f["session_idx"], idx, line))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        for comb, s_idx, idx, line in scored[:12]:
            selected_facts.append((s_idx, line))
        selected_facts.sort(key=lambda x: x[0])

    # Pair with Top-8 chronological dialogue turns from hybrid index
    turn_ctx, turn_meta = retrieve_bm25_recency(hybrid_turns, question, top_k=8, max_lines=14, max_bytes=4800)
    parts = []
    if selected_facts:
        parts.append("Verified Chronological Facts:\n" + "\n".join(x[1] for x in selected_facts))
    if turn_ctx:
        parts.append("Relevant Chronological Dialogue Turns:\n" + turn_ctx)
    return "\n\n".join(parts), {
        "selected_facts": len(selected_facts),
        "turn_chunks": turn_meta["chunk_ids"],
        "selected_turns": turn_meta["selected_lines"],
    }


def compute_quote(split: str, limit: int | None, arms: tuple[str, ...]) -> dict:
    rows, contexts = read_longmemeval_locked(split)
    if limit is not None:
        rows = rows[:limit]
    n = len(rows)
    n_hist = len({r["context_id"] for r in rows})
    total_sessions = sum(len(parse_longmemeval_sessions(contexts[cid])) for cid in {r["context_id"] for r in rows})

    est = {}
    total_usd = Decimal(0)
    for arm in arms:
        if arm == "official_full_context":
            in_tok = n * 410_000
            out_tok = n * LME_MAX_TOKENS
            calls = n
        elif arm in {"naive_bge_top8", "bm25_bge_recency"}:
            in_tok = n * 2400
            out_tok = n * LME_MAX_TOKENS
            calls = n
        elif arm == "ircot_bge":
            in_tok = n * (3 * 1800 + 2400)
            out_tok = n * (3 * 96 + LME_MAX_TOKENS)
            calls = n * 4
        elif arm == "mem0_style":
            in_tok = total_sessions * 2 * 900 + n * 1200
            out_tok = total_sessions * 2 * 350 + n * LME_MAX_TOKENS
            calls = total_sessions * 2 + n
        elif arm == "carr_v2":
            in_tok = total_sessions * 1400 + n * 2000
            out_tok = total_sessions * 220 + n * LME_MAX_TOKENS
            calls = total_sessions + n
        else:
            raise ValueError(arm)
        usd = token_cost(0, in_tok, out_tok, "peak")
        est[arm] = {"calls": calls, "est_input_tokens": in_tok, "est_peak_usd": str(usd)}
        total_usd += usd
    return {"split": split, "histories": n_hist, "questions": n, "total_sessions": total_sessions, "arms": est, "total_est_peak_usd": str(total_usd)}


def run_longmemeval(
    split: str,
    limit: int | None,
    arms: tuple[str, ...],
    out: Path,
    budget: Decimal,
    key_file: Path | None,
    *,
    max_workers: int = 8,
) -> dict:
    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY.")

    rows, contexts = read_longmemeval_locked(split)
    if limit is not None:
        rows = rows[:limit]

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "traces", "cache", "requests"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
        local_files_only=True, specific_model_path=str(BGE_FOLDER),
    )

    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r)

    manifest = {
        "generator": "src/run_longmemeval.py",
        "split": split,
        "questions": len(rows),
        "arms": list(arms),
        "budget_cap_usd": str(budget),
        "status": "running",
        "started_utc": utc_now().isoformat(),
    }
    dump_json(out / "manifest.json", manifest)

    extraction_usage: dict[str, dict] = {}
    mem0_write_usage: dict[str, dict] = {}
    embed_lock = Lock()

    class LockedEmbedder:
        def __init__(self, inner):
            self.inner = inner

        def embed(self, texts, batch_size=32):
            with embed_lock:
                return list(self.inner.embed(texts, batch_size=batch_size))

    safe_embedder = LockedEmbedder(embedder)

    parsed_sessions_by_ctx = {cid: parse_longmemeval_sessions(contexts[cid]) for cid in sorted(by_ctx)}
    transcript_by_ctx = {cid: format_longmemeval_transcript(sess) for cid, sess in parsed_sessions_by_ctx.items()}

    mem0_stores: dict[str, object] = {}
    if "mem0_style" in arms:
        def build_mem0_lme(cid: str) -> tuple[str, object, dict]:
            turn_lines = transcript_by_ctx[cid].splitlines()
            stmts = [{"idx": i, "order": i, "text": ln} for i, ln in enumerate(turn_lines) if ln.strip()]
            mstore, w_use = consolidate_mem0_history(
                stmts,
                safe_embedder,
                lambda tag, msgs, mtok, jo: meter.call(f"lme_mem0_{cid[:8]}_{tag}", msgs, mtok, json_output=jo),
                cache_path=out / "cache" / f"mem0_lme_{cid[:12]}.json",
                batch_size=24,
            )
            return cid, mstore, w_use

        with ThreadPoolExecutor(max_workers=max(1, len(by_ctx))) as mpool:
            for cid, mstore, w_use in mpool.map(build_mem0_lme, sorted(by_ctx)):
                mem0_stores[cid] = mstore
                mem0_write_usage[cid] = w_use

    for cid, ctx_rows in sorted(by_ctx.items()):
        raw_context = contexts[cid]
        sessions = parsed_sessions_by_ctx[cid]
        transcript = transcript_by_ctx[cid]

        need_index = any(a != "official_full_context" for a in arms)
        hybrid_idx = HybridLexicalDenseIndex(embedder, transcript, tombstones=None) if need_index else None
        dense_idx = (
            TombstoneDenseIndex(
                embedder, transcript, tombstones=None,
                _chunks=hybrid_idx.chunks, _vectors=hybrid_idx.vectors,
            )
            if hybrid_idx is not None
            else None
        )

        # CARR v2 span-verified dialogue extraction
        active_facts, fact_lines, fact_vecs, superseded_cnt = [], [], None, 0
        if "carr_v2" in arms:
            extracted, ext_use = extract_dialogue_facts_batch(
                sessions,
                lambda tag, msgs, mtok, jo: meter.call(f"lme_{cid[:8]}_{tag}", msgs, mtok, json_output=jo),
                cache_path=out / "cache" / f"carr_lme_facts_{cid[:12]}.json",
                max_workers=10,
            )
            active_facts, fact_lines, fact_vecs, superseded_cnt = _build_lme_fact_index(extracted, embedder)
            extraction_usage[cid] = {**ext_use, "extracted_total": len(extracted), "superseded_archived": superseded_cnt}

        mem0_store = mem0_stores.get(cid)

        def run_one(sample: dict, arm: str) -> None:
            sid = sample["id"]
            q = sample["question"]
            case_path = out / "cases" / f"{arm}--{sid}.json"
            if case_path.exists():
                return
            extra_calls: list[dict] = []
            if arm == "official_full_context":
                ctx_str = raw_context
                arm_trace = {"full_chars": len(raw_context)}
            elif arm == "naive_bge_top8":
                assert dense_idx is not None
                ctx_str, chunk_ids, _ = dense_idx.search(q, top_k=8)
                arm_trace = {"chunk_ids": chunk_ids}
            elif arm == "bm25_bge_recency":
                assert hybrid_idx is not None
                ctx_str, arm_trace = retrieve_bm25_recency(hybrid_idx, q, top_k=8, max_lines=16, max_bytes=4800)
            elif arm == "ircot_bge":
                assert dense_idx is not None
                ctx_str, arm_trace = run_ircot(
                    dense_idx,
                    q,
                    lambda tag, msgs, mtok: meter.call(tag, msgs, mtok, json_output=False),
                    call_prefix=f"lme_{sid}",
                )
                extra_calls = arm_trace.pop("reasoning_calls", [])
            elif arm == "mem0_style":
                assert mem0_store is not None
                ctx_str, mem_ids = mem0_store.search_context(q, top_k=8)
                arm_trace = {"memory_ids": mem_ids}
            elif arm == "carr_v2":
                assert hybrid_idx is not None
                ctx_str, arm_trace = retrieve_carr_v2_lme(q, fact_lines, fact_vecs, active_facts, hybrid_idx, embedder)
            else:
                raise ValueError(arm)

            msgs = longmemeval_messages(ctx_str, q)
            ans_call = meter.call(f"{arm}--{sid}", msgs, LME_MAX_TOKENS, json_output=False)
            metrics = (
                score_longmemeval(ans_call["answer"], sample["answers"])
                if not ans_call["error"]
                else {"substring_exact_match": False, "exact_match": False, "f1": 0.0, "rougeL_f1": 0.0, "rougeL_recall": 0.0}
            )
            extra_prompt = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in extra_calls)
            extra_usd = sum(Decimal(c.get("usage_usd") or "0") for c in extra_calls)

            dump_json(case_path, {
                "dataset": "longmemeval_s",
                "arm": arm,
                "sample_id": sid,
                "question_type": sample["question_type"],
                "context_id": cid,
                "question": q,
                "reference": sample["answers"],
                "prediction": ans_call["answer"],
                "correct": metrics["substring_exact_match"],
                "metrics": metrics,
                "trace": arm_trace,
                "error": ans_call["error"],
                "request_id": ans_call["id"],
                "usage": ans_call["usage"],
                "usage_usd": ans_call["usage_usd"],
                "extra_prompt_tokens": extra_prompt,
                "extra_usd": str(extra_usd),
                "end_to_end_ms": ans_call["elapsed_ms"],
            })

        ordered = sorted(ctx_rows, key=lambda x: x["id"])
        for arm in arms:
            if arm == "official_full_context":
                # Sequential warmup for first 2 questions per history to populate server prefix cache
                for s in ordered[:2]:
                    run_one(s, arm)
                with ThreadPoolExecutor(max_workers=min(max_workers, 6)) as pool:
                    for f in [pool.submit(run_one, s, arm) for s in ordered[2:]]:
                        f.result()
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for f in [pool.submit(run_one, s, arm) for s in ordered]:
                        f.result()

    summary = summarize_longmemeval(out, arms, extraction_usage, mem0_write_usage)
    manifest["status"] = "complete"
    manifest["ended_utc"] = utc_now().isoformat()
    manifest["usage_usd"] = str(meter.usage_cost)
    dump_json(out / "manifest.json", manifest)
    return summary


def summarize_longmemeval(
    out: Path,
    arms: tuple[str, ...],
    extraction_usage: dict | None = None,
    mem0_write_usage: dict | None = None,
) -> dict:
    cases = [load(p) for p in (out / "cases").glob("*.json")]
    qtypes = sorted({c["question_type"] for c in cases})
    groups = ["all", "factual_5cat"] + qtypes
    rows = []
    for arm in arms:
        arm_cases = [c for c in cases if c["arm"] == arm]
        for grp in groups:
            if grp == "all":
                sub = arm_cases
            elif grp == "factual_5cat":
                sub = [c for c in arm_cases if c["question_type"] != "single-session-preference"]
            else:
                sub = [c for c in arm_cases if c["question_type"] == grp]
            if not sub:
                continue
            ans_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
            ans_hit = sum((c.get("usage") or {}).get("cache_hit_tokens", 0) for c in sub)
            ans_usd = sum(Decimal(c.get("usage_usd") or "0") for c in sub)
            ext_usd = sum(Decimal(c.get("extra_usd") or "0") for c in sub)
            rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "subem_correct": sum(bool(c["correct"]) for c in sub),
                "subem_pct": round(100 * sum(bool(c["correct"]) for c in sub) / len(sub), 2),
                "mean_f1": round(100 * mean(c["metrics"]["f1"] for c in sub), 2),
                "mean_rougeL_f1": round(100 * mean(c["metrics"]["rougeL_f1"] for c in sub), 2),
                "errors": sum(bool(c["error"]) for c in sub),
                "ans_prompt_tokens": ans_in,
                "ans_input_per_q": round(ans_in / len(sub), 1),
                "cache_hit_pct": round(100 * ans_hit / ans_in, 2) if ans_in else 0.0,
                "ans_usd": str(ans_usd),
                "extra_usd": str(ext_usd),
                "mean_api_ms": round(mean(c["end_to_end_ms"] for c in sub), 2),
            })

    summary = {
        "generator": "src/run_longmemeval.py",
        "rows": rows,
        "extraction_usage": extraction_usage or {},
        "mem0_write_usage": mem0_write_usage or {},
    }
    write_csv(out / "summary.csv", rows)
    dump_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(LME_ARMS))
    ap.add_argument("--quote", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("50.0"))
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    selected = tuple(args.arms)
    if args.quote or not args.execute:
        print(json.dumps(compute_quote(args.split, args.limit, selected), indent=2))
    else:
        out_dir = args.run_dir or (ROOT / f"results/{args.split}/longmemeval-v2-{utc_now():%Y%m%dT%H%M%SZ}")
        res = run_longmemeval(args.split, args.limit, selected, out_dir, args.budget, args.key_file, max_workers=args.workers)
        print(json.dumps({
            "out_dir": str(out_dir),
            "summary_rows": [r for r in res["rows"] if r["group"] in {"all", "factual_5cat", "knowledge-update", "temporal-reasoning"}],
        }, indent=2))
