"""Metered, checkpoint-resumable runner for CARR v2 and controlled baselines on FactConsolidation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
import os
from pathlib import Path
from statistics import mean
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.long_context import MAX_TOKENS
from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.ircot_baseline import run_ircot
from src.baselines_v2.mem0_style import consolidate_mem0_history
from src.carr_v2.open_parser import archive_v2
from src.carr_v2.semantic_router import (
    RecencyHybridStore,
    parse_plan_v2,
    planner_messages_v2,
    retrieve_v2,
)
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex, TombstoneDenseIndex
from src.evaluator import LongContext, dump_json, load_dev, token_cost, utc_now
from src.factual_v2 import records_for
from src.run_v2 import BGE_FOLDER
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE
from src.v2_transport import FLASH, PROVIDER, credentials, request_json, usage_of

ALL_ARMS = (
    "tombstone_bge_top8",
    "bm25_bge_recency",
    "ircot_bge",
    "mem0_style",
    "carr_v2_planner_only",
    "carr_v2",
)


class SafeMeter:
    """Thread-safe metered API client with retries on transient errors and no key leakage."""

    def __init__(self, out: Path, key: str, cap: Decimal, *, max_retries: int = 3):
        self.out = out
        self.key = key
        self.cap = cap
        self.max_retries = max_retries
        self.spent = Decimal(0)
        self.reserved = Decimal(0)
        self.usage_cost = Decimal(0)
        self.lock = Lock()
        self.stopped = False
        (out / "requests").mkdir(parents=True, exist_ok=True)

    def call(
        self,
        request_id: str,
        messages: list[dict],
        max_tokens: int,
        *,
        json_output: bool = False,
    ) -> dict:
        req_path = self.out / "requests" / f"{request_id}.json"
        if req_path.exists():
            cached = load(req_path)
            if not cached.get("error"):
                return cached

        payload = {
            "model": FLASH,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
            "thinking": {"type": "disabled"},
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}

        ceiling = min(PROVIDER[FLASH][2], sum(len(m["content"].encode("utf-8")) for m in messages) + 1024)
        reserve = token_cost(0, ceiling, max_tokens, "peak")

        for attempt in range(self.max_retries + 1):
            with self.lock:
                if self.stopped or self.spent + self.reserved + reserve > self.cap:
                    raise RuntimeError(f"Budget guard stopped before {request_id} (spent={self.spent}, cap={self.cap}).")
                self.reserved += reserve

            started = time.perf_counter()
            status, response, error = request_json(FLASH, "/chat/completions", self.key, payload)
            elapsed = (time.perf_counter() - started) * 1000
            use = usage_of(response)
            charge = (
                None
                if use is None
                else token_cost(
                    use["cache_hit_tokens"],
                    use["prompt_tokens"] - use["cache_hit_tokens"],
                    use["completion_tokens"],
                    "peak",
                )
            )
            answer, finish = None, None
            if not error:
                try:
                    first = response["choices"][0]
                    answer, finish = first["message"]["content"], first["finish_reason"]
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("empty")
                except (TypeError, KeyError, IndexError, ValueError):
                    error = "invalid_completion"

            with self.lock:
                self.reserved -= reserve
                self.spent += reserve if charge is None else charge
                if charge is not None:
                    self.usage_cost += charge
                if status in {401, 402, 403, 404}:
                    self.stopped = True

            transient = (status in {429, 500, 502, 503, 504}) or (status is None) or (error == "invalid_completion")
            if error and transient and attempt < self.max_retries and not self.stopped:
                time.sleep(1.5 * (2**attempt))
                continue

            result = {
                "id": request_id,
                "elapsed_ms": round(elapsed, 3),
                "request": payload,
                "response": response,
                "http_status": status,
                "error": error,
                "answer": answer,
                "finish": finish,
                "usage": use,
                "usage_usd": str(charge) if charge is not None else None,
            }
            dump_json(req_path, result)
            return result

        raise RuntimeError(f"Exhausted retries for {request_id}")


def load_factcon_split(split: str, limit: int | None) -> tuple[list[dict], dict[str, str]]:
    if split == "dev":
        v3_manifest = load(ROOT / "results/dev/v3-20261123T111806Z-586a10/manifest.json")
        rows, contexts, _ = load_dev()
        by_id = {r["id"]: r for r in rows}
        selected = [by_id[sid] for sid in v3_manifest["sample_ids"]]
        if limit is not None:
            selected = selected[:limit]
        return selected, contexts
    rows, contexts = read_locked("factcon", "test")
    if limit is not None:
        rows = rows[:limit]
    return rows, contexts


def compute_quote(split: str, limit: int | None, arms: tuple[str, ...]) -> dict:
    rows, contexts = load_factcon_split(split, limit)
    n = len(rows)
    # Conservative token estimates at peak pricing
    arm_estimates = {}
    total_usd = Decimal(0)
    for arm in arms:
        if arm in {"tombstone_bge_top8", "bm25_bge_recency"}:
            in_tok = n * 2100
            out_tok = n * MAX_TOKENS
            calls = n
        elif arm == "ircot_bge":
            in_tok = n * (3 * 1600 + 2100)
            out_tok = n * (3 * 96 + MAX_TOKENS)
            calls = n * 4
        elif arm == "mem0_style":
            n_lines = sum(len(records_for(contexts[cid], "numbered")[0]) for cid in {r["context_id"] for r in rows})
            w_calls = (n_lines + 15) // 16
            in_tok = w_calls * 950 + n * 1000
            out_tok = w_calls * 420 + n * MAX_TOKENS
            calls = w_calls + n
        else:
            # CARR v2 arms share 1 planner call per question
            in_tok = n * (220 + 450)
            out_tok = n * (110 + MAX_TOKENS)
            calls = n * 2
        usd = token_cost(0, in_tok, out_tok, "peak")
        arm_estimates[arm] = {"calls": calls, "est_input_tokens": in_tok, "est_output_tokens": out_tok, "est_peak_usd": str(usd)}
        total_usd += usd
    return {"split": split, "questions": n, "arms": arm_estimates, "total_est_peak_usd": str(total_usd)}


def run_experiment(
    split: str,
    limit: int | None,
    arms: tuple[str, ...],
    out: Path,
    budget: Decimal,
    key_file: Path | None,
    *,
    max_workers: int = 10,
) -> dict:
    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DEEPSEEK_API_KEY.")

    rows, contexts = load_factcon_split(split, limit)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "traces", "cache", "requests"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    baseline = LongContext(ROOT / ".venv/nltk_data")
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
        local_files_only=True, specific_model_path=str(BGE_FOLDER),
    )

    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r)

    manifest = {
        "generator": "src/run_carr_v2.py",
        "split": split,
        "questions": len(rows),
        "arms": list(arms),
        "budget_cap_usd": str(budget),
        "status": "running",
        "started_utc": utc_now().isoformat(),
    }
    dump_json(out / "manifest.json", manifest)

    mem0_write_usage: dict[str, dict] = {}
    need_planner = any(a in {"carr_v2", "carr_v2_planner_only"} for a in arms)
    planner_lock = Lock()
    embed_lock = Lock()

    class LockedEmbedder:
        def __init__(self, inner):
            self.inner = inner

        def embed(self, texts, batch_size=32):
            with embed_lock:
                return list(self.inner.embed(texts, batch_size=batch_size))

    safe_embedder = LockedEmbedder(embedder)

    # Pre-consolidate Mem0 across independent histories concurrently (strictly chronological within each history)
    mem0_stores: dict[str, object] = {}
    if "mem0_style" in arms:
        def build_mem0_ctx(cid: str) -> tuple[str, object, dict]:
            raw_text = contexts[cid]
            raw_lines = raw_text.splitlines()
            records, _, _ = archive_v2(raw_text, "numbered")
            stmts = [
                {"idx": i, "order": r.order, "text": raw_lines[r.index]}
                for i, r in enumerate(sorted(records.values(), key=lambda x: x.order))
            ]
            mstore, w_use = consolidate_mem0_history(
                stmts,
                safe_embedder,
                lambda tag, msgs, mtok, jo: meter.call(f"mem0_{cid[:8]}_{tag}", msgs, mtok, json_output=jo),
                cache_path=out / "cache" / f"mem0_factcon_{cid[:12]}.json",
                batch_size=32,
            )
            return cid, mstore, w_use

        with ThreadPoolExecutor(max_workers=max(1, len(by_ctx))) as mpool:
            for cid, mstore, w_use in mpool.map(build_mem0_ctx, sorted(by_ctx)):
                mem0_stores[cid] = mstore
                mem0_write_usage[cid] = w_use

    def get_plan(sample: dict, entities: list[str]) -> tuple[dict, dict | None]:
        sid = sample["id"]
        with planner_lock:
            call = meter.call(f"plan_v2--{sid}", planner_messages_v2(sample["question"]), 192, json_output=True)
        plan = parse_plan_v2(call, sample["question"], entities)
        return plan, call

    for cid, ctx_rows in sorted(by_ctx.items()):
        raw_text = contexts[cid]
        raw_lines = raw_text.splitlines()
        records, clean_text, decisions = archive_v2(raw_text, "numbered")
        tombstones = {d["removed_line"] for d in decisions}

        # Build shared chunk embeddings ONCE for all tombstone/hybrid/IRCoT arms
        raw_hybrid = HybridLexicalDenseIndex(embedder, raw_text, tombstones=None)
        tomb_hybrid = raw_hybrid.with_tombstones(tombstones)
        raw_dense = TombstoneDenseIndex(
            embedder, raw_text, tombstones=None, _chunks=raw_hybrid.chunks, _vectors=raw_hybrid.vectors,
        )
        tomb_dense = raw_dense.with_tombstones(tombstones)
        repack_hybrid = HybridLexicalDenseIndex(embedder, clean_text, tombstones=None) if (
            "carr_v2_planner_only" in arms
        ) else None

        active_recs = [rec for idx, rec in records.items() if idx not in tombstones]
        store_v2 = RecencyHybridStore(active_recs, raw_lines, embedder, tomb_hybrid, use_exact=True)
        store_repack = (
            RecencyHybridStore(active_recs, raw_lines, embedder, repack_hybrid, use_exact=True)
            if repack_hybrid is not None
            else None
        )

        # Retrieve pre-consolidated Mem0-style store for this context if requested
        mem0_store = mem0_stores.get(cid)

        def process_sample(sample: dict) -> None:
            sid = sample["id"]
            q = sample["question"]
            plan, plan_call = (None, None)
            if need_planner:
                plan, plan_call = get_plan(sample, store_v2.entities)

            trace_payload = {"sample_id": sid, "context_id": cid, "plan": plan, "plan_call": plan_call["id"] if plan_call else None}

            for arm in arms:
                case_path = out / "cases" / f"{arm}--{sid}.json"
                if case_path.exists():
                    continue
                extra_calls: list[dict] = []
                if arm == "tombstone_bge_top8":
                    ctx_str, chunk_ids, _ = tomb_dense.search(q)
                    arm_trace = {"chunk_ids": chunk_ids}
                elif arm == "bm25_bge_recency":
                    ctx_str, arm_trace = retrieve_bm25_recency(raw_hybrid, q)
                elif arm == "ircot_bge":
                    ctx_str, arm_trace = run_ircot(
                        raw_dense,
                        q,
                        lambda tag, msgs, mtok: meter.call(tag, msgs, mtok, json_output=False),
                        call_prefix=sid,
                    )
                    extra_calls = arm_trace.pop("reasoning_calls", [])
                elif arm == "mem0_style":
                    assert mem0_store is not None
                    ctx_str, mem_ids = mem0_store.search_context(q, top_k=8)
                    arm_trace = {"memory_ids": mem_ids}
                elif arm == "carr_v2_planner_only":
                    assert store_repack is not None and plan is not None
                    ev, arm_trace = retrieve_v2(store_repack, plan, q)
                    ctx_str = "\n".join(ev)
                elif arm == "carr_v2":
                    assert plan is not None
                    ev, arm_trace = retrieve_v2(store_v2, plan, q)
                    ctx_str = "\n".join(ev)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                msgs = baseline.messages(sample, ctx_str)
                ans_call = meter.call(f"{arm}--{sid}", msgs, MAX_TOKENS, json_output=False)
                correct = bool(not ans_call["error"] and baseline.score(ans_call["answer"], sample["answers"]))

                extra_prompt = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in extra_calls)
                extra_comp = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in extra_calls)
                extra_usd = sum(Decimal(c.get("usage_usd") or "0") for c in extra_calls)

                row = {
                    "dataset": "factcon",
                    "arm": arm,
                    "sample_id": sid,
                    "source": sample["source"],
                    "context_id": cid,
                    "question": q,
                    "reference": sample["answers"],
                    "prediction": ans_call["answer"],
                    "correct": correct,
                    "error": ans_call["error"],
                    "request_id": ans_call["id"],
                    "usage": ans_call["usage"],
                    "usage_usd": ans_call["usage_usd"],
                    "extra_calls": [c["id"] for c in extra_calls],
                    "extra_prompt_tokens": extra_prompt,
                    "extra_completion_tokens": extra_comp,
                    "extra_usd": str(extra_usd),
                    "end_to_end_ms": ans_call["elapsed_ms"],
                }
                dump_json(case_path, row)
                trace_payload[arm] = arm_trace

            dump_json(out / "traces" / f"{sid}.json", trace_payload)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(process_sample, s) for s in ctx_rows]
            for fut in as_completed(futs):
                fut.result()

    summary = summarize_run(out, rows, arms, mem0_write_usage)
    manifest["status"] = "complete"
    manifest["ended_utc"] = utc_now().isoformat()
    manifest["usage_usd"] = str(meter.usage_cost)
    dump_json(out / "manifest.json", manifest)
    return summary


def summarize_run(out: Path, rows: list[dict], arms: tuple[str, ...], mem0_write_usage: dict | None = None) -> dict:
    cases = [load(p) for p in (out / "cases").glob("*.json")]
    plan_reqs = [load(p) for p in (out / "requests").glob("plan_v2--*.json")]
    plan_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in plan_reqs)
    plan_out = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in plan_reqs)
    plan_usd = sum(Decimal(r.get("usage_usd") or "0") for r in plan_reqs)

    table_rows = []
    for arm in arms:
        arm_cases = [c for c in cases if c["arm"] == arm]
        for grp in ("all", "sh", "mh", "32k", "64k", "262k"):
            sub = [
                c for c in arm_cases
                if grp == "all" or f"_{grp}_" in c["sample_id"] or c["source"].endswith(f"_{grp}")
            ]
            if not sub:
                continue
            ans_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
            ans_out = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in sub)
            ans_usd = sum(Decimal(c.get("usage_usd") or "0") for c in sub)
            ext_in = sum(c.get("extra_prompt_tokens", 0) for c in sub)
            ext_usd = sum(Decimal(c.get("extra_usd") or "0") for c in sub)
            table_rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "correct": sum(bool(c["correct"]) for c in sub),
                "acc_pct": round(100 * sum(bool(c["correct"]) for c in sub) / len(sub), 2),
                "errors": sum(bool(c["error"]) for c in sub),
                "ans_prompt_tokens": ans_in,
                "ans_input_per_q": round(ans_in / len(sub), 1),
                "extra_prompt_tokens": ext_in,
                "ans_usd": str(ans_usd),
                "extra_usd": str(ext_usd),
                "mean_api_ms": round(mean(c["end_to_end_ms"] for c in sub), 2),
            })

    summary = {
        "generator": "src/run_carr_v2.py",
        "rows": table_rows,
        "shared_planner_v2": {
            "calls": len(plan_reqs),
            "prompt_tokens": plan_in,
            "completion_tokens": plan_out,
            "usage_usd": str(plan_usd),
        },
        "mem0_write_usage": mem0_write_usage or {},
    }
    write_csv(out / "summary.csv", table_rows)
    dump_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    ap.add_argument("--quote", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("50.0"))
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    selected_arms = tuple(args.arms)
    if args.quote or not args.execute:
        print(json.dumps(compute_quote(args.split, args.limit, selected_arms), indent=2))
    else:
        out_dir = args.run_dir or (ROOT / f"results/{args.split}/carr-v2-{utc_now():%Y%m%dT%H%M%SZ}")
        res = run_experiment(args.split, args.limit, selected_arms, out_dir, args.budget, args.key_file, max_workers=args.workers)
        print(json.dumps({
            "out_dir": str(out_dir),
            "shared_planner_v2": res["shared_planner_v2"],
            "all_rows": [r for r in res["rows"] if r["group"] in {"all", "sh", "mh"}],
        }, indent=2))
