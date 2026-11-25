"""Checkpoint-resumable cross-backbone replay runner for OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MemoryAgentBench.utils.eval_other_utils import calculate_metrics
from src.evaluator import LongContext, dump_json, utc_now
from src.v2_report import load, write_csv

FC_V2_DIR = ROOT / "results/test/carr-v2-formal-20261124"
LME_V2_DIR = ROOT / "results/test/longmemeval-v2-formal-20261124"
FC_V1_DIR = ROOT / "results/test/formal-v7-20261123T134503331644Z"


def load_suite_items(suite: str, limit: int | None = None, lengths: tuple[str, ...] | None = None) -> list[dict]:
    items = []
    if suite == "factcon_v2_carr":
        base, arm = FC_V2_DIR, "carr_v2"
    elif suite == "factcon_v1_carr":
        base, arm = FC_V1_DIR, "unified_a"
    elif suite in ("factcon_full_32k", "factcon_full_64k", "factcon_full_262k", "factcon_full_all"):
        base, arm = FC_V1_DIR, "official_full_context"
    elif suite == "longmemeval_v2_carr":
        base, arm = LME_V2_DIR, "carr_v2"
    else:
        raise ValueError(f"Unsupported suite: {suite}")

    for p in sorted((base / "cases").glob(f"{arm}--*.json")):
        c = load(p)
        sid = c["sample_id"]
        if suite == "factcon_full_32k" and "_32k_" not in sid:
            continue
        if suite == "factcon_full_64k" and "_64k_" not in sid:
            continue
        if suite == "factcon_full_262k" and "_262k_" not in sid:
            continue
        if lengths and not any(f"_{ln}_" in sid for ln in lengths):
            continue
        req = load(base / "requests" / f"{c['request_id']}.json")
        items.append({
            "suite": suite,
            "sample_id": sid,
            "context_id": c.get("context_id", ""),
            "question_type": c.get("question_type", ""),
            "source": c.get("source", ""),
            "reference": c["reference"],
            "messages": req["request"]["messages"],
            "orig_correct": c["correct"],
        })
    items.sort(key=lambda x: (x["context_id"], x["sample_id"]))
    if limit is not None:
        items = items[:limit]
    return items


def post_chat(endpoint: str, key: str, payload: dict, timeout: int = 180) -> tuple[int | None, dict, float]:
    req = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body, round((time.perf_counter() - t0) * 1000.0, 2)
    except HTTPError as err:
        with err:
            raw = err.read(8192).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"error": raw}
        return err.code, body, round((time.perf_counter() - t0) * 1000.0, 2)
    except (URLError, TimeoutError, OSError, ValueError) as err:
        return None, {"error": type(err).__name__}, round((time.perf_counter() - t0) * 1000.0, 2)


def run_suite(
    endpoint: str,
    key: str,
    model: str,
    suite: str,
    run_dir: Path,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    workers: int = 8,
    limit: int | None = None,
    lengths: tuple[str, ...] | None = None,
    max_retries: int = 3,
    execute: bool = False,
) -> dict:
    items = load_suite_items(suite, limit=limit, lengths=lengths)
    print(f"Suite={suite} | model={model} | items={len(items)} | max_tokens={max_tokens}", flush=True)
    if not execute:
        est_bytes = sum(sum(len(m["content"].encode("utf-8")) for m in it["messages"]) for it in items)
        print(f"Dry run: {len(items)} requests, ~{est_bytes/4:,.0f} approx prompt tokens.", flush=True)
        return {"suite": suite, "items": len(items), "dry_run": True}

    (run_dir / "cases").mkdir(parents=True, exist_ok=True)
    scorer = LongContext(ROOT / ".venv/nltk_data")

    def process_one(item: dict) -> dict:
        sid = item["sample_id"]
        case_path = run_dir / "cases" / f"{suite}--{sid}.json"
        if case_path.exists():
            cached = load(case_path)
            if cached.get("http_status") == 200 and not cached.get("error"):
                return cached

        payload = {
            "model": model,
            "messages": item["messages"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        status, body, elapsed_ms = None, {}, 0.0
        for attempt in range(max_retries + 1):
            status, body, elapsed_ms = post_chat(endpoint, key, payload)
            if status == 200 and isinstance(body, dict) and body.get("choices"):
                break
            if status in (401, 402, 403):
                break
            if status in (429, 500, 502, 503, 504, None) and attempt < max_retries:
                time.sleep(2.5 * (attempt + 1))
                continue
            break

        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        msg = choice.get("message") or {}
        content = msg.get("content")
        answer = content.strip() if isinstance(content, str) else ""
        finish = choice.get("finish_reason")
        usage = body.get("usage") if isinstance(body, dict) else None
        has_err = (status != 200) or not bool(answer)
        err_msg = None
        if status != 200:
            err_msg = str(body.get("error", f"HTTP {status}"))
        elif not answer:
            err_msg = f"empty_content (finish={finish})"

        correct = False
        metrics = {}
        if answer:
            if suite.startswith("longmemeval"):
                m = calculate_metrics(answer, item["reference"])
                metrics = {k: m[k] for k in ("substring_exact_match", "exact_match", "f1", "rougeL_f1", "rougeL_recall")}
                correct = bool(metrics["substring_exact_match"])
            else:
                correct = bool(scorer.score(answer, item["reference"]))

        rec = {
            "suite": suite,
            "model": model,
            "served_model": body.get("model") if isinstance(body, dict) else None,
            "sample_id": sid,
            "context_id": item["context_id"],
            "question_type": item["question_type"],
            "source": item["source"],
            "http_status": status,
            "error": err_msg,
            "finish_reason": finish,
            "prediction": answer,
            "reference": item["reference"],
            "correct": correct,
            "metrics": metrics,
            "orig_correct": item["orig_correct"],
            "usage": usage,
            "elapsed_ms": elapsed_ms,
            "timestamp_utc": utc_now().isoformat(),
        }
        dump_json(case_path, rec)
        return rec

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_one, it) for it in items]
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["sample_id"])
    summary = summarize_suite(suite, model, endpoint, results)
    dump_json(run_dir / f"summary_{suite}.json", summary)
    write_csv(run_dir / f"summary_{suite}.csv", summary["rows"])
    print(json.dumps(summary["rows"], indent=2), flush=True)
    return summary


def summarize_suite(suite: str, model: str, endpoint: str, results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {"all": results}
    if suite.startswith("factcon"):
        for tag in ("sh", "mh", "32k", "64k", "262k"):
            sub = [r for r in results if f"_{tag}_" in r["sample_id"]]
            if sub:
                groups[tag] = sub
    elif suite.startswith("longmemeval"):
        groups["factual_5cat"] = [r for r in results if r.get("question_type") != "single-session-preference"]
        for qt in sorted({r.get("question_type", "") for r in results if r.get("question_type")}):
            groups[qt] = [r for r in results if r.get("question_type") == qt]

    rows = []
    for gname, sub in groups.items():
        n = len(sub)
        corr = sum(1 for r in sub if r["correct"])
        errs = sum(1 for r in sub if r.get("error"))
        len_lim = sum(1 for r in sub if r.get("finish_reason") == "length")
        pt = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in sub)
        ct = sum((r["usage"] or {}).get("completion_tokens", 0) for r in sub)
        ch = sum(((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) for r in sub)
        rt = sum(((r["usage"] or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for r in sub)
        row = {
            "suite": suite,
            "model": model,
            "group": gname,
            "questions": n,
            "correct": corr,
            "acc_pct": round(100.0 * corr / n, 2) if n else 0.0,
            "errors": errs,
            "length_limited": len_lim,
            "prompt_tokens": pt,
            "mean_prompt_tokens": round(pt / n, 1) if n else 0.0,
            "completion_tokens": ct,
            "cached_tokens": ch,
            "reasoning_tokens": rt,
            "mean_api_ms": round(mean(r["elapsed_ms"] for r in sub), 2) if sub else 0.0,
        }
        if suite.startswith("longmemeval"):
            f1_vals = [float((r.get("metrics") or {}).get("f1", 0.0)) for r in sub]
            rl_vals = [float((r.get("metrics") or {}).get("rougeL_f1", 0.0)) for r in sub]
            row["mean_f1"] = round(100.0 * sum(f1_vals) / n, 2) if n else 0.0
            row["mean_rougeL_f1"] = round(100.0 * sum(rl_vals) / n, 2) if n else 0.0
        rows.append(row)

    return {
        "generator": "src/run_openai_compat_backbone.py",
        "endpoint": endpoint,
        "model": model,
        "suite": suite,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="https://api.tokenplus.cn/v1/chat/completions")
    ap.add_argument("--model", default="z-ai/glm-5.3")
    ap.add_argument("--suite", required=True, choices=(
        "factcon_v2_carr", "factcon_v1_carr", "factcon_full_32k",
        "factcon_full_64k", "factcon_full_262k", "factcon_full_all", "longmemeval_v2_carr",
    ))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lengths", nargs="+", choices=("32k", "64k", "262k"))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    if args.execute and not key:
        raise SystemExit("Set OPENAI_COMPAT_API_KEY in environment.")
    run_suite(
        args.endpoint, key, args.model, args.suite, args.run_dir,
        max_tokens=args.max_tokens, temperature=args.temperature,
        workers=args.workers, limit=args.limit,
        lengths=tuple(args.lengths) if args.lengths else None,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
