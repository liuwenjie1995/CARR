"""Controlled same-backbone Chain-Aware Resolution (CAR-style) runner on FactConsolidation 562Q."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
import math
from pathlib import Path
from statistics import mean
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.long_context import MAX_TOKENS
from src.audit_revision_v2 import percentile
from src.baselines_v2.chain_aware_car import run_car_chain
from src.carr_v2.open_parser import archive_v2
from src.carr_v2.semantic_router import RecencyHybridStore
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.run_carr_v2 import SafeMeter
from src.run_hipporag_v2 import _load_embedder
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_transport import FLASH, credentials

RUN_FC_V2 = ROOT / "results" / "test" / "carr-v2-formal-20261124"
DEFAULT_OUT = ROOT / "results" / "test" / "car-baseline-v2"


def _exact_mcnemar(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def run_car(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    samples, contexts = read_locked("factcon", "test")
    n = len(samples)
    if not execute:
        # Avg ~2.5 hop calls + 1 answer call = ~3.5 calls/Q
        est_calls = int(n * 3.5)
        est_usd = token_cost(0, int(n * (250 * 2.5 + 270)), int(n * (45 * 2.5 + MAX_TOKENS)), "peak")
        q = {"questions": n, "est_calls": est_calls, "est_peak_usd": str(est_usd)}
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
    embedder = _load_embedder()

    by_ctx = defaultdict(list)
    for s in samples:
        by_ctx[s["context_id"]].append(s)

    for cid, ctx_rows in sorted(by_ctx.items()):
        raw_text = contexts[cid]
        raw_lines = raw_text.splitlines()
        records, _clean, decisions = archive_v2(raw_text, "numbered")
        tombstones = {d["removed_line"] for d in decisions}
        raw_hybrid = HybridLexicalDenseIndex(embedder, raw_text, tombstones=None)
        tomb_hybrid = raw_hybrid.with_tombstones(tombstones)
        active_recs = [rec for idx, rec in records.items() if idx not in tombstones]
        store = RecencyHybridStore(active_recs, raw_lines, embedder, tomb_hybrid, use_exact=True)

        def one(samp: dict) -> None:
            sid = samp["id"]
            cp = out / "cases" / f"car_style--{sid}.json"
            if cp.exists():
                return
            q = samp["question"]
            ctx_str, tr = run_car_chain(
                store,
                q,
                lambda tag, msgs, mtok, jo: meter.call(tag, msgs, mtok, json_output=jo),
                call_prefix=sid,
                max_hops=4,
            )
            hop_calls = tr.pop("hop_calls", [])
            msgs = scorer.messages(samp, ctx_str)
            ans_call = meter.call(f"car_style--{sid}", msgs, MAX_TOKENS, json_output=False)
            corr = bool(not ans_call.get("error") and scorer.score(ans_call.get("answer") or "", samp["answers"]))

            ext_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in hop_calls)
            ext_usd = sum(Decimal(str(c.get("usage_usd") or "0")) for c in hop_calls)
            ext_ms = sum(float(c.get("elapsed_ms") or 0.0) for c in hop_calls)
            e2e_ms = ext_ms + float(ans_call.get("elapsed_ms") or 0.0)

            dump_json(cp, {
                "dataset": "factcon",
                "arm": "car_style",
                "sample_id": sid,
                "source": samp["source"],
                "context_id": cid,
                "question": q,
                "reference": samp["answers"],
                "prediction": ans_call.get("answer") or "",
                "correct": corr,
                "error": ans_call.get("error") or any(c.get("error") for c in hop_calls),
                "usage": ans_call.get("usage"),
                "usage_usd": ans_call.get("usage_usd"),
                "hop_calls_count": len(hop_calls),
                "extra_prompt_tokens": ext_in,
                "extra_usd": str(ext_usd),
                "ans_ms": float(ans_call.get("elapsed_ms") or 0.0),
                "e2e_api_ms": e2e_ms,
            })
            dump_json(out / "traces" / f"{sid}.json", tr)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, s) for s in ctx_rows]
            for fut in as_completed(futs):
                fut.result()

    cases = [load(p) for p in sorted((out / "cases").glob("car_style--*.json"))]
    rows = []
    for grp in ("all", "sh", "mh", "32k", "64k", "262k"):
        sub = [c for c in cases if grp == "all" or f"_{grp}_" in c["sample_id"] or c["source"].endswith(f"_{grp}")]
        ans_in = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in sub)
        ext_in = sum(c.get("extra_prompt_tokens", 0) for c in sub)
        ans_usd = sum(Decimal(str(c.get("usage_usd") or "0")) for c in sub)
        ext_usd = sum(Decimal(str(c.get("extra_usd") or "0")) for c in sub)
        e2e_lat = [c["e2e_api_ms"] for c in sub]
        ans_lat = [c["ans_ms"] for c in sub]
        calls_per_q = mean(c["hop_calls_count"] + 1 for c in sub)
        rows.append({
            "arm": "car_style",
            "group": grp,
            "questions": len(sub),
            "correct": sum(int(bool(c["correct"])) for c in sub),
            "acc_pct": round(100.0 * sum(int(bool(c["correct"])) for c in sub) / len(sub), 2),
            "errors": sum(int(bool(c.get("error"))) for c in sub),
            "mean_llm_calls_per_q": round(calls_per_q, 2),
            "ans_input_per_q": round(ans_in / len(sub), 1),
            "total_input_per_q": round((ans_in + ext_in) / len(sub), 1),
            "ans_usd": str(ans_usd),
            "extra_usd": str(ext_usd),
            "total_usd": str(ans_usd + ext_usd),
            "ans_p50_ms": round(percentile(ans_lat, 50), 1),
            "e2e_api_mean_ms": round(mean(e2e_lat), 1),
            "e2e_api_p50_ms": round(percentile(e2e_lat, 50), 1),
            "e2e_api_p95_ms": round(percentile(e2e_lat, 95), 1),
        })

    # Paired McNemar vs CARR v2 & D core
    v2_by_sid = {s["id"]: load(RUN_FC_V2 / "cases" / f"carr_v2--{s['id']}.json") for s in samples}
    car_by_sid = {c["sample_id"]: c for c in cases}
    w = sum(1 for sid in v2_by_sid if v2_by_sid[sid]["correct"] and not car_by_sid[sid]["correct"])
    l = sum(1 for sid in v2_by_sid if not v2_by_sid[sid]["correct"] and car_by_sid[sid]["correct"])

    report = {
        "generator": "src/run_car_baseline_v2.py",
        "generated_utc": utc_now().isoformat(),
        "total_api_usd": str(meter.usage_cost),
        "rows": rows,
        "paired_carr_v2_vs_car_style": {"wins": w, "losses": l, "p": _exact_mcnemar(w, l)},
    }
    dump_json(out / "summary.json", report)
    write_csv(out / "summary.csv", rows)

    r_all = next(r for r in rows if r["group"] == "all")
    r_sh = next(r for r in rows if r["group"] == "sh")
    r_mh = next(r for r in rows if r["group"] == "mh")
    print(
        f"=== Step 2 CAR-style Summary ===\n"
        f"CAR-style All={r_all['correct']}/562 ({r_all['acc_pct']}%) | SH={r_sh['acc_pct']}% | MH={r_mh['acc_pct']}% | "
        f"Calls/Q={r_all['mean_llm_calls_per_q']} | E2E P50={r_all['e2e_api_p50_ms']}ms P95={r_all['e2e_api_p95_ms']}ms | "
        f"TotUSD=${r_all['total_usd']} | vs CARR v2={w}-{l} (p={report['paired_carr_v2_vs_car_style']['p']:.4g})"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("3.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run_car(args.execute, args.budget, args.key_file, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
