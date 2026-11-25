"""Independent Cross-Family LongMemEval LLM-as-Judge Re-Evaluation (Concern #4).

Reads already-saved 240Q predictions on disk for the 5 key systems:
  - official_full_context
  - bm25_bge_recency
  - hipporag_tombstone_ppr
  - carr_v2
  - carr_v2_glm53

Evaluates them with the exact official per-category LongMemEval Judge templates using an
independent model family (`qwen/qwen4-max` on TokenPlus) completely separate from DeepSeek
and GLM, then compares every verdict item-by-item against `lme_judge_cache` (DeepSeek Flash)
to report raw Agreement % and Cohen's kappa.
"""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_longmemeval_judge import (
    CATEGORIES,
    CACHE_DIR as DS_JUDGE_CACHE,
    _format_ref,
    exact_mcnemar,
    load_saved_cases,
    official_lme_judge_prompt,
    parse_yes_no,
)
from src.evaluator import dump_json, utc_now
from src.run_openai_compat_backbone import post_chat
from src.v2_report import write_csv

TARGET_ARMS = (
    "official_full_context",
    "bm25_bge_recency",
    "hipporag_tombstone_ppr",
    "carr_v2",
    "carr_v2_glm53",
)
ENDPOINT = "https://api.tokenplus.cn/v1/chat/completions"
JUDGE_MODEL = "qwen/qwen4-max"
CACHE_DIR = ROOT / "results" / "analysis" / "lme_judge_qwen4_cache"
OUT_JSON = ROOT / "results" / "analysis" / "longmemeval_crossfamily_judge.json"
OUT_CSV = ROOT / "results" / "analysis" / "longmemeval_crossfamily_judge.csv"


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _load_ds_verdict(arm: str, sid: str) -> bool:
    p = DS_JUDGE_CACHE / "requests" / f"judge--{arm}--{sid}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("error"):
        raise ValueError(f"Unexpected DS judge error in {p}")
    return parse_yes_no(rec.get("answer") or "")


def run_crossfamily_judge(execute: bool, endpoint: str, model: str, workers: int) -> dict:
    loaded = load_saved_cases()
    sub_loaded = {arm: loaded[arm] for arm in TARGET_ARMS}
    total_calls = sum(len(v) for v in sub_loaded.values())
    if not execute:
        est_prompt_tok = total_calls * 260
        quote = {
            "arms": list(TARGET_ARMS),
            "judge_model": model,
            "endpoint": endpoint,
            "total_calls": total_calls,
            "est_prompt_tokens": est_prompt_tok,
        }
        print(json.dumps(quote, indent=2))
        return quote

    key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENAI_COMPAT_API_KEY in environment.")

    (CACHE_DIR / "requests").mkdir(parents=True, exist_ok=True)
    tasks = [(arm, d) for arm in TARGET_ARMS for d in sub_loaded[arm]]
    verdicts: dict[tuple[str, str], dict] = {}

    def eval_one(arm: str, d: dict) -> tuple[str, str, dict]:
        sid = d["sample_id"]
        qtype = d["question_type"]
        cpath = CACHE_DIR / "requests" / f"xfjudge--{arm}--{sid}.json"
        if cpath.exists():
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            if cached.get("http_status") == 200 and not cached.get("error") and (cached.get("answer") or "").strip():
                ans = cached["answer"]
                return arm, sid, {
                    "sample_id": sid,
                    "context_id": d["context_id"],
                    "question_type": qtype,
                    "qwen_judge_correct": parse_yes_no(ans),
                    "ds_judge_correct": _load_ds_verdict(arm, sid),
                    "judge_raw": ans,
                    "error": None,
                    "usage": cached.get("usage") or {},
                }

        ref_str = _format_ref(d["reference"])
        pred_str = str(d.get("prediction") or "")
        prompt = official_lme_judge_prompt(qtype, d["question"], ref_str, pred_str, abstention=("_abs" in sid))

        status, body, elapsed_ms = None, {}, 0.0
        answer, finish, err = "", None, None
        usage = {}
        for max_tok in (512, 1024):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tok,
            }
            for attempt in range(4):
                status, body, elapsed_ms = post_chat(endpoint, key, payload, timeout=120)
                if status == 200 and isinstance(body, dict) and body.get("choices"):
                    break
                if status in (401, 402, 403):
                    break
                if attempt < 3:
                    time.sleep(2.0 * (attempt + 1))
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            msg = choice.get("message") or {}
            answer = (msg.get("content") or "").strip()
            finish = choice.get("finish_reason")
            usage = body.get("usage") if isinstance(body, dict) else {}
            if status == 200 and answer:
                err = None
                break
            err = f"http_{status}" if status != 200 else f"empty_content(finish={finish})"
            if status in (401, 402, 403):
                break

        dump_json(cpath, {
            "id": f"xfjudge--{arm}--{sid}",
            "model": model,
            "http_status": status,
            "elapsed_ms": elapsed_ms,
            "answer": answer,
            "finish_reason": finish,
            "usage": usage,
            "error": err,
        })
        return arm, sid, {
            "sample_id": sid,
            "context_id": d["context_id"],
            "question_type": qtype,
            "qwen_judge_correct": (not err) and parse_yes_no(answer),
            "ds_judge_correct": _load_ds_verdict(arm, sid),
            "judge_raw": answer,
            "error": err,
            "usage": usage or {},
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_one, arm, d) for arm, d in tasks]
        for fut in as_completed(futs):
            arm, sid, res = fut.result()
            verdicts[arm, sid] = res

    summary_rows = []
    inter_judge = {}
    by_arm_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    all_pairs: list[tuple[bool, bool]] = []

    for arm in TARGET_ARMS:
        items = [verdicts[arm, d["sample_id"]] for d in sub_loaded[arm]]
        for it in items:
            by_arm_sample[arm][it["sample_id"]] = it
        pairs = [(it["ds_judge_correct"], it["qwen_judge_correct"]) for it in items]
        all_pairs.extend(pairs)
        agree = sum(1 for x, y in pairs if x == y)
        inter_judge[arm] = {
            "questions": len(pairs),
            "ds_yes": sum(1 for x, _ in pairs if x),
            "qwen_yes": sum(1 for _, y in pairs if y),
            "both_yes": sum(1 for x, y in pairs if x and y),
            "ds_only": sum(1 for x, y in pairs if x and not y),
            "qwen_only": sum(1 for x, y in pairs if not x and y),
            "both_no": sum(1 for x, y in pairs if not x and not y),
            "agreement_pct": round(100.0 * agree / len(pairs), 2),
            "cohens_kappa": round(cohens_kappa(pairs), 4),
        }

        groups = {
            "all": items,
            "factual_5cat": [it for it in items if it["question_type"] != "single-session-preference"],
            **{cat: [it for it in items if it["question_type"] == cat] for cat in CATEGORIES},
        }
        for grp, sub in groups.items():
            n = len(sub)
            qc = sum(1 for x in sub if x["qwen_judge_correct"])
            dc = sum(1 for x in sub if x["ds_judge_correct"])
            errs = sum(1 for x in sub if x["error"])
            pt = sum(int((x.get("usage") or {}).get("prompt_tokens", 0)) for x in sub)
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": n,
                "qwen_judge_correct": qc,
                "qwen_judge_pct": round(100.0 * qc / n, 2) if n else 0.0,
                "ds_judge_correct": dc,
                "ds_judge_pct": round(100.0 * dc / n, 2) if n else 0.0,
                "errors": errs,
                "prompt_tokens": pt,
            })

    pooled_agree = sum(1 for x, y in all_pairs if x == y)
    inter_judge["pooled_1200"] = {
        "questions": len(all_pairs),
        "ds_yes": sum(1 for x, _ in all_pairs if x),
        "qwen_yes": sum(1 for _, y in all_pairs if y),
        "both_yes": sum(1 for x, y in all_pairs if x and y),
        "ds_only": sum(1 for x, y in all_pairs if x and not y),
        "qwen_only": sum(1 for x, y in all_pairs if not x and y),
        "both_no": sum(1 for x, y in all_pairs if not x and not y),
        "agreement_pct": round(100.0 * pooled_agree / len(all_pairs), 2),
        "cohens_kappa": round(cohens_kappa(all_pairs), 4),
    }

    paired = {}
    for a, b in (
        ("carr_v2", "official_full_context"),
        ("carr_v2", "bm25_bge_recency"),
        ("carr_v2", "hipporag_tombstone_ppr"),
        ("carr_v2_glm53", "carr_v2"),
        ("carr_v2_glm53", "official_full_context"),
        ("hipporag_tombstone_ppr", "official_full_context"),
    ):
        ma, mb = by_arm_sample[a], by_arm_sample[b]
        wa = sum(1 for sid in ma if ma[sid]["qwen_judge_correct"] and not mb[sid]["qwen_judge_correct"])
        la = sum(1 for sid in ma if not ma[sid]["qwen_judge_correct"] and mb[sid]["qwen_judge_correct"])
        wf = sum(1 for sid in ma if ma[sid]["question_type"] != "single-session-preference" and ma[sid]["qwen_judge_correct"] and not mb[sid]["qwen_judge_correct"])
        lf = sum(1 for sid in ma if ma[sid]["question_type"] != "single-session-preference" and not ma[sid]["qwen_judge_correct"] and mb[sid]["qwen_judge_correct"])
        paired[f"{a}_vs_{b}"] = {
            "all_wins": wa,
            "all_losses": la,
            "all_p": exact_mcnemar(wa, la),
            "factual5_wins": wf,
            "factual5_losses": lf,
            "factual5_p": exact_mcnemar(wf, lf),
        }

    total_errors = sum(r["errors"] for r in summary_rows if r["group"] == "all")
    report = {
        "generator": "src/eval_longmemeval_judge_crossfamily.py",
        "judge_model": model,
        "endpoint": endpoint,
        "generated_utc": utc_now().isoformat(),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "rows": summary_rows,
        "inter_judge_vs_deepseek": inter_judge,
        "paired_mcnemar_qwen": paired,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    dump_json(OUT_JSON, report)
    write_csv(OUT_CSV, summary_rows)

    print("=== Cross-Family Qwen4-Max Judge Summary ===")
    for arm in TARGET_ARMS:
        r_all = next(r for r in summary_rows if r["arm"] == arm and r["group"] == "all")
        r_f5 = next(r for r in summary_rows if r["arm"] == arm and r["group"] == "factual_5cat")
        ij = inter_judge[arm]
        print(
            f"{arm:24s} | Qwen All={r_all['qwen_judge_correct']}/240 ({r_all['qwen_judge_pct']:5.2f}%) "
            f"| F5={r_f5['qwen_judge_correct']}/216 ({r_f5['qwen_judge_pct']:5.2f}%) "
            f"| Agree={ij['agreement_pct']:5.2f}% kappa={ij['cohens_kappa']:.4f} | Err={r_all['errors']}"
        )
    print(f"Pooled 1,200 verdicts: Agree={inter_judge['pooled_1200']['agreement_pct']:.2f}% kappa={inter_judge['pooled_1200']['cohens_kappa']:.4f} Err={total_errors}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run_crossfamily_judge(args.execute, args.endpoint, args.model, args.workers)


if __name__ == "__main__":
    main()
