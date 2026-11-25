"""Official LongMemEval LLM-as-Judge re-evaluation over already-saved 240Q x 9 arm predictions.

Reads existing predictions on disk (zero re-retrieval, zero re-generation) and evaluates them
with the exact per-category LongMemEval Judge templates from
baselines/MemoryAgentBench/llm_based_eval/longmem_qa_evaluate.py using DeepSeek Flash.
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
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import dump_json, token_cost, utc_now
from src.run_carr_v2 import SafeMeter
from src.v2_report import write_csv
from src.v2_transport import FLASH, credentials, usage_of

RUN_LME = ROOT / "results" / "test" / "longmemeval-v2-formal-20261124" / "cases"
RUN_HIPPO_LME = ROOT / "results" / "test" / "hipporag-lme-formal-20261124" / "cases"
RUN_GLM53_CASES = ROOT / "results" / "test" / "tokenplus-glm53-v2-20261124" / "cases"
CACHE_DIR = ROOT / "results" / "analysis" / "lme_judge_cache"
OUT_JSON = ROOT / "results" / "analysis" / "longmemeval_official_judge.json"
OUT_CSV = ROOT / "results" / "analysis" / "longmemeval_official_judge.csv"

LME_ARM_SPECS = [
    ("official_full_context", RUN_LME, "official_full_context--*.json"),
    ("naive_bge_top8", RUN_LME, "naive_bge_top8--*.json"),
    ("bm25_bge_recency", RUN_LME, "bm25_bge_recency--*.json"),
    ("ircot_bge", RUN_LME, "ircot_bge--*.json"),
    ("mem0_style", RUN_LME, "mem0_style--*.json"),
    ("hipporag_ppr", RUN_HIPPO_LME, "hipporag_ppr--*.json"),
    ("hipporag_tombstone_ppr", RUN_HIPPO_LME, "hipporag_tombstone_ppr--*.json"),
    ("carr_v2", RUN_LME, "carr_v2--*.json"),
    ("carr_v2_glm53", RUN_GLM53_CASES, "longmemeval_v2_carr--*.json"),
]

CATEGORIES = (
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
)


def official_lme_judge_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Verbatim LongMemEval official per-category Judge templates from longmem_qa_evaluate.py."""
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a subset "
                "of the information required by the answer, answer no. In addition, do not penalize off-by-one "
                "errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
                "the model's response is still correct. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long as "
                "it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise ValueError(f"Unsupported LongMemEval task type: {task}")
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def parse_yes_no(text: str) -> bool:
    low = (text or "").strip().lower()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    m = re.search(r"\b(yes|no)\b", low)
    return bool(m and m.group(1) == "yes")


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def load_saved_cases() -> dict[str, list[dict]]:
    loaded: dict[str, list[dict]] = {}
    q_by_sid: dict[str, str] = {}
    for arm, folder, pattern in LME_ARM_SPECS:
        rows = []
        for p in sorted(folder.glob(pattern)):
            d = json.loads(p.read_text(encoding="utf-8"))
            if "question_type" not in d:
                continue
            if "question" in d:
                q_by_sid[d["sample_id"]] = d["question"]
            rows.append(d)
        if len(rows) != 240:
            raise ValueError(f"Expected 240 cases for {arm} in {folder}/{pattern}, got {len(rows)}")
        loaded[arm] = rows
    for arm, rows in loaded.items():
        for d in rows:
            if "question" not in d:
                d["question"] = q_by_sid[d["sample_id"]]
    return loaded


def _format_ref(ref) -> str:
    if isinstance(ref, list):
        return str(ref[0]) if len(ref) == 1 else " ; ".join(str(x) for x in ref)
    return str(ref)


def run_judge(execute: bool, budget: Decimal, key_file: Path | None, workers: int) -> dict:
    loaded = load_saved_cases()
    total_calls = sum(len(v) for v in loaded.values())
    est_usd = token_cost(0, total_calls * 260, total_calls * 4, "peak")
    if not execute:
        summary = {"arms": list(loaded.keys()), "total_calls": total_calls, "est_peak_usd": str(est_usd)}
        print(json.dumps(summary, indent=2))
        return summary

    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DeepSeek API key.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meter = SafeMeter(CACHE_DIR, key, budget)

    tasks = []
    for arm, rows in loaded.items():
        for d in rows:
            tasks.append((arm, d))

    verdicts: dict[tuple[str, str], dict] = {}

    def eval_one(arm: str, d: dict) -> tuple[str, str, dict]:
        sid = d["sample_id"]
        qtype = d["question_type"]
        ref_str = _format_ref(d["reference"])
        pred_str = str(d.get("prediction") or "")
        prompt = official_lme_judge_prompt(qtype, d["question"], ref_str, pred_str, abstention=("_abs" in sid))
        req_id = f"judge--{arm}--{sid}"
        rec = meter.call(req_id, [{"role": "user", "content": prompt}], max_tokens=16, json_output=False)
        ans = rec.get("answer") or ""
        err = rec.get("error")
        u = usage_of(rec.get("response") or {}) if not rec.get("usage") else rec["usage"]
        usd = Decimal(str(rec.get("usage_usd") or "0"))
        verdict = (not err) and parse_yes_no(ans)
        return arm, sid, {
            "sample_id": sid,
            "context_id": d["context_id"],
            "question_type": qtype,
            "judge_correct": bool(verdict),
            "judge_raw": ans,
            "subem_correct": bool(d["correct"]),
            "f1": float((d.get("metrics") or {}).get("f1", 0.0)),
            "error": err,
            "prompt_tokens": int((u or {}).get("prompt_tokens", 0)),
            "usage_usd": str(usd),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_one, arm, d) for arm, d in tasks]
        for fut in as_completed(futs):
            arm, sid, res = fut.result()
            verdicts[arm, sid] = res

    summary_rows = []
    by_arm_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    per_history: dict[str, dict] = {}
    for arm, rows in loaded.items():
        items = [verdicts[arm, d["sample_id"]] for d in rows]
        for it in items:
            by_arm_sample[arm][it["sample_id"]] = it

        groups = {
            "all": items,
            "factual_5cat": [it for it in items if it["question_type"] != "single-session-preference"],
            **{cat: [it for it in items if it["question_type"] == cat] for cat in CATEGORIES},
        }
        for grp, sub in groups.items():
            n = len(sub)
            jc = sum(int(x["judge_correct"]) for x in sub)
            sc = sum(int(x["subem_correct"]) for x in sub)
            mf1 = mean(x["f1"] for x in sub) if sub else 0.0
            j_usd = sum(Decimal(x["usage_usd"]) for x in sub)
            j_tok = sum(x["prompt_tokens"] for x in sub)
            errs = sum(int(bool(x["error"])) for x in sub)
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": n,
                "judge_correct": jc,
                "judge_pct": round(100.0 * jc / n, 2) if n else 0.0,
                "subem_correct": sc,
                "subem_pct": round(100.0 * sc / n, 2) if n else 0.0,
                "mean_f1": round(mf1, 2),
                "judge_errors": errs,
                "judge_prompt_tokens": j_tok,
                "judge_usd": str(j_usd),
            })

        # Per-history breakdown (4 independent histories)
        ctx_map: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            ctx_map[it["context_id"][:12]].append(it)
        hist_pcts = []
        hist_dict = {}
        for cid, sub in sorted(ctx_map.items()):
            jc = sum(int(x["judge_correct"]) for x in sub)
            pct = round(100.0 * jc / len(sub), 2)
            hist_pcts.append(pct)
            hist_dict[cid] = {"correct": jc, "questions": len(sub), "judge_pct": pct}
        per_history[arm] = {
            "histories": hist_dict,
            "per_history_mean_pct": round(mean(hist_pcts), 2),
            "per_history_std_pct": round(stdev(hist_pcts), 2) if len(hist_pcts) > 1 else 0.0,
            "per_history_range_pct": [min(hist_pcts), max(hist_pcts)],
        }

    # Paired McNemar comparisons on All (240Q) and Factual-5 (216Q)
    pairs_spec = [
        ("carr_v2", "official_full_context"),
        ("carr_v2", "naive_bge_top8"),
        ("carr_v2", "bm25_bge_recency"),
        ("carr_v2", "ircot_bge"),
        ("carr_v2", "mem0_style"),
        ("carr_v2", "hipporag_ppr"),
        ("carr_v2", "hipporag_tombstone_ppr"),
        ("carr_v2_glm53", "carr_v2"),
        ("carr_v2_glm53", "official_full_context"),
        ("hipporag_tombstone_ppr", "official_full_context"),
    ]
    paired = {}
    for a, b in pairs_spec:
        ma, mb = by_arm_sample[a], by_arm_sample[b]
        w_all = sum(1 for sid in ma if ma[sid]["judge_correct"] and not mb[sid]["judge_correct"])
        l_all = sum(1 for sid in ma if not ma[sid]["judge_correct"] and mb[sid]["judge_correct"])
        w_f5 = sum(1 for sid in ma if ma[sid]["question_type"] != "single-session-preference" and ma[sid]["judge_correct"] and not mb[sid]["judge_correct"])
        l_f5 = sum(1 for sid in ma if ma[sid]["question_type"] != "single-session-preference" and not ma[sid]["judge_correct"] and mb[sid]["judge_correct"])
        paired[f"{a}_vs_{b}"] = {
            "all_wins": w_all,
            "all_losses": l_all,
            "all_p": exact_mcnemar(w_all, l_all),
            "factual5_wins": w_f5,
            "factual5_losses": l_f5,
            "factual5_p": exact_mcnemar(w_f5, l_f5),
        }

    total_judge_usd = sum(Decimal(r["judge_usd"]) for r in summary_rows if r["group"] == "all")
    total_judge_errors = sum(r["judge_errors"] for r in summary_rows if r["group"] == "all")
    report = {
        "generator": "src/eval_longmemeval_judge.py",
        "judge_model": FLASH,
        "generated_utc": utc_now().isoformat(),
        "total_calls": total_calls,
        "total_errors": total_judge_errors,
        "total_judge_usd": str(total_judge_usd),
        "rows": summary_rows,
        "per_history": per_history,
        "paired_mcnemar": paired,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    dump_json(OUT_JSON, report)
    write_csv(OUT_CSV, summary_rows)

    print("=== LongMemEval Official LLM-as-Judge Summary ===")
    for r in summary_rows:
        if r["group"] == "all":
            pref = next(x for x in summary_rows if x["arm"] == r["arm"] and x["group"] == "single-session-preference")
            f5 = next(x for x in summary_rows if x["arm"] == r["arm"] and x["group"] == "factual_5cat")
            print(
                f"{r['arm']:24s} | Judge All={r['judge_correct']}/240 ({r['judge_pct']:5.2f}%) "
                f"| F5={f5['judge_correct']}/216 ({f5['judge_pct']:5.2f}%) "
                f"| Pref={pref['judge_correct']}/24 ({pref['judge_pct']:5.2f}%) "
                f"| SubEM={r['subem_correct']}/240 | Err={r['judge_errors']}"
            )
    print(f"Total Judge USD: ${total_judge_usd:.6f}, Total Errors: {total_judge_errors}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("2.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run_judge(args.execute, args.budget, args.key_file, args.workers)


if __name__ == "__main__":
    main()
