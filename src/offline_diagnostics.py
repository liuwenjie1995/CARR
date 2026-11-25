"""Reproducible offline mechanistic, counterfactual, and cross-backbone diagnostics for formal-v7.

Reads only locked inputs and archived formal/dev logs; makes zero API calls and
does not mutate any frozen source or result file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from math import comb
from pathlib import Path
import re
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import dump_json, token_cost
from src.factual_v2 import (
    AMBIGUOUS, NEGATED, Observation, norm, projection, records_for, rule_observation,
)
from src.factual_v3 import expanded_rule
from src.formal_v7 import FACT_ARMS, archive
from src.hop_retrieval import ALIASES, BRIDGES, CUES, MAX_FACTS, structured
from src.run_unified_dev import DIRECT, PINNED_FACT_PATTERNS
from src.v2_data import read_locked
from src.v2_report import load, write_csv

FORMAL_RUN = ROOT / "results/test/formal-v7-20261123T134503331644Z"
ARK_FORMAL = ROOT / "results/test/ark-backbones-formal-20261124"
ARK_DEV = ROOT / "results/dev/ark-backbones-v7-20261124T005359Z"
BAILIAN_DEV = ROOT / "results/dev/bailian-backbones-smoke-20261124T022529Z"
V3_DEV = ROOT / "results/dev/v3-20261123T111806Z-586a10"
V7_DEV = ROOT / "results/dev/unified-prompt-v7-20261123T133145804847Z-paid"
OUT_DIR = ROOT / "results/analysis/formal_v7_diagnostics"

# Hardened local schema extensions (evaluated strictly offline as post-hoc diagnostics)
ALIGNED_CUES = {**DIRECT}
HARDENED_FACT_PATTERNS = (
    *PINNED_FACT_PATTERNS,
    (r"The official language of (.+?) is (.+)\.", "official_language"),
    (r"The language of (.+?) is (.+)\.", "official_language"),
)
# Only treat negation/uncertainty tokens as disqualifying when they appear in
# lowercase predicate context rather than inside capitalized proper titles
# (e.g., "Tomorrow Never Knows").
PREDICATE_NEGATED = re.compile(r"\b(?:not|never|might|may|possibly|rumou?r(?:ed)?)\b")


def has_alias(text: str, refs: list[str]) -> bool:
    nt = norm(text)
    for ref in refs:
        nr = norm(ref)
        if nr and re.search(r"(?<!\w)" + re.escape(nr) + r"(?!\w)", nt):
            return True
    return False


def mcnemar_exact(wins: int, losses: int) -> str:
    n = wins + losses
    if n == 0:
        return "1"
    p = min(Decimal(1), Decimal(2 * sum(comb(n, k) for k in range(min(wins, losses) + 1))) / (2**n))
    return f"{p:.3g}"


def load_formal_cases():
    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted((FORMAL_RUN / "cases").glob("*.json")):
        row = load(path)
        by_arm[row["arm"]][row["sample_id"]] = row
    return by_arm


def analyze_evidence_and_latency(by_arm: dict[str, dict[str, dict]]):
    evidence_rows = []
    latency_rows = []
    sh_route_counts = Counter()

    for arm in FACT_ARMS:
        api_ms, e2e_ms, prep_ms = [], [], []
        sh_hit = mh_hit = sh_ok_hit = mh_ok_hit = sh_n = mh_n = 0
        for sid, row in sorted(by_arm[arm].items()):
            req = load(FORMAL_RUN / "requests" / f"{row['request_id']}.json")
            api_t = float(req["elapsed_ms"])
            e2e_t = float(row["end_to_end_ms"])
            api_ms.append(api_t)
            e2e_ms.append(e2e_t)
            prep_ms.append(max(0.0, e2e_t - api_t))

            content = req["request"]["messages"][1]["content"]
            ctx, _, _ = content.rpartition(row["question"])
            hit = has_alias(ctx, row["reference"])
            if "_sh_" in sid:
                sh_n += 1
                if hit:
                    sh_hit += 1
                    if row["correct"]:
                        sh_ok_hit += 1
            else:
                mh_n += 1
                if hit:
                    mh_hit += 1
                    if row["correct"]:
                        mh_ok_hit += 1

        tot_hit = sh_hit + mh_hit
        evidence_rows.append({
            "arm": arm,
            "all_recall": tot_hit,
            "all_total": sh_n + mh_n,
            "all_recall_pct": round(100 * tot_hit / (sh_n + mh_n), 2),
            "sh_recall": sh_hit,
            "sh_total": sh_n,
            "sh_recall_pct": round(100 * sh_hit / sh_n, 2),
            "sh_acc_given_hit": sh_ok_hit,
            "sh_acc_given_hit_pct": round(100 * sh_ok_hit / sh_hit, 2) if sh_hit else 0.0,
            "mh_recall": mh_hit,
            "mh_total": mh_n,
            "mh_recall_pct": round(100 * mh_hit / mh_n, 2),
            "mh_acc_given_hit": mh_ok_hit,
            "mh_acc_given_hit_pct": round(100 * mh_ok_hit / mh_hit, 2) if mh_hit else 0.0,
        })
        latency_rows.append({
            "arm": arm,
            "mean_api_ms": round(mean(api_ms), 2),
            "median_api_ms": round(median(api_ms), 2),
            "mean_e2e_ms": round(mean(e2e_ms), 2),
            "median_e2e_ms": round(median(e2e_ms), 2),
            "mean_shared_prep_ms": round(mean(prep_ms), 2),
        })

    for sid, row in sorted(by_arm["unified_a"].items()):
        if "_sh_" not in sid:
            continue
        trace = load(FORMAL_RUN / "traces" / f"{sid}.json")
        sh_route_counts[(trace["route"]["route"], trace["clean_relay"]["status"], row["correct"])] += 1

    sh_routes_summary = {
        f"{r}|{s}|{c}": n for (r, s, c), n in sorted(sh_route_counts.items())
    }
    return evidence_rows, latency_rows, sh_routes_summary


def analyze_chunk_displacement(by_arm: dict[str, dict[str, dict]], histories: dict[str, str]):
    """Quantify chunk-boundary displacement across all 562 questions."""
    clean_histories = {}
    removed_lines_by_ctx = {}
    for cid, text in histories.items():
        _, clean, decisions = archive(text, "numbered")
        clean_histories[cid] = clean
        raw_lines = text.splitlines()
        removed_lines_by_ctx[cid] = [raw_lines[d["removed_line"]] for d in decisions]

    results = {}
    for label, raw_arm, clean_arm in [
        ("bge_top8", "plain_bge_top8", "rule_archive_bge_top8"),
        ("relay", "raw_exact_latest_relay", "unified_a"),
    ]:
        raw_only_wins = []
        clean_only_wins = []
        raw_win_gold_in_clean_history = 0
        raw_win_gold_in_raw_prompt_not_clean_prompt = 0
        raw_win_gold_removed_by_archive = 0
        clean_win_gold_gained_in_prompt = 0
        prompt_diffs = 0

        for sid in sorted(by_arm[raw_arm]):
            r_raw = by_arm[raw_arm][sid]
            r_clean = by_arm[clean_arm][sid]
            req_raw = load(FORMAL_RUN / "requests" / f"{r_raw['request_id']}.json")
            req_clean = load(FORMAL_RUN / "requests" / f"{r_clean['request_id']}.json")
            c_raw = req_raw["request"]["messages"][1]["content"]
            c_clean = req_clean["request"]["messages"][1]["content"]
            if c_raw != c_clean:
                prompt_diffs += 1
            if r_raw["correct"] == r_clean["correct"]:
                continue

            ctx_raw, _, _ = c_raw.rpartition(r_raw["question"])
            ctx_clean, _, _ = c_clean.rpartition(r_clean["question"])
            in_raw = has_alias(ctx_raw, r_raw["reference"])
            in_clean = has_alias(ctx_clean, r_clean["reference"])
            in_clean_hist = has_alias(clean_histories[r_raw["context_id"]], r_raw["reference"])
            in_removed = any(has_alias(line, r_raw["reference"]) for line in removed_lines_by_ctx[r_raw["context_id"]])

            if r_raw["correct"] and not r_clean["correct"]:
                raw_only_wins.append(sid)
                if in_clean_hist:
                    raw_win_gold_in_clean_history += 1
                if in_raw and not in_clean:
                    raw_win_gold_in_raw_prompt_not_clean_prompt += 1
                if in_removed and not in_clean_hist:
                    raw_win_gold_removed_by_archive += 1
            else:
                clean_only_wins.append(sid)
                if in_clean and not in_raw:
                    clean_win_gold_gained_in_prompt += 1

        results[label] = {
            "prompt_diffs": prompt_diffs,
            "identical_prompts": 562 - prompt_diffs,
            "clean_only_wins": len(clean_only_wins),
            "raw_only_wins": len(raw_only_wins),
            "clean_win_gold_gained_in_prompt": clean_win_gold_gained_in_prompt,
            "raw_win_gold_in_clean_history": raw_win_gold_in_clean_history,
            "raw_win_gold_displaced_from_prompt": raw_win_gold_in_raw_prompt_not_clean_prompt,
            "raw_win_gold_deleted_by_archive": raw_win_gold_removed_by_archive,
        }
    return results


def analyze_recency_collisions(by_arm: dict[str, dict[str, dict]]):
    """Quantify multi-version exact-index collisions and newest-vs-oldest value selection."""
    stats = {}
    for relay_key, arm_name in [("raw_relay", "raw_exact_latest_relay"), ("clean_relay", "unified_a")]:
        total_exact_steps = 0
        multi_cand_steps = 0
        conflicting_value_steps = 0
        by_relation_conflicts = Counter()
        sh_conflict_questions = 0
        sh_newest_matches_gold = 0
        sh_oldest_matches_gold = 0
        mh_conflict_questions = 0
        mh_final_conflict = 0
        mh_final_newest_matches_gold = 0
        mh_final_oldest_matches_gold = 0

        for sid, row in sorted(by_arm[arm_name].items()):
            trace = load(FORMAL_RUN / "traces" / f"{sid}.json")
            relay = trace[relay_key]
            events = relay.get("events", [])
            q_has_conflict = False
            for ev in events:
                if ev.get("channel") != "exact_latest":
                    continue
                total_exact_steps += 1
                cands = ev.get("candidates", [])
                if len(cands) > 1:
                    multi_cand_steps += 1
                    distinct_vals = {norm(c["value"]) for c in cands}
                    if len(distinct_vals) > 1:
                        conflicting_value_steps += 1
                        by_relation_conflicts[ev["relation"]] += 1
                        q_has_conflict = True

            if q_has_conflict:
                if "_sh_" in sid:
                    sh_conflict_questions += 1
                    # For SH direct fact, there is 1 event
                    cands = events[0]["candidates"]
                    newest = max(cands, key=lambda c: c["order"])
                    oldest = min(cands, key=lambda c: c["order"])
                    if has_alias(newest["value"], row["reference"]):
                        sh_newest_matches_gold += 1
                    if has_alias(oldest["value"], row["reference"]):
                        sh_oldest_matches_gold += 1
                else:
                    mh_conflict_questions += 1
                    # Check if the final completed hop had a conflict
                    if relay.get("status") == "completed_route" and events:
                        last_ev = events[-1]
                        if last_ev.get("channel") == "exact_latest" and len({norm(c["value"]) for c in last_ev.get("candidates", [])}) > 1:
                            mh_final_conflict += 1
                            cands = last_ev["candidates"]
                            newest = max(cands, key=lambda c: c["order"])
                            oldest = min(cands, key=lambda c: c["order"])
                            if has_alias(newest["value"], row["reference"]):
                                mh_final_newest_matches_gold += 1
                            if has_alias(oldest["value"], row["reference"]):
                                mh_final_oldest_matches_gold += 1

        stats[relay_key] = {
            "total_exact_steps": total_exact_steps,
            "multi_candidate_steps": multi_cand_steps,
            "conflicting_value_steps": conflicting_value_steps,
            "top_conflicting_relations": dict(by_relation_conflicts.most_common(12)),
            "sh_conflict_questions": sh_conflict_questions,
            "sh_newest_matches_gold": sh_newest_matches_gold,
            "sh_oldest_matches_gold": sh_oldest_matches_gold,
            "mh_conflict_questions": mh_conflict_questions,
            "mh_final_hop_conflict": mh_final_conflict,
            "mh_final_newest_matches_gold": mh_final_newest_matches_gold,
            "mh_final_oldest_matches_gold": mh_final_oldest_matches_gold,
        }
    return stats


def hardened_parse_record(record):
    """Post-hoc hardened parser: title-safe negation + official_language/developer/music_type."""
    # Try original structured first
    fact = structured(record)
    if fact:
        return fact
    # Check pinned + hardened patterns
    for regex, relation in HARDENED_FACT_PATTERNS:
        if match := re.fullmatch(regex, record.text):
            entity, value = match.groups()
            return Observation(record.index, record.order, entity, relation, value, "assert",
                               (record.start, record.start + len(record.text)), "local")
    # Retry rule_observation / expanded_rule with title-safe negation if original was blocked by capitalized "Never"/"May"/etc.
    if NEGATED.search(record.text) and not PREDICATE_NEGATED.search(record.text):
        # Temporarily mask capitalized negation words so rule_observation / expanded_rule can parse
        masked = re.sub(r"\b(Not|Never|Might|May|Possibly|Rumou?r(?:ed)?)\b", lambda m: "__CAP_" + m.group(1) + "__", record.text)
        dummy = type(record)(record.index, masked, record.start, record.order)
        parsed = expanded_rule(dummy)
        if parsed:
            restore = lambda s: re.sub(r"__CAP_([A-Za-z]+)__", r"\1", s)
            return Observation(
                parsed.index, parsed.order, restore(parsed.entity), parsed.relation,
                restore(parsed.value), parsed.operation, parsed.span, "hardened_title_safe",
            )
    return None


def analyze_hardening_counterfactual(by_arm: dict[str, dict[str, dict]], histories: dict[str, str], eventqa_histories: dict[str, str]):
    """Offline replay of saved plans and single-hop/multi-hop coverage under hardened local rules."""
    # 1. Parser coverage delta across 32K, 64K, 262K
    indices_by_ctx = {}
    lines_by_ctx = {}
    coverage_delta = {}
    for cid, text in histories.items():
        recs, _ = records_for(text, "numbered")
        lines = text.splitlines()
        lines_by_ctx[cid] = {r.order: lines[r.index] for r in recs.values()}
        orig_count = 0
        hard_count = 0
        title_safe_recovered = 0
        idx = defaultdict(list)
        for r in recs.values():
            orig = structured(r)
            if not orig:
                for p, rel in PINNED_FACT_PATTERNS:
                    if m := re.fullmatch(p, r.text):
                        orig = Observation(r.index, r.order, m[1], rel, m[2], "assert", (r.start, r.start + len(r.text)), "local")
                        break
            if orig:
                orig_count += 1
            hard = hardened_parse_record(r)
            if hard:
                hard_count += 1
                if hard.origin == "hardened_title_safe":
                    title_safe_recovered += 1
                idx[hard.key].append(hard)
        indices_by_ctx[cid] = idx
        coverage_delta[cid] = {
            "numbered_facts": len(recs),
            "orig_indexed": orig_count,
            "hardened_indexed": hard_count,
            "gained_facts": hard_count - orig_count,
            "title_safe_recovered": title_safe_recovered,
        }

    # 2. Verify EventQA safety under title-safe negation
    eventqa_removed_hardened = 0
    for cid, text in eventqa_histories.items():
        recs, dup = records_for(text, "dated")
        facts = [f for r in recs.values() if (f := rule_observation(r))]
        _, dec = projection(text, facts, "dated", duplicate_ids=dup)
        eventqa_removed_hardened += len(dec)

    # 3. Replay ALL 130 saved LLM plans with ALIGNED_CUES + hardened index (exact-latest only, no new API calls!)
    replayed_plans = 0
    orig_rejected = 0
    recovered_valid_plans = 0
    recovered_completed_chains = 0
    recovered_gold_in_evidence = 0
    recovered_cases = []

    stalled_orig = 0
    stalled_recovered_gold = 0
    stalled_recovered_cases = []

    for sid, row in sorted(by_arm["unified_a"].items()):
        if "_mh_" not in sid:
            continue
        trace = load(FORMAL_RUN / "traces" / f"{sid}.json")
        if not trace.get("planner_request"):
            continue
        replayed_plans += 1
        status = trace["clean_relay"]["status"]
        plan_req = load(FORMAL_RUN / "requests" / f"{trace['planner_request']}.json")
        raw_ans = plan_req.get("answer") or ""

        # Parse with ALIGNED_CUES
        parsed_plan = None
        try:
            val = json.loads(raw_ans)
            start, rels = val["start"], val["relations"]
            if isinstance(start, str) and norm(start) and norm(start) in norm(row["question"]):
                if isinstance(rels, list) and 1 <= len(rels) <= MAX_FACTS and all(isinstance(x, str) for x in rels):
                    rels = [ALIASES.get(norm(x).replace(" ", "_"), norm(x).replace(" ", "_")) for x in rels]
                    if all(x in ALIGNED_CUES for x in rels):
                        parsed_plan = {"start": start, "relations": rels}
        except Exception:
            parsed_plan = None

        def run_exact_chain(plan_obj, cid):
            idx = indices_by_ctx[cid]
            entity = plan_obj["start"]
            queue = [(r, True) for r in plan_obj["relations"]]
            proof, visited = [], set()
            ok = True
            while queue:
                rel, allow_bridge = queue.pop(0)
                cands = idx.get((norm(entity), rel), [])
                if not cands:
                    # Also try stripping leading "the "
                    alt = norm(entity).removeprefix("the ")
                    cands = idx.get((alt, rel), []) or idx.get(("the " + alt, rel), [])
                if cands:
                    chosen = max(cands, key=lambda f: f.order)
                    if len(proof) >= MAX_FACTS or chosen.order in visited:
                        ok = False
                        break
                    visited.add(chosen.order)
                    proof.append(chosen.order)
                    entity = chosen.value
                elif allow_bridge and rel in BRIDGES and len(proof) + len(queue) + len(BRIDGES[rel]) <= MAX_FACTS:
                    queue = [(p, False) for p in BRIDGES[rel]] + queue
                else:
                    ok = False
                    break
            ev_text = "\n".join(lines_by_ctx[cid][o] for o in proof)
            return ok, proof, ev_text

        if status == "invalid_model_plan":
            orig_rejected += 1
            if parsed_plan is not None:
                recovered_valid_plans += 1
                ok, proof, ev_text = run_exact_chain(parsed_plan, row["context_id"])
                if ok:
                    recovered_completed_chains += 1
                if has_alias(ev_text, row["reference"]):
                    recovered_gold_in_evidence += 1
                    recovered_cases.append({"sample_id": sid, "plan": parsed_plan, "completed": ok, "fact_ids": proof})
        elif status == "no_matching_fact" and not row["correct"] and parsed_plan is not None:
            stalled_orig += 1
            ok, proof, ev_text = run_exact_chain(parsed_plan, row["context_id"])
            if ok and has_alias(ev_text, row["reference"]):
                stalled_recovered_gold += 1
                stalled_recovered_cases.append({"sample_id": sid, "plan": parsed_plan, "fact_ids": proof})

    return {
        "coverage_delta_by_ctx": coverage_delta,
        "eventqa_removed_lines": eventqa_removed_hardened,
        "replayed_plans": replayed_plans,
        "orig_rejected_plans": orig_rejected,
        "recovered_valid_plans": recovered_valid_plans,
        "recovered_completed_chains": recovered_completed_chains,
        "recovered_gold_in_evidence": recovered_gold_in_evidence,
        "recovered_rejected_examples": recovered_cases[:10],
        "stalled_orig_wrong": stalled_orig,
        "stalled_recovered_gold": stalled_recovered_gold,
        "stalled_recovered_examples": stalled_recovered_cases[:10],
    }


def analyze_cache_sensitivity():
    """Analytical cost comparison across full-context prefix-cache hit rates (0% to 99.34%)."""
    summary = load(FORMAL_RUN / "summary.json")
    billing = {r["arm"]: r for r in summary["billing"]}
    full_in = int(billing["official_full_context"]["prompt_tokens"])
    full_out = int(billing["official_full_context"]["completion_tokens"])
    carr_ans_usd = Decimal(billing["unified_a"]["usage_usd"])
    carr_plan_usd = Decimal(billing["shared_decomposition"]["usage_usd"])
    carr_total_usd = carr_ans_usd + carr_plan_usd

    rows = []
    for hit_pct in [0.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.3424]:
        hit_tok = round(full_in * hit_pct / 100.0)
        miss_tok = full_in - hit_tok
        full_usd = token_cost(hit_tok, miss_tok, full_out, "off_peak")
        rows.append({
            "full_cache_hit_pct": round(hit_pct, 2),
            "full_context_usd": f"{full_usd:.6f}",
            "carr_e2e_usd": f"{carr_total_usd:.6f}",
            "cost_ratio_e2e": round(float(full_usd / carr_total_usd), 2),
            "cost_ratio_ans_only": round(float(full_usd / carr_ans_usd), 2),
        })
    return rows


def analyze_cross_backbones(by_arm: dict[str, dict[str, dict]]):
    """Reconcile all completed and partial cross-backbone runs (dev + formal test)."""
    out = {"dev_25": [], "dev_bailian_5": [], "formal_test": [], "formal_32k_paired": {}}

    # 1. Dev 25-question comparison: DeepSeek Flash vs Kimi-K3.1-Pro vs GLM-5.4
    v3_sum = {r["arm"]: r for r in load(V3_DEV / "summary.json")["arms"]}
    v7_sum = {(r["arm"], r["group"]): r for r in load(V7_DEV / "summary.json")["arms"]}
    out["dev_25"].append({
        "model": "deepseek-flash",
        "arm": "official_full_context",
        "questions": v3_sum["full"]["n"],
        "correct": v3_sum["full"]["correct"],
        "prompt_tokens": v3_sum["full"]["answer_prompt_tokens"],
        "completion_tokens": v3_sum["full"]["answer_completion_tokens"],
        "length_limited": 0,
    })
    out["dev_25"].append({
        "model": "deepseek-flash",
        "arm": "unified_a",
        "questions": v7_sum["official_evidence", "all"]["questions"],
        "correct": v7_sum["official_evidence", "all"]["correct"],
        "prompt_tokens": v7_sum["official_evidence", "all"]["prompt_tokens"],
        "completion_tokens": v7_sum["official_evidence", "all"]["completion_tokens"],
        "length_limited": 0,
    })
    ark_dev_sum = load(ARK_DEV / "summary.json")
    for row in ark_dev_sum["totals"]:
        if row["type"] == "all":
            out["dev_25"].append({
                "model": row["model"],
                "arm": row["arm"],
                "questions": row["recorded"],
                "correct": row["correct"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "length_limited": row["length_limited"],
            })

    # 2. Bailian 5-question dev smoke
    bailian_sum = load(BAILIAN_DEV / "summary.json")
    for row in bailian_sum["totals"]:
        out["dev_bailian_5"].append({
            "model": row["model"],
            "arm": row["arm"],
            "questions": row["questions"],
            "correct": row["correct"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
        })

    # 3. Formal test cross-backbone replay (GLM-5.4 + Kimi-K3.1-Pro)
    ark_rows = [load(p) for p in (ARK_FORMAL / "results").glob("*.json")]
    by_ma = defaultdict(dict)
    for r in ark_rows:
        by_ma[(r["model"], r["arm"])][r["sample_id"]] = r

    for (model, arm), d in sorted(by_ma.items()):
        for stratum in ("all", "sh", "mh", "32k", "64k", "262k"):
            sub = [x for sid, x in d.items() if stratum == "all" or stratum in sid.split("_")]
            if not sub:
                continue
            answered = [x for x in sub if x["answered"]]
            out["formal_test"].append({
                "model": model,
                "arm": arm,
                "stratum": stratum,
                "recorded": len(sub),
                "correct": sum(x["correct"] for x in sub),
                "acc_pct": round(100 * sum(x["correct"] for x in sub) / len(sub), 2),
                "answered": len(answered),
                "acc_answered_pct": round(100 * sum(x["correct"] for x in answered) / len(answered), 2) if answered else 0.0,
                "length_limited": sum(x["finish"] == "length" for x in sub),
                "prompt_tokens": sum(x["usage"]["prompt_tokens"] for x in sub if x["usage"]),
                "mean_prompt_tokens": round(sum(x["usage"]["prompt_tokens"] for x in sub if x["usage"]) / len(sub), 1),
                "completion_tokens": sum(x["usage"]["completion_tokens"] for x in sub if x["usage"]),
            })

    # 4. Paired 32K complete-stratum (175Q) and all-recorded (184Q) comparison across GLM-5.4 and DeepSeek Flash
    glm_full = by_ma["glm-5.4", "official_full_context"]
    glm_carr = by_ma["glm-5.4", "unified_a"]
    ids_32k = sorted(sid for sid in (glm_full.keys() & glm_carr.keys()) if "_32k_" in sid)
    ids_paired = sorted(glm_full.keys() & glm_carr.keys())
    ds_full = by_arm["official_full_context"]
    ds_carr = by_arm["unified_a"]
    kimi_carr = by_ma["kimi-k3.1-pro", "unified_a"]

    def summarize_subset(ids, model_map, label):
        sub = [model_map[i] for i in ids if i in model_map]
        sh = [x for x in sub if "_sh_" in x["sample_id"]]
        mh = [x for x in sub if "_mh_" in x["sample_id"]]
        pt = sum(x["usage"]["prompt_tokens"] for x in sub if x["usage"])
        return {
            "label": label,
            "n": len(sub),
            "correct": sum(x["correct"] for x in sub),
            "sh_correct": sum(x["correct"] for x in sh),
            "sh_n": len(sh),
            "mh_correct": sum(x["correct"] for x in mh),
            "mh_n": len(mh),
            "mean_prompt_tokens": round(pt / len(sub), 1) if sub else 0.0,
        }

    def pair_block(ids):
        return {
            "n_paired": len(ids),
            "glm_full_only_wins": sum(glm_full[i]["correct"] and not glm_carr[i]["correct"] for i in ids),
            "glm_carr_only_wins": sum(glm_carr[i]["correct"] and not glm_full[i]["correct"] for i in ids),
            "glm_mcnemar_p": mcnemar_exact(
                sum(glm_full[i]["correct"] and not glm_carr[i]["correct"] for i in ids),
                sum(glm_carr[i]["correct"] and not glm_full[i]["correct"] for i in ids),
            ),
            "arms": [
                summarize_subset(ids, ds_full, "deepseek-flash | full"),
                summarize_subset(ids, ds_carr, "deepseek-flash | carr"),
                summarize_subset(ids, glm_full, "glm-5.4 | full"),
                summarize_subset(ids, glm_carr, "glm-5.4 | carr"),
                summarize_subset(ids, kimi_carr, "kimi-k3.1-pro | carr"),
            ],
        }

    out["formal_32k_paired"] = pair_block(ids_32k)
    out["formal_all_paired"] = pair_block(ids_paired)

    # 5. Agreement between DeepSeek Flash CARR and GLM-5.4 CARR across all 562 questions
    ids_562 = sorted(ds_carr.keys() & glm_carr.keys())
    ds_only = sum(ds_carr[i]["correct"] and not glm_carr[i]["correct"] for i in ids_562)
    glm_only = sum(glm_carr[i]["correct"] and not ds_carr[i]["correct"] for i in ids_562)
    both_ok = sum(ds_carr[i]["correct"] and glm_carr[i]["correct"] for i in ids_562)
    both_no = sum((not ds_carr[i]["correct"]) and (not glm_carr[i]["correct"]) for i in ids_562)
    out["carr_ds_vs_glm_562"] = {
        "n": len(ids_562),
        "both_correct": both_ok,
        "both_wrong": both_no,
        "ds_only": ds_only,
        "glm_only": glm_only,
        "agreement_pct": round(100 * (both_ok + both_no) / len(ids_562), 2),
        "mcnemar_p": mcnemar_exact(ds_only, glm_only),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    by_arm = load_formal_cases()
    _, fact_histories = read_locked("factcon", "test")
    _, eventqa_histories = read_locked("eventqa", "test")

    evidence_rows, latency_rows, sh_routes = analyze_evidence_and_latency(by_arm)
    displacement = analyze_chunk_displacement(by_arm, fact_histories)
    recency = analyze_recency_collisions(by_arm)
    hardening = analyze_hardening_counterfactual(by_arm, fact_histories, eventqa_histories)
    cache_sens = analyze_cache_sensitivity()
    cross_bb = analyze_cross_backbones(by_arm)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "generator": "src/offline_diagnostics.py",
        "evidence_recall": evidence_rows,
        "latency_decomposition": latency_rows,
        "sh_route_breakdown": sh_routes,
        "chunk_displacement": displacement,
        "recency_collisions": recency,
        "hardening_counterfactual": hardening,
        "cache_sensitivity": cache_sens,
        "cross_backbones": cross_bb,
    }
    dump_json(args.out / "diagnostics_summary.json", report)
    write_csv(args.out / "evidence_recall.csv", evidence_rows)
    write_csv(args.out / "latency_decomposition.csv", latency_rows)
    write_csv(args.out / "cache_sensitivity.csv", cache_sens)
    write_csv(args.out / "cross_backbone_formal.csv", cross_bb["formal_test"])
    print(json.dumps({
        "evidence_recall": evidence_rows,
        "latency_decomposition": latency_rows,
        "sh_route_breakdown": sh_routes,
        "chunk_displacement": displacement,
        "recency_collisions": recency,
        "hardening_counterfactual": {k: v for k, v in hardening.items() if not k.endswith("_examples")},
        "cache_sensitivity": cache_sens,
        "cross_backbones_summary": {
            "dev_25": cross_bb["dev_25"],
            "formal_32k_paired": cross_bb["formal_32k_paired"],
            "carr_ds_vs_glm_562": cross_bb["carr_ds_vs_glm_562"],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
