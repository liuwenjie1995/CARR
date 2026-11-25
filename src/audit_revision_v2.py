"""Offline deep audit for Route A Major Revision (zero API calls).

Produces results/analysis/revision_offline_audit.json covering:
1. Full-chain Recall, Planning Accuracy, Entity Linking, Temporal Validity, Per-hop & Channel Attribution.
2. Parsing Coverage vs. Independent Extraction Precision & Conflict Resolution Accuracy on FactConsolidation.
3. Per-History Breakdown (3 FactConsolidation histories + 4 LongMemEval-S* test histories) & split-lock checks.
4. End-to-End Latency (P50 / P95 / Mean) combining logged API elapsed_ms with measured local indexing & retrieval ms.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.carr_v2.open_parser import archive_v2, parse_numbered_v2
from src.carr_v2.semantic_router import _entity_variants
from src.carr_v2.tombstone_index import tokenize_lexical
from src.evaluator import dump_json
from src.factual_v2 import norm
from src.v2_data import read_locked

RUN_FC_V2 = ROOT / "results" / "test" / "carr-v2-formal-20261124"
RUN_FC_V1 = ROOT / "results" / "test" / "formal-v7-20261123T180402Z"
RUN_LME_V2 = ROOT / "results" / "test" / "longmemeval-v2-formal-20261124"
RUN_HIPPO_FC = ROOT / "results" / "test" / "hipporag-fc-formal-20261124"
RUN_HIPPO_LME = ROOT / "results" / "test" / "hipporag-lme-formal-20261124"
OUT_PATH = ROOT / "results" / "analysis" / "revision_offline_audit.json"


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


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


def _find_v1_run() -> Path | None:
    if RUN_FC_V1.exists():
        return RUN_FC_V1
    for p in sorted((ROOT / "results" / "test").glob("formal-v7*")):
        if (p / "requests").exists():
            return p
    return None


def audit_mechanistic_and_extraction() -> tuple[dict, dict, dict]:
    samples, contexts = read_locked("factcon", "test")
    sample_by_id = {s["id"]: s for s in samples}

    parsed_by_ctx: dict[str, dict] = {}
    local_build_profile: dict[str, dict] = {}
    extraction_audit_by_ctx: dict[str, dict] = {}

    for cid, text in contexts.items():
        lens = {s["source"].split("_")[-1] for s in samples if s["context_id"] == cid}
        label = sorted(lens)[0] if lens else cid[:8]

        tracemalloc.start()
        t0 = time.perf_counter()
        records, _clean_text, decisions = archive_v2(text, "numbered")
        t_parse = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        rec_by_order = {r.order: r for r in records.values()}
        by_key: dict[tuple[str, str], list] = defaultdict(list)
        by_ent: dict[str, list] = defaultdict(list)
        facts_by_order: dict[int, object] = {}
        for rec in records.values():
            f = parse_numbered_v2(rec)
            if f is not None:
                by_key[f.key].append(f)
                by_ent[norm(f.entity)].append(f)
                facts_by_order[f.order] = f
        t_index = (time.perf_counter() - t1) * 1000.0

        lines = [rec.text for rec in sorted(records.values(), key=lambda r: r.order)]
        chunks = ["\n".join(lines[i:i + 220]) for i in range(0, len(lines), 220)]
        t2 = time.perf_counter()
        _ = [tokenize_lexical(ch) for ch in chunks]
        t_bm25_tok = (time.perf_counter() - t2) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        parsed_by_ctx[cid] = {
            "label": label,
            "rec_by_order": rec_by_order,
            "by_key": by_key,
            "by_ent": by_ent,
            "facts_by_order": facts_by_order,
            "decisions": decisions,
        }
        local_build_profile[label] = {
            "context_id": cid,
            "numbered_lines": len(records),
            "parsed_facts": len(facts_by_order),
            "chunks_220": len(chunks),
            "archive_v2_ms": round(t_parse, 2),
            "kv_index_ms": round(t_index, 2),
            "bm25_tokenize_ms": round(t_bm25_tok, 2),
            "peak_tracemalloc_mb": round(peak_bytes / (1024 * 1024), 2),
        }

        valid_roundtrip = 0
        invalid_examples = []
        for order, f in facts_by_order.items():
            raw = rec_by_order[order].text
            ent_ok = bool(f.entity.strip()) and f.entity.strip().lower() in raw.lower()
            val_ok = bool(f.value.strip()) and f.value.strip().lower() in raw.lower()
            dist_ok = norm(f.entity) != norm(f.value)
            if ent_ok and val_ok and dist_ok:
                valid_roundtrip += 1
            elif len(invalid_examples) < 5:
                invalid_examples.append({
                    "order": order,
                    "raw": raw,
                    "entity": f.entity,
                    "rel": f.relation,
                    "val": f.value,
                })

        conflict_keys_total = 0
        conflict_keys_slots = 0
        conflict_slots_tombstoned = 0
        removed_ids = {d["removed_line"] for d in decisions}
        single_value_false_removals = 0
        from src.factual_v2 import SINGLE_SLOTS
        for key, flist in by_key.items():
            distinct_vals = {norm(f.value) for f in flist if f.operation == "assert"}
            if len(distinct_vals) >= 2:
                conflict_keys_total += 1
                if key[1] in SINGLE_SLOTS:
                    conflict_keys_slots += 1
                max_order = max(f.order for f in flist)
                winner_rec_idx = facts_by_order[max_order].index
                winner_kept = winner_rec_idx not in removed_ids
                older_conflicting = [
                    f for f in flist
                    if f.order < max_order and norm(f.value) != norm(facts_by_order[max_order].value)
                ]
                all_older_archived = all(f.index in removed_ids for f in older_conflicting)
                if key[1] in SINGLE_SLOTS and winner_kept and all_older_archived:
                    conflict_slots_tombstoned += 1
            elif len(distinct_vals) == 1 and len(flist) == 1:
                if flist[0].index in removed_ids:
                    single_value_false_removals += 1

        extraction_audit_by_ctx[label] = {
            "numbered_lines": len(records),
            "parsed_facts": len(facts_by_order),
            "parsing_coverage_pct": round(100.0 * len(facts_by_order) / max(1, len(records)), 2),
            "verbatim_roundtrip_valid": valid_roundtrip,
            "verbatim_roundtrip_precision_pct": round(100.0 * valid_roundtrip / max(1, len(facts_by_order)), 2),
            "conflict_keys_with_distinct_values": conflict_keys_total,
            "conflict_keys_in_single_slots": conflict_keys_slots,
            "conflict_slots_tombstoned": conflict_slots_tombstoned,
            "conflict_tombstone_rate_pct": round(100.0 * conflict_slots_tombstoned / max(1, conflict_keys_slots), 2),
            "single_value_false_removals": single_value_false_removals,
            "archived_lines_total": len(removed_ids),
            "invalid_examples": invalid_examples,
        }

    cases_v2 = {}
    for p in (RUN_FC_V2 / "cases").glob("carr_v2--*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        cases_v2[d["sample_id"]] = d

    cases_po = {}
    for p in (RUN_FC_V2 / "cases").glob("carr_v2_planner_only--*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        cases_po[d["sample_id"]] = d

    def _eval_route(route_dict: dict, ctx_info: dict, samp: dict) -> tuple[str, bool, bool, bool, bool]:
        evs = route_dict.get("events") or []
        status = route_dict.get("status") or ""
        fact_ids = route_dict.get("fact_ids") or []
        last_val = route_dict.get("last_value") or ""
        executed_hops = [e for e in evs if e.get("chosen") is not None]
        all_hops_latest = len(executed_hops) > 0
        connected = len(executed_hops) > 0
        prev_val = None
        for ev in executed_hops:
            chosen_ord = ev["chosen"]
            fact_obj = ctx_info["facts_by_order"].get(chosen_ord)
            if fact_obj is not None:
                all_cands = ctx_info["by_key"].get(fact_obj.key, [])
                if not (all_cands and chosen_ord == max(f.order for f in all_cands)):
                    all_hops_latest = False
                if prev_val is not None:
                    if not (set(_entity_variants(fact_obj.entity)) & set(_entity_variants(prev_val))):
                        connected = False
                prev_val = fact_obj.value
            else:
                all_hops_latest = False
                connected = False
        final_val_matches = _matches_ref(last_val, samp["answers"])
        full_chain_hit = (status == "completed_route") and all_hops_latest and connected and final_val_matches
        retrieved_text = "\n".join(
            f"{fid}. {ctx_info['rec_by_order'][fid].text}"
            for fid in fact_ids if fid in ctx_info["rec_by_order"]
        )
        gold_alias_hit = _matches_ref(retrieved_text, samp["answers"])
        return status, all_hops_latest, connected, full_chain_hit, gold_alias_hit

    mech = {"sh": defaultdict(int), "mh": defaultdict(int), "all": defaultdict(int)}
    mech_po = {"sh": defaultdict(int), "mh": defaultdict(int), "all": defaultdict(int)}
    per_hop_mh = defaultdict(lambda: defaultdict(int))
    chan_profile_counts = Counter()
    chan_profile_correct = Counter()
    chan_profile_mh_counts = Counter()
    chan_profile_mh_correct = Counter()
    discordant_po_vs_v2 = []
    local_query_ms: list[float] = []

    for p in sorted((RUN_FC_V2 / "traces").glob("*.json")):
        tr = json.loads(p.read_text(encoding="utf-8"))
        sid = tr["sample_id"]
        samp = sample_by_id[sid]
        cid = samp["context_id"]
        ctx_info = parsed_by_ctx[cid]
        hop_group = "mh" if "_mh_" in sid else "sh"
        plan = tr.get("plan") or {}
        cv2 = tr.get("carr_v2") or {}
        ok = bool(cases_v2[sid]["correct"])
        ok_po = bool(cases_po[sid]["correct"])
        if ok != ok_po:
            discordant_po_vs_v2.append({
                "sample_id": sid,
                "carr_v2_correct": ok,
                "planner_only_correct": ok_po,
                "v2_status": cv2.get("status"),
                "po_status": (tr.get("carr_v2_planner_only") or {}).get("status"),
            })

        start_ent = plan.get("start") or ""
        steps = plan.get("steps") or []
        mode = plan.get("mode") or "none"

        tq0 = time.perf_counter()
        cur_e = start_ent
        for st in steps:
            rel = st.get("relation", "")
            found = None
            for evar in _entity_variants(cur_e):
                cands = ctx_info["by_key"].get((evar, rel), [])
                if cands:
                    found = max(cands, key=lambda f: f.order)
                    break
            if not found:
                break
            cur_e = found.value
        local_query_ms.append((time.perf_counter() - tq0) * 1000.0)

        start_linked = False
        if start_ent:
            for evar in _entity_variants(start_ent):
                if ctx_info["by_ent"].get(evar):
                    start_linked = True
                    break

        mode_match = (hop_group == "sh" and mode == "single") or (hop_group == "mh" and mode == "multi")

        evs = cv2.get("events") or []
        status = cv2.get("status")
        fact_ids = cv2.get("fact_ids") or []
        last_val = cv2.get("last_value") or ""

        executed_hops = [e for e in evs if e.get("chosen") is not None]
        all_hops_latest = len(executed_hops) > 0
        connected = len(executed_hops) > 0
        prev_val = None
        for idx_h, ev in enumerate(executed_hops, 1):
            chosen_ord = ev["chosen"]
            fact_obj = ctx_info["facts_by_order"].get(chosen_ord)
            if fact_obj is not None:
                all_cands = ctx_info["by_key"].get(fact_obj.key, [])
                is_latest = bool(all_cands) and chosen_ord == max(f.order for f in all_cands)
                if not is_latest:
                    all_hops_latest = False
                if prev_val is not None:
                    ent_vars = set(_entity_variants(fact_obj.entity))
                    prev_vars = set(_entity_variants(prev_val))
                    if not (ent_vars & prev_vars):
                        connected = False
                prev_val = fact_obj.value
            else:
                all_hops_latest = False
                connected = False

            if hop_group == "mh":
                per_hop_mh[f"hop_{idx_h}"]["attempted"] += 1
                per_hop_mh[f"hop_{idx_h}"]["resolved"] += 1
                if fact_obj is not None and ctx_info["by_key"].get(fact_obj.key):
                    if chosen_ord == max(f.order for f in ctx_info["by_key"][fact_obj.key]):
                        per_hop_mh[f"hop_{idx_h}"]["latest_valid"] += 1

        if hop_group == "mh":
            for idx_e, ev in enumerate(evs):
                if ev.get("chosen") is None and not ev.get("bridge"):
                    hop_slot = len([x for x in evs[:idx_e] if x.get("chosen") is not None]) + 1
                    per_hop_mh[f"hop_{hop_slot}"]["failed_stops"] += 1

        final_val_matches = _matches_ref(last_val, samp["answers"])
        full_chain_hit = (status == "completed_route") and all_hops_latest and connected and final_val_matches

        retrieved_text = "\n".join(
            f"{fid}. {ctx_info['rec_by_order'][fid].text}"
            for fid in fact_ids if fid in ctx_info["rec_by_order"]
        )
        gold_alias_hit = _matches_ref(retrieved_text, samp["answers"])

        chans = tuple(sorted({e.get("channel") for e in evs if e.get("channel")}))
        chan_key = "+".join(chans) if chans else "none"
        chan_profile_counts[chan_key] += 1
        chan_profile_correct[chan_key] += int(ok)
        if hop_group == "mh":
            chan_profile_mh_counts[chan_key] += 1
            chan_profile_mh_correct[chan_key] += int(ok)

        po = tr.get("carr_v2_planner_only") or {}
        po_status, po_latest, po_conn, po_full_chain, po_alias = _eval_route(po, ctx_info, samp)
        for grp in (hop_group, "all"):
            m = mech[grp]
            m["total"] += 1
            m["correct"] += int(ok)
            m["entity_linked"] += int(start_linked)
            m["plan_mode_match"] += int(mode_match)
            m["completed_route"] += int(status == "completed_route")
            m["all_hops_latest"] += int(all_hops_latest)
            m["connected_chain"] += int(connected)
            m["gold_alias_hit"] += int(gold_alias_hit)
            m["full_chain_hit"] += int(full_chain_hit)
            m["correct_given_full_chain"] += int(ok and full_chain_hit)
            m["correct_given_gold_alias"] += int(ok and gold_alias_hit)
            m["alias_hit_without_full_chain"] += int(gold_alias_hit and not full_chain_hit)

            mp = mech_po[grp]
            mp["total"] += 1
            mp["correct"] += int(ok_po)
            mp["completed_route"] += int(po_status == "completed_route")
            mp["all_hops_latest"] += int(po_latest)
            mp["connected_chain"] += int(po_conn)
            mp["gold_alias_hit"] += int(po_alias)
            mp["full_chain_hit"] += int(po_full_chain)
            mp["correct_given_full_chain"] += int(ok_po and po_full_chain)
            mp["correct_given_gold_alias"] += int(ok_po and po_alias)

    mech_summary = {}
    for grp, m in mech.items():
        tot = m["total"]
        mech_summary[grp] = {
            **dict(m),
            "accuracy_pct": round(100.0 * m["correct"] / tot, 2),
            "entity_linking_pct": round(100.0 * m["entity_linked"] / tot, 2),
            "plan_mode_match_pct": round(100.0 * m["plan_mode_match"] / tot, 2),
            "completed_route_pct": round(100.0 * m["completed_route"] / tot, 2),
            "temporal_validity_all_hops_pct": round(100.0 * m["all_hops_latest"] / tot, 2),
            "gold_alias_recall_pct": round(100.0 * m["gold_alias_hit"] / tot, 2),
            "full_chain_recall_pct": round(100.0 * m["full_chain_hit"] / tot, 2),
            "acc_given_full_chain_pct": round(100.0 * m["correct_given_full_chain"] / max(1, m["full_chain_hit"]), 2),
            "acc_given_gold_alias_pct": round(100.0 * m["correct_given_gold_alias"] / max(1, m["gold_alias_hit"]), 2),
        }

    channel_breakdown = {
        "all": {
            k: {"total": v, "correct": chan_profile_correct[k], "acc_pct": round(100.0 * chan_profile_correct[k] / v, 2)}
            for k, v in chan_profile_counts.most_common()
        },
        "mh": {
            k: {"total": v, "correct": chan_profile_mh_correct[k], "acc_pct": round(100.0 * chan_profile_mh_correct[k] / v, 2)}
            for k, v in chan_profile_mh_counts.most_common()
        },
        "discordant_planner_only_vs_carr_v2": discordant_po_vs_v2,
    }

    local_timing = {
        "build_by_history": local_build_profile,
        "local_exact_relay_ms": {
            "mean": round(mean(local_query_ms), 4),
            "p50": round(percentile(local_query_ms, 50), 4),
            "p95": round(percentile(local_query_ms, 95), 4),
            "max": round(max(local_query_ms), 4),
        },
    }

    mech_po_summary = {}
    for grp, mp in mech_po.items():
        tot = mp["total"]
        mech_po_summary[grp] = {
            **dict(mp),
            "accuracy_pct": round(100.0 * mp["correct"] / tot, 2),
            "completed_route_pct": round(100.0 * mp["completed_route"] / tot, 2),
            "gold_alias_recall_pct": round(100.0 * mp["gold_alias_hit"] / tot, 2),
            "full_chain_recall_pct": round(100.0 * mp["full_chain_hit"] / tot, 2),
        }

    return {
        "mechanistic": mech_summary,
        "planner_only_mechanistic": mech_po_summary,
        "per_hop_mh": {k: dict(v) for k, v in sorted(per_hop_mh.items())},
        "channel_breakdown": channel_breakdown,
    }, extraction_audit_by_ctx, local_timing


def audit_per_history_and_splits() -> dict:
    splits = {}
    for sl in sorted((ROOT / "data").rglob("split-lock.json")):
        d = json.loads(sl.read_text(encoding="utf-8"))
        splits[str(sl.relative_to(ROOT)).replace("\\", "/")] = {
            "cross_split_intersections": d.get("cross_split_intersections"),
            "test_locked": d.get("test_locked"),
            "partitions": d.get("partitions"),
        }

    fc_v2_sum = json.loads((RUN_FC_V2 / "summary.json").read_text(encoding="utf-8"))
    fc_per_hist: dict[str, dict] = defaultdict(dict)
    for r in fc_v2_sum["rows"]:
        if r["group"] in ("32k", "64k", "262k"):
            fc_per_hist[r["arm"]][r["group"]] = {
                "correct": r["correct"],
                "questions": r["questions"],
                "acc_pct": round(100.0 * r["correct"] / r["questions"], 2),
            }
    hip_fc_aud = json.loads((ROOT / "results" / "analysis" / "v2_audit_hipporag_fc.json").read_text(encoding="utf-8"))
    for r in hip_fc_aud["summary"]:
        if r["group"] in ("32k", "64k", "262k"):
            fc_per_hist[r["arm"]][r["group"]] = {
                "correct": r["correct"],
                "questions": r["questions"],
                "acc_pct": round(100.0 * r["correct"] / r["questions"], 2),
            }
    for _arm, hmap in fc_per_hist.items():
        vals = [hmap[g]["acc_pct"] for g in ("32k", "64k", "262k") if g in hmap]
        if len(vals) == 3:
            hmap["per_history_mean_pct"] = round(mean(vals), 2)
            hmap["per_history_std_pct"] = round(stdev(vals), 2)
            hmap["per_history_range_pct"] = [min(vals), max(vals)]

    lme_per_hist: dict[str, dict] = {}
    arm_dirs = [
        ("official_full_context", RUN_LME_V2 / "cases", "official_full_context--*.json"),
        ("naive_bge_top8", RUN_LME_V2 / "cases", "naive_bge_top8--*.json"),
        ("bm25_bge_recency", RUN_LME_V2 / "cases", "bm25_bge_recency--*.json"),
        ("ircot_bge", RUN_LME_V2 / "cases", "ircot_bge--*.json"),
        ("mem0_style", RUN_LME_V2 / "cases", "mem0_style--*.json"),
        ("carr_v2", RUN_LME_V2 / "cases", "carr_v2--*.json"),
        ("hipporag_ppr", RUN_HIPPO_LME / "cases", "hipporag_ppr--*.json"),
        ("hipporag_tombstone_ppr", RUN_HIPPO_LME / "cases", "hipporag_tombstone_ppr--*.json"),
    ]
    for arm, cdir, pat in arm_dirs:
        by_ctx = defaultdict(lambda: {"correct": 0, "questions": 0, "f1_sum": 0.0})
        for p in sorted(cdir.glob(pat)):
            d = json.loads(p.read_text(encoding="utf-8"))
            cid = d["context_id"][:12]
            by_ctx[cid]["questions"] += 1
            by_ctx[cid]["correct"] += int(bool(d["correct"]))
            by_ctx[cid]["f1_sum"] += float((d.get("metrics") or {}).get("f1", 0.0))
        out_c = {}
        pcts = []
        for cid, st in sorted(by_ctx.items()):
            pct = round(100.0 * st["correct"] / st["questions"], 2)
            pcts.append(pct)
            out_c[cid] = {
                "correct": st["correct"],
                "questions": st["questions"],
                "subem_pct": pct,
                "mean_f1": round(st["f1_sum"] / st["questions"], 2),
            }
        lme_per_hist[arm] = {
            "histories": out_c,
            "per_history_mean_subem_pct": round(mean(pcts), 2) if pcts else 0.0,
            "per_history_std_subem_pct": round(stdev(pcts), 2) if len(pcts) > 1 else 0.0,
            "per_history_range_subem_pct": [min(pcts), max(pcts)] if pcts else [],
        }

    return {
        "split_locks": splits,
        "factconsolidation_per_history": dict(fc_per_hist),
        "longmemeval_per_history": lme_per_hist,
    }


def audit_end_to_end_latency() -> dict:
    plan_lat: dict[str, float] = {}
    for p in (RUN_FC_V2 / "requests").glob("plan_v2--*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        sid = p.stem.split("--", 1)[1]
        if d.get("elapsed_ms") is not None:
            plan_lat[sid] = float(d["elapsed_ms"])

    ircot_extra_lat: dict[str, float] = defaultdict(float)
    for p in (RUN_FC_V2 / "requests").glob("ircot*--*.json"):
        if p.name.startswith("ircot_bge--"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        parts = p.stem.split("--")
        if len(parts) >= 2 and d.get("elapsed_ms") is not None:
            ircot_extra_lat[parts[1]] += float(d["elapsed_ms"])

    fc_latency: dict[str, dict] = {}
    for arm in ("tombstone_bge_top8", "bm25_bge_recency", "ircot_bge", "mem0_style", "carr_v2_planner_only", "carr_v2"):
        ans_ms = []
        e2e_ms = []
        for p in sorted((RUN_FC_V2 / "requests").glob(f"{arm}--*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            sid = p.stem.split("--", 1)[1]
            a_ms = float(d.get("elapsed_ms") or 0.0)
            ans_ms.append(a_ms)
            if arm in ("carr_v2", "carr_v2_planner_only"):
                e2e_ms.append(a_ms + plan_lat.get(sid, 0.0))
            elif arm == "ircot_bge":
                e2e_ms.append(a_ms + ircot_extra_lat.get(sid, 0.0))
            else:
                e2e_ms.append(a_ms)
        fc_latency[arm] = {
            "n": len(ans_ms),
            "answer_mean_ms": round(mean(ans_ms), 1) if ans_ms else 0.0,
            "answer_p50_ms": round(percentile(ans_ms, 50), 1),
            "answer_p95_ms": round(percentile(ans_ms, 95), 1),
            "e2e_api_mean_ms": round(mean(e2e_ms), 1) if e2e_ms else 0.0,
            "e2e_api_p50_ms": round(percentile(e2e_ms, 50), 1),
            "e2e_api_p95_ms": round(percentile(e2e_ms, 95), 1),
        }

    v1_dir = _find_v1_run()
    if v1_dir and (v1_dir / "requests").exists():
        full_ms = [
            float(json.loads(p.read_text(encoding="utf-8")).get("elapsed_ms") or 0.0)
            for p in sorted((v1_dir / "requests").glob("official_full_context--*.json"))
        ]
        if full_ms:
            fc_latency["official_full_context_v1"] = {
                "n": len(full_ms),
                "answer_mean_ms": round(mean(full_ms), 1),
                "answer_p50_ms": round(percentile(full_ms, 50), 1),
                "answer_p95_ms": round(percentile(full_ms, 95), 1),
                "e2e_api_mean_ms": round(mean(full_ms), 1),
                "e2e_api_p50_ms": round(percentile(full_ms, 50), 1),
                "e2e_api_p95_ms": round(percentile(full_ms, 95), 1),
            }

    if (RUN_HIPPO_FC / "requests").exists():
        for arm in ("hipporag_ppr", "hipporag_tombstone_ppr"):
            ms = [
                float(json.loads(p.read_text(encoding="utf-8")).get("elapsed_ms") or 0.0)
                for p in sorted((RUN_HIPPO_FC / "requests").glob(f"{arm}--*.json"))
            ]
            if ms:
                fc_latency[arm] = {
                    "n": len(ms),
                    "answer_mean_ms": round(mean(ms), 1),
                    "answer_p50_ms": round(percentile(ms, 50), 1),
                    "answer_p95_ms": round(percentile(ms, 95), 1),
                    "e2e_api_mean_ms": round(mean(ms), 1),
                    "e2e_api_p50_ms": round(percentile(ms, 50), 1),
                    "e2e_api_p95_ms": round(percentile(ms, 95), 1),
                }

    lme_latency: dict[str, dict] = {}
    for arm in ("official_full_context", "naive_bge_top8", "bm25_bge_recency", "ircot_bge", "mem0_style", "carr_v2"):
        ms = [
            float(json.loads(p.read_text(encoding="utf-8")).get("elapsed_ms") or 0.0)
            for p in sorted((RUN_LME_V2 / "requests").glob(f"{arm}--*.json"))
        ]
        if ms:
            lme_latency[arm] = {
                "n": len(ms),
                "answer_mean_ms": round(mean(ms), 1),
                "answer_p50_ms": round(percentile(ms, 50), 1),
                "answer_p95_ms": round(percentile(ms, 95), 1),
            }

    return {
        "planner_v2_ms": {
            "n": len(plan_lat),
            "mean": round(mean(plan_lat.values()), 1) if plan_lat else 0.0,
            "p50": round(percentile(list(plan_lat.values()), 50), 1),
            "p95": round(percentile(list(plan_lat.values()), 95), 1),
        },
        "factconsolidation": fc_latency,
        "longmemeval": lme_latency,
    }


def main() -> None:
    mech_block, ext_audit, local_timing = audit_mechanistic_and_extraction()
    per_hist = audit_per_history_and_splits()
    lat = audit_end_to_end_latency()

    report = {
        "generator": "src/audit_revision_v2.py",
        "mechanistic_full_chain": mech_block,
        "extraction_and_conflict_audit": ext_audit,
        "local_compute_profile": local_timing,
        "per_history_and_splits": per_hist,
        "end_to_end_latency": lat,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump_json(OUT_PATH, report)

    print("=== Phase 1 Offline Audit Summary ===")
    for g in ("all", "sh", "mh"):
        m = mech_block["mechanistic"][g]
        print(
            f"[{g.upper()}] Acc={m['correct']}/{m['total']} ({m['accuracy_pct']}%) | "
            f"EntLink={m['entity_linking_pct']}% | ModeMatch={m['plan_mode_match_pct']}% | "
            f"GoldAlias={m['gold_alias_hit']}/{m['total']} ({m['gold_alias_recall_pct']}%) | "
            f"FullChain={m['full_chain_hit']}/{m['total']} ({m['full_chain_recall_pct']}%) | "
            f"Acc|FullChain={m['acc_given_full_chain_pct']}%"
        )
    print("Per-hop MH:", mech_block["per_hop_mh"])
    print("Extraction audit:", {
        k: (v["parsing_coverage_pct"], v["verbatim_roundtrip_precision_pct"], v["conflict_tombstone_rate_pct"], v["single_value_false_removals"])
        for k, v in ext_audit.items()
    })
    print("Local build:", local_timing["build_by_history"])
    print("Local exact relay ms:", local_timing["local_exact_relay_ms"])
    print("FC E2E API latency (CARR v2 vs Full):", lat["factconsolidation"].get("carr_v2"), lat["factconsolidation"].get("official_full_context_v1"))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
