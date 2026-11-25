"""Reconciliation tests for formal v2 FactConsolidation and LongMemEval-S* runs."""

from decimal import Decimal
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_factconsolidation_v2_formal_reconciliation():
    run_dir = ROOT / "results" / "test" / "carr-v2-formal-20261124"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cases = list((run_dir / "cases").glob("*.json"))
    assert len(cases) == 562 * 6

    counts = {}
    ans_usd = Decimal("0")
    extra_usd = Decimal("0")
    for p in cases:
        c = json.loads(p.read_text(encoding="utf-8"))
        assert not c.get("error")
        arm = c["arm"]
        counts.setdefault(arm, [0, 0])
        counts[arm][0] += 1
        counts[arm][1] += int(bool(c["correct"]))
        ans_usd += Decimal(str(c["usage_usd"]))
        extra_usd += Decimal(str(c.get("extra_usd", "0")))

    assert counts == {
        "carr_v2": [562, 448],
        "carr_v2_planner_only": [562, 452],
        "mem0_style": [562, 302],
        "bm25_bge_recency": [562, 185],
        "ircot_bge": [562, 117],
        "tombstone_bge_top8": [562, 115],
    }
    planner_usd = Decimal(str(summary["shared_planner_v2"]["usage_usd"]))
    mem0_write_usd = sum(Decimal(str(v["usage_usd"])) for v in summary["mem0_write_usage"].values())
    assert ans_usd + extra_usd + planner_usd + mem0_write_usd == Decimal("2.434929768")


def test_longmemeval_v2_formal_reconciliation():
    run_dir = ROOT / "results" / "test" / "longmemeval-v2-formal-20261124"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"

    cases = list((run_dir / "cases").glob("*.json"))
    assert len(cases) == 240 * 6

    counts = {}
    ans_usd = Decimal("0")
    extra_usd = Decimal("0")
    for p in cases:
        c = json.loads(p.read_text(encoding="utf-8"))
        assert not c.get("error")
        arm = c["arm"]
        counts.setdefault(arm, [0, 0])
        counts[arm][0] += 1
        counts[arm][1] += int(bool(c["correct"]))
        ans_usd += Decimal(str(c["usage_usd"]))
        extra_usd += Decimal(str(c.get("extra_usd", "0")))

    assert counts == {
        "official_full_context": [240, 113],
        "carr_v2": [240, 113],
        "bm25_bge_recency": [240, 101],
        "naive_bge_top8": [240, 90],
        "ircot_bge": [240, 85],
        "mem0_style": [240, 84],
    }
    ext_usd = sum(Decimal(str(v["usage_usd"])) for v in summary["extraction_usage"].values())
    mem0_usd = sum(Decimal(str(v["usage_usd"])) for v in summary["mem0_write_usage"].values())
    assert ans_usd + extra_usd + ext_usd + mem0_usd == Decimal(str(manifest["usage_usd"]))


def test_sensitivity_artifacts_complete():
    sens = json.loads(
        (ROOT / "results" / "analysis" / "v2_sensitivity" / "sensitivity_summary.json").read_text(encoding="utf-8")
    )
    assert len(sens["chunk_topk_grid"]) == 36
    assert len(sens["max_hops_sweep"]) == 4
    assert len(sens["displacement_summary"]) == 9
    assert len(sens["parser_v2_coverage"]) == 3


def test_ark_glm54_resumed_32k_and_diagnostics():
    ark = json.loads(
        (ROOT / "results" / "test" / "ark-backbones-formal-20261124" / "summary.json").read_text(encoding="utf-8")
    )
    assert ark["recorded"] == 948
    by_key = {(r["model"], r["arm"], r["stratum"]): r for r in ark["totals"]}
    assert by_key["glm-5.4", "official_full_context", "32k"]["recorded"] == 175
    assert by_key["glm-5.4", "official_full_context", "32k"]["correct"] == 149
    assert by_key["glm-5.4", "official_full_context", "262k"]["recorded"] == 9
    assert by_key["glm-5.4", "official_full_context", "262k"]["correct"] == 3
    diag = json.loads(
        (ROOT / "results" / "analysis" / "formal_v7_diagnostics" / "diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert diag["cross_backbones"]["formal_32k_paired"]["n_paired"] == 175


def test_tokenplus_glm53_carr_v2_reconciliation():
    glm_dir = ROOT / "results" / "test" / "tokenplus-glm53-v2-20261124"
    fc = json.loads((glm_dir / "summary_factcon_v2_carr.json").read_text(encoding="utf-8"))
    fc_all = next(r for r in fc["rows"] if r["group"] == "all")
    assert fc_all["questions"] == 562 and fc_all["correct"] == 448

    lme = json.loads((glm_dir / "summary_longmemeval_v2_carr.json").read_text(encoding="utf-8"))
    lme_all = next(r for r in lme["rows"] if r["group"] == "all")
    assert lme_all["questions"] == 240 and lme_all["correct"] == 126

    ag = json.loads((ROOT / "results" / "analysis" / "v2_audit_glm53.json").read_text(encoding="utf-8"))
    assert ag["factcon_carr_v2_glm53_vs_ds"]["n"] == 562
    assert ag["factcon_carr_v2_glm53_vs_ds"]["agreement_pct"] == 98.93

    full32 = json.loads((glm_dir / "summary_factcon_full_32k.json").read_text(encoding="utf-8"))
    r32 = next(r for r in full32["rows"] if r["group"] == "all")
    assert r32["questions"] == 175 and r32["correct"] == 141
    assert ag["factcon_32k_glm53_carr_v2_vs_full"]["carr_v2_correct"] == 146

    full64 = json.loads((glm_dir / "summary_factcon_full_64k.json").read_text(encoding="utf-8"))
    r64 = next(r for r in full64["rows"] if r["group"] == "all")
    assert r64["questions"] == 190 and r64["correct"] == 142
    assert ag["factcon_64k_glm53_carr_v2_vs_full"]["carr_v2_correct"] == 161
    assert ag["factcon_262k_partial_glm53_carr_v2_vs_full"]["n"] == 143
    assert ag["factcon_262k_partial_glm53_carr_v2_vs_full"]["full_correct"] == 54


def test_tokenplus_kimi31_carr_v2_reconciliation():
    kimi_dir = ROOT / "results" / "test" / "tokenplus-kimi-k31-v2-20261124"
    s = json.loads((kimi_dir / "summary_factcon_v2_carr.json").read_text(encoding="utf-8"))
    by_g = {r["group"]: r for r in s["rows"]}
    assert by_g["all"]["questions"] == 365 and by_g["all"]["correct"] == 309
    assert by_g["32k"]["correct"] == 146 and by_g["64k"]["correct"] == 163
    ag = json.loads((ROOT / "results" / "analysis" / "v2_audit_kimi31.json").read_text(encoding="utf-8"))
    assert ag["kimi31_vs_ds_365"]["agreement_pct"] == 98.9
    assert ag["kimi31_vs_glm53_365"]["agreement_pct"] == 99.45


def test_tokenplus_qwen4max_carr_v2_and_four_family_unanimity():
    q_dir = ROOT / "results" / "test" / "tokenplus-qwen4max-v2-20261124"
    s = json.loads((q_dir / "summary_factcon_v2_carr.json").read_text(encoding="utf-8"))
    by_g = {r["group"]: r for r in s["rows"]}
    assert by_g["all"]["questions"] == 365 and by_g["all"]["correct"] == 309
    assert by_g["32k"]["correct"] == 146 and by_g["64k"]["correct"] == 163
    ag = json.loads((ROOT / "results" / "analysis" / "v2_audit_qwen4.json").read_text(encoding="utf-8"))
    assert ag["qwen4_vs_kimi31_365"]["agreement_pct"] == 100.0
    assert ag["four_family_unanimous_count"] == 361
    assert ag["four_family_unanimous_pct"] == 98.9


def test_hipporag_ppr_formal_reconciliation():
    fc_dir = ROOT / "results" / "test" / "hipporag-fc-formal-20261124"
    fc_man = json.loads((fc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert fc_man["status"] == "complete"
    fc_cases = [json.loads(p.read_text(encoding="utf-8")) for p in (fc_dir / "cases").glob("*.json")]
    assert len(fc_cases) == 562 * 2 and all(not c.get("error") for c in fc_cases)
    fc_sum = json.loads((fc_dir / "summary.json").read_text(encoding="utf-8"))
    fc_map = {(r["arm"], r["group"]): r for r in fc_sum["rows"]}
    assert fc_map["hipporag_ppr", "all"]["correct"] == 305
    assert fc_map["hipporag_tombstone_ppr", "all"]["correct"] == 307

    lme_dir = ROOT / "results" / "test" / "hipporag-lme-formal-20261124"
    lme_man = json.loads((lme_dir / "manifest.json").read_text(encoding="utf-8"))
    assert lme_man["status"] == "complete"
    lme_cases = [json.loads(p.read_text(encoding="utf-8")) for p in (lme_dir / "cases").glob("*.json")]
    assert len(lme_cases) == 240 * 2 and all(not c.get("error") for c in lme_cases)
    lme_sum = json.loads((lme_dir / "summary.json").read_text(encoding="utf-8"))
    lme_map = {(r["arm"], r["group"]): r for r in lme_sum["rows"]}
    assert lme_map["hipporag_ppr", "all"]["subem_correct"] == 116
    assert lme_map["hipporag_tombstone_ppr", "all"]["subem_correct"] == 119


def test_revision_offline_audit_and_judge_and_ablation_and_stress():
    import re

    rev = json.loads(
        (ROOT / "results" / "analysis" / "revision_offline_audit.json").read_text(encoding="utf-8")
    )
    mech = rev["mechanistic_full_chain"]["mechanistic"]
    assert mech["all"]["total"] == 562
    assert mech["all"]["full_chain_hit"] == 446
    assert mech["mh"]["total"] == 276
    assert mech["mh"]["full_chain_hit"] == 173

    jdg = json.loads(
        (ROOT / "results" / "analysis" / "longmemeval_official_judge.json").read_text(encoding="utf-8")
    )
    assert jdg["total_calls"] == 2160
    assert jdg["total_errors"] == 0
    j_map = {(r["arm"], r["group"]): r for r in jdg["rows"]}
    assert j_map["official_full_context", "all"]["judge_correct"] == 166
    assert j_map["carr_v2", "all"]["judge_correct"] == 179
    assert j_map["carr_v2", "factual_5cat"]["judge_correct"] == 168
    assert j_map["hipporag_tombstone_ppr", "all"]["judge_correct"] == 183
    assert j_map["carr_v2_glm53", "all"]["judge_correct"] == 193

    ab = json.loads(
        (ROOT / "results" / "test" / "ablation-matrix-v2" / "summary.json").read_text(encoding="utf-8")
    )
    ab_map = {(r["arm"], r["group"]): r for r in ab["rows"]}
    assert all(r["errors"] == 0 for r in ab["rows"])
    assert ab_map["A_fact_hybrid", "all"]["correct"] == 292
    assert ab_map["B_latest_dict_hybrid", "all"]["correct"] == 304
    assert ab_map["C_planner_no_recency", "all"]["correct"] == 98
    assert ab_map["D_planner_exact_recency", "all"]["correct"] == 441
    assert ab_map["E_plus_semantic_rel", "all"]["correct"] == 446
    assert ab_map["F_plus_hybrid_fallback", "all"]["correct"] == 452
    assert ab_map["G_full_carr_v2", "all"]["correct"] == 448

    st = json.loads(
        (ROOT / "results" / "test" / "paraphrase-stress-v2" / "summary.json").read_text(encoding="utf-8")
    )
    st_map = {(r["arm"], r["group"]): r for r in st["rows"]}
    assert all(r["errors"] == 0 for r in st["rows"])
    assert st_map["full_context", "all"]["correct"] == 100
    assert st_map["carr_v2_rule", "all"]["correct"] == 0
    assert st_map["carr_v2_llm_extract", "all"]["correct"] == 64

    bib_txt = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_txt))
    assert {"luo2024rog", "reddy2026chainaware"} <= bib_keys
    for doc_name in ("main_en.tex", "main_zh.tex", "main_zh.md"):
        doc_txt = (ROOT / "paper" / doc_name).read_text(encoding="utf-8")
        cited = set()
        for grp in re.findall(r"\\cite\{([^}]+)\}", doc_txt):
            cited.update(k.strip() for k in grp.split(","))
        missing = bib_keys - cited
        assert not missing, f"{doc_name} missing citations: {missing}"


def test_round2_experiment_reconciliation() -> None:
    """Reconcile Round 2 sweep, CAR-style, 117-line stress, frozen holdout, and Group F full-chain numbers."""
    g = json.loads((ROOT / "results" / "analysis" / "graph_regime_sweep.json").read_text(encoding="utf-8"))
    scales = g["scales"]
    assert [s["topology"]["active_facts"] for s in scales] == [500, 2000, 6000, 16450]
    for s in scales:
        for grp in ("all", "sh", "mh", "mh_2hop", "mh_3plus_hop"):
            assert s["hipporag_tombstone_ppr"][grp]["errors"] == 0
            assert s["carr_v2"][grp]["errors"] == 0
    assert [s["hipporag_tombstone_ppr"]["mh"]["acc_pct"] for s in scales] == [27.55, 16.33, 12.24, 9.18]
    assert all(s["carr_v2"]["all"]["acc_pct"] == 71.57 for s in scales)

    car = json.loads((ROOT / "results" / "test" / "car-baseline-v2" / "summary.json").read_text(encoding="utf-8"))
    car_map = {(r["arm"], r["group"]): r for r in car["rows"]}
    assert car_map["car_style", "all"]["errors"] == 0
    assert (car_map["car_style", "all"]["correct"], car_map["car_style", "all"]["questions"]) == (416, 562)
    assert car_map["car_style", "sh"]["correct"] == 257
    assert car_map["car_style", "mh"]["correct"] == 159
    assert car_map["car_style", "all"]["mean_llm_calls_per_q"] == 3.35
    assert (car["paired_carr_v2_vs_car_style"]["wins"], car["paired_carr_v2_vs_car_style"]["losses"]) == (45, 13)

    st21 = json.loads((ROOT / "results" / "test" / "paraphrase-stress-v2-1" / "summary.json").read_text(encoding="utf-8"))
    assert st21["lines_per_history"] == 117
    st_map = {(r["arm"], r["group"]): r for r in st21["rows"]}
    assert all(r["errors"] == 0 for r in st21["rows"])
    assert st_map["full_context", "all"]["correct"] == 100
    assert st_map["bm25_bge_recency", "all"]["correct"] == 100
    assert st_map["carr_v2_rule_line_fb", "all"]["correct"] == 98
    assert st_map["carr_v2_llm_tx_time", "all"]["correct"] == 61
    assert st_map["carr_v2_llm_tx_time", "out_of_order_timestamp"]["correct"] == 0
    assert st_map["carr_v2_llm_valid_time", "all"]["correct"] == 81
    assert st_map["carr_v2_llm_valid_time", "out_of_order_timestamp"]["correct"] == 20

    ho = json.loads((ROOT / "results" / "test" / "frozen-holdout-v2" / "summary.json").read_text(encoding="utf-8"))
    assert ho["frozen_src_carr_v2"] is True
    assert ho["total_errors"] == 0
    ho_lme = {r["arm"]: r for r in ho["lme_dev_summary"]}
    assert (ho_lme["carr_v2"]["judge_pass"], ho_lme["carr_v2"]["f5_judge_pass"]) == (45, 42)
    assert (ho_lme["hipporag_tombstone_ppr"]["judge_pass"], ho_lme["official_full_context"]["judge_pass"]) == (46, 47)
    ho_uns = {r["arm"]: r for r in ho["unseen_rel_summary"]}
    assert ho_uns["carr_v2"]["correct"] == 67
    assert (ho_uns["carr_v2"]["sh_correct"], ho_uns["carr_v2"]["mh_correct"]) == (35, 32)
    assert ho_uns["hipporag_tombstone_ppr"]["correct"] == 100

    rev = json.loads((ROOT / "results" / "analysis" / "revision_offline_audit.json").read_text(encoding="utf-8"))
    po_m = rev["mechanistic_full_chain"]["planner_only_mechanistic"]
    assert (po_m["all"]["full_chain_hit"], po_m["all"]["full_chain_recall_pct"]) == (446, 79.36)
    assert (ROOT / "paper" / "figures" / "fig8_graph_regime_phase.pdf").exists()


def test_round3_reconciliation() -> None:
    """Reconcile Qwen4-Max cross-family Judge, tone calibration, formal Method, and CAR protocol."""
    xf = json.loads((ROOT / "results" / "analysis" / "longmemeval_crossfamily_judge.json").read_text(encoding="utf-8"))
    assert xf["judge_model"] == "qwen/qwen4-max"
    assert xf["total_calls"] == 1200
    assert xf["total_errors"] == 0
    pooled = xf["inter_judge_vs_deepseek"]["pooled_1200"]
    assert pooled["questions"] == 1200
    assert pooled["agreement_pct"] == 97.58
    assert pooled["cohens_kappa"] == 0.94

    xf_map = {(r["arm"], r["group"]): r for r in xf["rows"]}
    assert (xf_map["carr_v2", "all"]["qwen_judge_correct"], xf_map["carr_v2", "factual_5cat"]["qwen_judge_correct"]) == (176, 166)
    assert (xf_map["official_full_context", "all"]["qwen_judge_correct"], xf_map["official_full_context", "factual_5cat"]["qwen_judge_correct"]) == (164, 151)
    assert xf_map["hipporag_tombstone_ppr", "all"]["qwen_judge_correct"] == 179
    assert (xf_map["carr_v2_glm53", "all"]["qwen_judge_correct"], xf_map["carr_v2_glm53", "factual_5cat"]["qwen_judge_correct"]) == (188, 176)

    for fn in ("main_en.tex", "main_zh.tex", "main_zh.md"):
        txt = (ROOT / "paper" / fn).read_text(encoding="utf-8")
        assert "phase transition" not in txt.lower(), f"De-escalate phase transition in {fn}"
        assert "相变" not in txt, f"De-escalate 相变 in {fn}"

    en_txt = (ROOT / "paper" / "main_en.tex").read_text(encoding="utf-8")
    zh_txt = (ROOT / "paper" / "main_zh.tex").read_text(encoding="utf-8")
    assert r"\subsection{Operating Assumptions and Boundary Conditions}" in en_txt
    assert r"\subsection{适用前提与边界条件}" in zh_txt
    assert "CAR-style" in en_txt and "controlled reproduction" in en_txt.lower()
    assert "CAR-style" in zh_txt and "受控复现" in zh_txt
    assert r"\label{tab:crossfamily_judge}" in (ROOT / "paper" / "tables.tex").read_text(encoding="utf-8")


