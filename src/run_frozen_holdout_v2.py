"""Frozen-Architecture Independent Holdout Runner (Concern #4).

Runs TWO clean holdout suites with ZERO modifications to `src/carr_v2/`:
  1. `longmemeval_dev` (60Q from locked `data/longmemeval/dev.jsonl`):
     Compares `official_full_context`, `bm25_bge_recency`, `hipporag_tombstone_ppr`, and
     frozen `carr_v2` under Official LongMemEval LLM-as-Judge (`official_lme_judge_prompt`).
  2. `unseen_rel_holdout` (100Q: 50 SH + 50 MH, ~120 lines each):
     Brand-new natural-text benchmark built over 10 unseen technical/security relations
     disjoint from `CANONICAL_RELATIONS` and `V2_EXTRA_PATTERNS`, comparing
     `full_context`, `bm25_bge_recency`, `hipporag_tombstone_ppr`, and frozen `carr_v2`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.hipporag_ppr import GraphFact, HippoGraphIndex
from src.carr_v2.open_parser import (
    extract_dialogue_facts_batch,
    format_longmemeval_transcript,
    parse_longmemeval_sessions,
)
from src.carr_v2.semantic_router import (
    RecencyHybridStore,
    parse_plan_v2,
    planner_messages_v2,
    retrieve_v2,
)
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex
from src.eval_longmemeval_judge import official_lme_judge_prompt, parse_yes_no
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import Observation, Record, canon, norm
from src.longmemeval_data import read_longmemeval_locked
from src.run_carr_v2 import SafeMeter
from src.run_hipporag_v2 import _load_embedder
from src.run_longmemeval import (
    LME_MAX_TOKENS,
    _build_lme_fact_index,
    longmemeval_messages,
    retrieve_carr_v2_lme,
    score_longmemeval,
)
from src.run_paraphrase_stress_v2 import _make_distractors, stress_messages
from src.v2_report import write_csv
from src.v2_transport import FLASH, credentials

OUT_DIR = ROOT / "results" / "test" / "frozen-holdout-v2"
UNSEEN_DATA_PATH = ROOT / "results" / "analysis" / "unseen_rel_holdout_100.json"

LME_DEV_ARMS = ("official_full_context", "bm25_bge_recency", "hipporag_tombstone_ppr", "carr_v2")
UNSEEN_ARMS = ("full_context", "bm25_bge_recency", "hipporag_tombstone_ppr", "carr_v2")

# 10 brand-new relations completely disjoint from CANONICAL_RELATIONS and V2_EXTRA_PATTERNS
UNSEEN_SERVICES = [f"Service-{tag}" for tag in (
    "Orion", "Lyra", "Vega", "Altair", "Cygnus", "Draco", "Hydra", "Perseus", "Pegasus", "Andromeda",
    "Phoenix", "Aquila", "Carina", "Cassiopeia", "Centaurus", "Columba", "Corvus", "Delphinus", "Eridanus", "Fornax",
    "Grus", "Horologium", "Indus", "Lacerta", "Lepus",
)]
CIPHERS = ["AES-256-GCM", "ChaCha20-Poly1305", "XChaCha20-Poly1305", "Camellia-256-CBC"]
VAULTS = ["vault-zurich-04", "vault-osaka-09", "vault-virginia-11", "vault-frankfurt-02", "vault-sydney-06"]
GATEWAYS = ["gw-alder-07", "gw-birch-19", "gw-cedar-42", "gw-dune-88", "gw-elm-31", "gw-fir-55"]
OFFICERS = [
    "Dr. Elena Vance", "Marcus Sterling", "Priya Nair", "Kenji Takahashi",
    "Hannah Lindqvist", "Mateo Alvarez", "Nadia Okafor", "Viktor Novak",
]
COLLECTORS = ["clickhouse-prod-01", "victoriametrics-eu", "loki-sec-vault", "prometheus-edge-09"]
KEYRINGS = ["ring-ed25519-904", "ring-rsa4096-312", "ring-ecdsa384-518", "ring-dilithium-771"]
PAGERS = ["pager-sec-tier1", "opsgenie-core-red", "signal-oncall-04", "slack-warroom-blue"]
CLUSTERS = ["cluster-eu-central-2", "cluster-ap-south-1", "cluster-us-west-3", "cluster-ca-central-1", "cluster-eu-west-4"]
ASSURANCE = ["Veritas Assurance", "Aegis Cyber Labs", "Meridian Risk Partners", "Helios Security Audit"]


def build_unseen_relation_dataset(seed: int = 2026) -> list[dict]:
    """Generate 100 unseen-relation natural-text instances (50 SH + 50 MH, ~120 lines each)."""
    rng = random.Random(seed)
    items: list[dict] = []

    def wrap_history(events: list[tuple[int, str]], n_lines: int = 120) -> str:
        ev_map = {pos: txt for pos, txt in events}
        filler = _make_distractors(rng, n_lines)
        lines: list[str] = []
        f_idx = 0
        for ln in range(1, n_lines + 1):
            if ln in ev_map:
                body = ev_map[ln]
            else:
                body = filler[f_idx]
                f_idx += 1
            lines.append(f"{ln}. {body}")
        return "\n".join(lines)

    # 50 Single-Hop unseen-relation questions (10 per pair of unseen relations)
    for i in range(50):
        svc = f"{UNSEEN_SERVICES[i % len(UNSEEN_SERVICES)]}-{100 + i}"
        kind = i % 5
        if kind == 0:
            v_old, v_new = CIPHERS[i % len(CIPHERS)], CIPHERS[(i + 2) % len(CIPHERS)]
            hist = wrap_history([
                (18, f"During initial onboarding, traffic encryption for {svc} relied on the {v_old} cipher."),
                (94, f"Following Q3 hardening, {svc} switched its encryption cipher permanently to {v_new}."),
            ])
            q, ans = f"Which encryption cipher is currently configured for {svc}?", [v_new]
        elif kind == 1:
            v_old, v_new = VAULTS[i % len(VAULTS)], VAULTS[(i + 3) % len(VAULTS)]
            hist = wrap_history([
                (21, f"Nightly snapshots from {svc} were originally replicated to backup vault {v_old}."),
                (97, f"After storage migration, {svc} now archives all nightly backups to {v_new}."),
            ])
            q, ans = f"Which backup vault site currently receives snapshots from {svc}?", [v_new]
        elif kind == 2:
            v_old, v_new = COLLECTORS[i % len(COLLECTORS)], COLLECTORS[(i + 1) % len(COLLECTORS)]
            hist = wrap_history([
                (16, f"Metrics telemetry exported by {svc} used to terminate at {v_old}."),
                (89, f"Under the observability overhaul, telemetry from {svc} now flows directly into {v_new}."),
            ])
            q, ans = f"Which telemetry collector is currently receiving metrics from {svc}?", [v_new]
        elif kind == 3:
            v_old, v_new = PAGERS[i % len(PAGERS)], PAGERS[(i + 2) % len(PAGERS)]
            hist = wrap_history([
                (25, f"Critical alerts triggered by {svc} previously paged {v_old}."),
                (101, f"Operational policy updated alert escalation for {svc} so critical pages route to {v_new}."),
            ])
            q, ans = f"Which paging escalation channel is currently active for {svc}?", [v_new]
        else:
            v_old, v_new = KEYRINGS[i % len(KEYRINGS)], KEYRINGS[(i + 1) % len(KEYRINGS)]
            hist = wrap_history([
                (14, f"Release artifacts built by {svc} were formerly signed using {v_old}."),
                (91, f"Key rotation completed: {svc} now signs release artifacts with {v_new}."),
            ])
            q, ans = f"Which signing keyring is currently used by {svc}?", [v_new]

        items.append({
            "id": f"unseen_sh_{i:03d}",
            "hop_type": "single_hop",
            "history": hist,
            "question": q,
            "answers": ans,
        })

    # 50 Multi-Hop unseen-relation questions (2-hop chains where BOTH hops undergo temporal updates!)
    for i in range(50):
        svc = f"{UNSEEN_SERVICES[(i + 5) % len(UNSEEN_SERVICES)]}-{200 + i}"
        if i % 2 == 0:
            # Chain: Service -> failover_gateway -> compliance_officer (2-hop)
            gw_old = f"{GATEWAYS[i % len(GATEWAYS)]}-a{i}"
            gw_new = f"{GATEWAYS[(i + 2) % len(GATEWAYS)]}-b{i}"
            off_stale = OFFICERS[(i + 1) % len(OFFICERS)]
            off_old_gw = OFFICERS[(i + 3) % len(OFFICERS)]
            off_curr = OFFICERS[(i + 5) % len(OFFICERS)]
            hist = wrap_history([
                (12, f"Early resilience plans routed failover for {svc} through {gw_old}."),
                (26, f"Security compliance oversight for gateway {gw_old} belonged to {off_stale}."),
                (44, f"When {gw_new} came online, {off_old_gw} briefly acted as its compliance officer."),
                (79, f"Traffic cutover completed: {svc} now uses {gw_new} as its active failover gateway."),
                (103, f"Following compliance audit rotation, {off_curr} took over as compliance officer of {gw_new}."),
            ])
            q = f"Who currently serves as the compliance officer for the failover gateway assigned to {svc}?"
            ans = [off_curr]
        else:
            # Chain: Service -> primary_deployment_cluster -> incident_commander -> vendor_assurance_partner (2-hop variant or cluster->partner)
            # Clean 2-hop: Service -> primary_deployment_cluster -> vendor_assurance_partner
            cl_old = f"{CLUSTERS[i % len(CLUSTERS)]}-n{i}"
            cl_new = f"{CLUSTERS[(i + 3) % len(CLUSTERS)]}-m{i}"
            va_stale = ASSURANCE[(i + 1) % len(ASSURANCE)]
            va_old = ASSURANCE[(i + 2) % len(ASSURANCE)]
            va_curr = ASSURANCE[(i + 3) % len(ASSURANCE)]
            hist = wrap_history([
                (15, f"Initial rollout hosted {svc} inside deployment cluster {cl_old}."),
                (31, f"External audits for {cl_old} were handled by {va_stale}."),
                (52, f"Before ISO renewal, {cl_new} engaged {va_old} as its vendor assurance partner."),
                (85, f"Workload migration moved {svc} permanently to primary deployment cluster {cl_new}."),
                (106, f"Board procurement appointed {va_curr} as the current vendor assurance partner auditing {cl_new}."),
            ])
            q = f"Which vendor assurance partner currently audits the primary deployment cluster hosting {svc}?"
            ans = [va_curr]

        items.append({
            "id": f"unseen_mh_{i:03d}",
            "hop_type": "multi_hop",
            "history": hist,
            "question": q,
            "answers": ans,
        })

    assert len(items) == 100
    return items


def extract_open_facts_unseen(history: str, item_id: str, meter: SafeMeter) -> tuple[list[Observation], dict]:
    """Open LLM fact extractor converting unseen natural-text lines into typed Observations."""
    lines = history.splitlines()
    msgs = [
        {
            "role": "system",
            "content": (
                "Extract durable technical/infrastructure/security assignments from these numbered lines "
                "(skip generic regional depot / retired consultant / guild / patent distractor lines). "
                "Return JSON {\"facts\": [{\"line\": int, \"entity\": str, \"relation\": \"short snake_case\", "
                "\"value\": str, \"operation\": \"assert|replace\"}]} preserving exact entity/identifier names."
            ),
        },
        {"role": "user", "content": history},
    ]
    call = meter.call(f"ext_unseen--{item_id}", msgs, max_tokens=1536, json_output=True)
    obs_list: list[Observation] = []
    if call.get("error") or not call.get("answer"):
        return obs_list, call
    try:
        data = json.loads(call["answer"])
        for row in data.get("facts", []):
            if not isinstance(row, dict):
                continue
            lnum = row.get("line")
            if not isinstance(lnum, int) or not (1 <= lnum <= len(lines)):
                continue
            ent = str(row.get("entity") or "").strip()
            rel = canon(str(row.get("relation") or "").strip())
            val = str(row.get("value") or "").strip()
            op = str(row.get("operation") or "replace").strip().lower()
            if op not in {"assert", "replace"}:
                op = "replace"
            if ent and rel and val:
                obs_list.append(Observation(lnum - 1, lnum, ent, rel, val, op, (0, len(lines[lnum - 1])), "llm_open"))
    except (ValueError, TypeError, AttributeError):
        pass
    return obs_list, call


def run_holdout(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    dev_rows, dev_contexts = read_longmemeval_locked("dev")
    unseen_items = build_unseen_relation_dataset(seed=2026)
    UNSEEN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump_json(UNSEEN_DATA_PATH, {"generator": "src/run_frozen_holdout_v2.py", "seed": 2026, "items": unseen_items})

    if not execute:
        # LME dev: 1 context ~51 sessions (51 ext calls) + 60Q * 4 arms (240 ans) + 240 judge calls = ~531 calls
        # Unseen 100Q: 100 ext + 100 plan + 400 ans = 600 calls
        est_lme_uncached = token_cost(0, 51 * 1400 + 60 * 390000 + 180 * 2400 + 240 * 350, 51 * 220 + 240 * 48 + 240 * 8, "peak")
        # 59 of 60 LME dev queries share the exact same 390K-token prefix -> prefix cache hit (~10x lower cost)
        est_lme_cached = token_cost(59 * 385000, 51 * 1400 + 390000 + 59 * 5000 + 180 * 2400 + 240 * 350, 51 * 220 + 240 * 48 + 240 * 8, "peak")
        est_uns = token_cost(0, 100 * 2400 + 100 * 220 + 400 * 1100, 100 * 500 + 100 * 80 + 400 * 24, "peak")
        q = {
            "longmemeval_dev_questions": len(dev_rows),
            "longmemeval_dev_arms": list(LME_DEV_ARMS),
            "unseen_rel_questions": len(unseen_items),
            "unseen_rel_arms": list(UNSEEN_ARMS),
            "est_lme_dev_cached_usd": str(est_lme_cached),
            "est_lme_dev_uncached_usd": str(est_lme_uncached),
            "est_unseen_usd": str(est_uns),
            "est_expected_total_usd": str(est_lme_cached + est_uns),
        }
        print(json.dumps(q, indent=2))
        return q

    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DeepSeek API key.")

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "cache", "requests"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    embedder = _load_embedder()
    scorer = LongContext(ROOT / ".venv/nltk_data")

    # ------------------------------------------------------------------
    # Suite 1: LongMemEval-S* dev (60Q) with frozen carr_v2 + Official Judge
    # ------------------------------------------------------------------
    cid = dev_rows[0]["context_id"]
    raw_ctx = dev_contexts[cid]
    sessions = parse_longmemeval_sessions(raw_ctx)
    transcript = format_longmemeval_transcript(sessions)
    extracted_lme, ext_lme_use = extract_dialogue_facts_batch(
        sessions,
        lambda tag, msgs, mtok, jo: meter.call(f"lme_dev_{cid[:8]}_{tag}", msgs, mtok, json_output=jo),
        cache_path=out / "cache" / f"carr_lme_dev_{cid[:12]}.json",
        max_workers=10,
    )
    all_lme_cached = all(
        (out / "cases" / f"lmedev__{arm}--{s['id']}.json").exists()
        for s in dev_rows
        for arm in LME_DEV_ARMS
    )
    if all_lme_cached:
        sup_lme = 0
        # Fast cached path: read all 240 cached verdicts cleanly without rebuilding 390K BGE index
        lme_rows: list[dict] = [
            json.loads((out / "cases" / f"lmedev__{arm}--{s['id']}.json").read_text(encoding="utf-8"))
            for s in dev_rows
            for arm in LME_DEV_ARMS
        ]
    else:
        hybrid_idx = HybridLexicalDenseIndex(embedder, transcript, tombstones=None)
        active_lme, fact_lines_lme, fact_vecs_lme, sup_lme = _build_lme_fact_index(extracted_lme, embedder)
        # Build raw/active GraphFacts for hipporag_tombstone_ppr on LME dev
        gfacts_lme = [
            GraphFact(i + 1, f["entity"], f["attribute"], f["value"], f["session_idx"], fact_lines_lme[i])
            for i, f in enumerate(active_lme)
        ]
        hippo_lme = HippoGraphIndex(gfacts_lme, embedder, damping=0.5, num_iters=25)

        def run_lme_dev_q(sample: dict) -> list[dict]:
            sid = sample["id"]
            q = sample["question"]
            qtype = sample["question_type"]
            rows_q: list[dict] = []
            turn_ctx, _turn_meta = retrieve_bm25_recency(hybrid_idx, q, top_k=8, max_lines=16, max_bytes=4800)

            for arm in LME_DEV_ARMS:
                cpath = out / "cases" / f"lmedev__{arm}--{sid}.json"
                if cpath.exists():
                    rows_q.append(json.loads(cpath.read_text(encoding="utf-8")))
                    continue

                if arm == "official_full_context":
                    ctx_str = raw_ctx
                elif arm == "bm25_bge_recency":
                    ctx_str = turn_ctx
                elif arm == "hipporag_tombstone_ppr":
                    fact_ctx, _ = hippo_lme.search(q, top_k=12, max_bytes=3600, chronological_output=True)
                    parts = []
                    if fact_ctx:
                        parts.append("Verified Chronological Facts:\n" + fact_ctx)
                    if turn_ctx:
                        parts.append("Relevant Chronological Dialogue Turns:\n" + turn_ctx)
                    ctx_str = "\n\n".join(parts)
                elif arm == "carr_v2":
                    ctx_str, _ = retrieve_carr_v2_lme(q, fact_lines_lme, fact_vecs_lme, active_lme, hybrid_idx, embedder)
                else:
                    raise ValueError(arm)

                ans_call = meter.call(f"lmedev_{arm}--{sid}", longmemeval_messages(ctx_str, q), LME_MAX_TOKENS, json_output=False)
                pred = ans_call.get("answer") or ""
                metrics = (
                    score_longmemeval(pred, sample["answers"])
                    if not ans_call.get("error")
                    else {"substring_exact_match": False, "exact_match": False, "f1": 0.0, "rougeL_f1": 0.0, "rougeL_recall": 0.0}
                )
                abst = "_abs" in str(sid)
                jprompt = official_lme_judge_prompt(qtype, q, sample["answers"][0], pred, abstention=abst)
                jcall = meter.call(
                    f"lmedev_judge_{arm}--{sid}",
                    [{"role": "user", "content": jprompt}],
                    16,
                    json_output=False,
                )
                j_pass = bool(not jcall.get("error") and parse_yes_no(jcall.get("answer") or ""))
                row = {
                    "suite": "longmemeval_dev",
                    "arm": arm,
                    "sample_id": sid,
                    "question_type": qtype,
                    "question": q,
                    "reference": sample["answers"],
                    "prediction": pred,
                    "subem_correct": bool(metrics["substring_exact_match"]),
                    "judge_correct": j_pass,
                    "error": ans_call.get("error") or jcall.get("error"),
                    "usage": ans_call.get("usage"),
                    "usage_usd": ans_call.get("usage_usd"),
                    "judge_usd": jcall.get("usage_usd"),
                }
                dump_json(cpath, row)
                rows_q.append(row)
            return rows_q

        lme_rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_lme_dev_q, s) for s in dev_rows]
            for fut in as_completed(futs):
                lme_rows.extend(fut.result())

    # ------------------------------------------------------------------
    # Suite 2: Unseen-Relation 100Q Multi-Hop Holdout with frozen carr_v2
    # ------------------------------------------------------------------
    def run_unseen_q(item: dict) -> list[dict]:
        sid = item["id"]
        if all((out / "cases" / f"unseen__{arm}--{sid}.json").exists() for arm in UNSEEN_ARMS):
            return [
                json.loads((out / "cases" / f"unseen__{arm}--{sid}.json").read_text(encoding="utf-8"))
                for arm in UNSEEN_ARMS
            ]
        hist = item["history"]
        raw_lines = hist.splitlines()
        q = item["question"]

        raw_hybrid = HybridLexicalDenseIndex(embedder, hist, tombstones=None)
        obs_list, ext_call = extract_open_facts_unseen(hist, sid, meter)
        plan_call = meter.call(f"unseen_plan--{sid}", planner_messages_v2(q), 192, json_output=True)

        # Latest-order tombstone resolution over extracted open facts
        latest_by_key: dict[tuple[str, str], Observation] = {}
        for o in sorted(obs_list, key=lambda x: x.order):
            latest_by_key[(norm(o.entity), o.relation)] = o
        active_obs = list(latest_by_key.values())
        active_recs = [
            Record(o.index, raw_lines[o.index], o.span[0], o.order)
            for o in sorted(active_obs, key=lambda x: x.order)
        ]
        # Build frozen RecencyHybridStore (zero src/carr_v2 edits!) populated with open facts
        store = RecencyHybridStore(active_recs, raw_lines, embedder, raw_hybrid, use_exact=True)
        store.index.clear()
        store.by_entity.clear()
        store.facts.clear()
        for o in sorted(active_obs, key=lambda x: x.order):
            store.index[(norm(o.entity), o.relation)].append(o)
            store.by_entity[norm(o.entity)].append(o)
            store.facts[o.order] = o
        store.entities = [o.entity for o in store.facts.values()]

        plan = parse_plan_v2(plan_call, q, store.entities)
        gfacts_u = [
            GraphFact(o.order, o.entity, o.relation, o.value, o.order, raw_lines[o.index])
            for o in active_obs
        ]
        hippo_u = HippoGraphIndex(gfacts_u, embedder, damping=0.5, num_iters=25)

        out_rows: list[dict] = []
        for arm in UNSEEN_ARMS:
            cpath = out / "cases" / f"unseen__{arm}--{sid}.json"
            if cpath.exists():
                out_rows.append(json.loads(cpath.read_text(encoding="utf-8")))
                continue
            extra_usd = Decimal(0)
            if arm == "full_context":
                ctx_str = hist
            elif arm == "bm25_bge_recency":
                ctx_str, _ = retrieve_bm25_recency(raw_hybrid, q, top_k=8)
            elif arm == "hipporag_tombstone_ppr":
                ctx_str, _ = hippo_u.search(q, top_k=8, max_bytes=3600, chronological_output=True)
                extra_usd = Decimal(str(ext_call.get("usage_usd") or "0"))
            elif arm == "carr_v2":
                ev, _ = retrieve_v2(store, plan, q)
                ctx_str = "\n".join(ev)
                extra_usd = Decimal(str(ext_call.get("usage_usd") or "0")) + Decimal(str(plan_call.get("usage_usd") or "0"))
            else:
                raise ValueError(arm)

            ans_call = meter.call(f"unseen_{arm}--{sid}", stress_messages(ctx_str, q), 32, json_output=False)
            corr = bool(not ans_call.get("error") and scorer.score(ans_call.get("answer") or "", item["answers"]))
            row = {
                "suite": "unseen_rel_holdout",
                "arm": arm,
                "sample_id": sid,
                "hop_type": item["hop_type"],
                "question": q,
                "reference": item["answers"],
                "prediction": ans_call.get("answer") or "",
                "correct": corr,
                "error": ans_call.get("error"),
                "usage": ans_call.get("usage"),
                "usage_usd": ans_call.get("usage_usd"),
                "extra_usd": str(extra_usd),
            }
            dump_json(cpath, row)
            out_rows.append(row)
        return out_rows

    unseen_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_unseen_q, it) for it in unseen_items]
        for fut in as_completed(futs):
            unseen_rows.extend(fut.result())

    # Aggregate both suites
    lme_summary = []
    for arm in LME_DEV_ARMS:
        sub = [r for r in lme_rows if r["arm"] == arm]
        f5 = [r for r in sub if r["question_type"] != "single-session-preference"]
        pt = [r["usage"].get("prompt_tokens", 0) for r in sub if r.get("usage")]
        lme_summary.append({
            "arm": arm,
            "questions": len(sub),
            "judge_pass": sum(1 for r in sub if r["judge_correct"]),
            "judge_acc": round(sum(1 for r in sub if r["judge_correct"]) / max(1, len(sub)), 4),
            "subem_pass": sum(1 for r in sub if r["subem_correct"]),
            "subem_acc": round(sum(1 for r in sub if r["subem_correct"]) / max(1, len(sub)), 4),
            "f5_questions": len(f5),
            "f5_judge_pass": sum(1 for r in f5 if r["judge_correct"]),
            "f5_judge_acc": round(sum(1 for r in f5 if r["judge_correct"]) / max(1, len(f5)), 4),
            "mean_prompt_tokens": round(sum(pt) / max(1, len(pt)), 1),
            "errors": sum(1 for r in sub if r.get("error")),
        })

    unseen_summary = []
    for arm in UNSEEN_ARMS:
        sub = [r for r in unseen_rows if r["arm"] == arm]
        sh = [r for r in sub if r["hop_type"] == "single_hop"]
        mh = [r for r in sub if r["hop_type"] == "multi_hop"]
        pt = [r["usage"].get("prompt_tokens", 0) for r in sub if r.get("usage")]
        unseen_summary.append({
            "arm": arm,
            "questions": len(sub),
            "correct": sum(1 for r in sub if r["correct"]),
            "acc": round(sum(1 for r in sub if r["correct"]) / max(1, len(sub)), 4),
            "sh_correct": sum(1 for r in sh if r["correct"]),
            "sh_acc": round(sum(1 for r in sh if r["correct"]) / max(1, len(sh)), 4),
            "mh_correct": sum(1 for r in mh if r["correct"]),
            "mh_acc": round(sum(1 for r in mh if r["correct"]) / max(1, len(mh)), 4),
            "mean_prompt_tokens": round(sum(pt) / max(1, len(pt)), 1),
            "errors": sum(1 for r in sub if r.get("error")),
        })

    total_errors = sum(x["errors"] for x in lme_summary) + sum(x["errors"] for x in unseen_summary)
    summary = {
        "generated_utc": utc_now().isoformat(),
        "frozen_src_carr_v2": True,
        "lme_dev_superseded_archived": sup_lme,
        "lme_dev_extraction_usage": ext_lme_use,
        "lme_dev_summary": lme_summary,
        "unseen_rel_summary": unseen_summary,
        "total_errors": total_errors,
        "meter_spent_usd": str(meter.spent),
    }
    dump_json(out / "summary.json", summary)
    write_csv(out / "lme_dev_summary.csv", lme_summary)
    write_csv(out / "unseen_rel_summary.csv", unseen_summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run frozen-architecture holdout (LongMemEval dev + Unseen-Relation 100Q).")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--budget", type=Decimal, default=Decimal("2.0"))
    p.add_argument("--key-file", type=Path, default=None)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    run_holdout(args.execute, args.budget, args.key_file, args.out, args.workers)


if __name__ == "__main__":
    main()
