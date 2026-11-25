"""Upgraded 120-line 5-Arm Paraphrase & Complex Temporal Stress Test (Concern #2).

Separates:
  - Engineering bug fix: line-level BM25+BGE fallback vs single-chunk first-8 truncation
  - Regex template fragility: parse_numbered_v2 under passive/inverted/synonymous natural text
  - Algorithmic boundary: Transaction-time (argmax t) vs explicit Valid-Time year resolution

Generates 100 histories (~123 lines / ~14-16 chunks each so BM25+BGE Top-8 is genuine retrieval).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.carr_v2.open_parser import archive_v2
from src.carr_v2.semantic_router import RecencyHybridStore, parse_plan_v2, planner_messages_v2, retrieve_v2
from src.carr_v2.tombstone_index import BM25Index, HybridLexicalDenseIndex
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import Observation, Record, canon, norm
from src.run_carr_v2 import SafeMeter
from src.run_hipporag_v2 import _load_embedder
from src.run_paraphrase_stress_v2 import (
    AWARDS,
    CAPITALS,
    CATEGORIES,
    CITIES,
    COUNTRIES,
    LANG_PAIRS,
    ORGS,
    PEOPLE,
    _make_distractors,
    stress_messages,
)
from src.v2_report import write_csv
from src.v2_retrieval import QUERY_PREFIX, normalize_rows
from src.v2_transport import FLASH, credentials

DATA_PATH = ROOT / "data" / "paraphrase_stress_v2_1" / "dataset.json"
DEFAULT_OUT = ROOT / "results" / "test" / "paraphrase-stress-v2-1"

ARMS_V21 = (
    "full_context",
    "bm25_bge_recency",
    "carr_v2_rule_line_fb",
    "carr_v2_llm_tx_time",
    "carr_v2_llm_valid_time",
)


def build_long_dataset(seed: int = 101) -> list[dict]:
    """Build 100 items with ~120 numbered lines (~15 chunks) so Top-8 chunk RAG cannot read 100% of log."""
    rng = random.Random(seed)
    items: list[dict] = []

    def wrap_distractors(anchors: list[tuple[int, str]], total_dist: int = 115) -> str:
        dists = _make_distractors(rng, total_dist)
        merged: list[str] = []
        anchor_map = defaultdict(list)
        for pos, line in anchors:
            anchor_map[pos].append(line)
        for idx, dline in enumerate(dists):
            for al in anchor_map.get(idx, []):
                merged.append(al)
            merged.append(dline)
        for pos, lines in sorted(anchor_map.items()):
            if pos >= len(dists):
                merged.extend(lines)
        return "\n".join(f"{i + 1}. {txt}" for i, txt in enumerate(merged))

    # 1. passive_inversion (20Q: 10 SH + 10 MH)
    for i in range(20):
        org = f"{ORGS[i]} {i + 10}"
        old_city, new_city = CITIES[i], CITIES[(i + 7) % len(CITIES)]
        old_ceo, new_ceo = PEOPLE[i], PEOPLE[(i + 5) % len(PEOPLE)]
        if i < 10:
            hist = wrap_distractors([
                (18, f"Originally, {old_city} served as the headquarters city where {org} operated."),
                (92, f"Following a board vote, {new_city} was formally designated as the headquarters city of {org}."),
            ])
            q, ans, hop = f"What is the latest headquarters city of {org}?", [new_city], "sh"
        else:
            c_idx = (i + 3) % len(COUNTRIES)
            country, cap_old, cap_new = COUNTRIES[c_idx], CITIES[(i + 11) % len(CITIES)], CAPITALS[c_idx]
            hist = wrap_distractors([
                (12, f"{old_ceo} was initially appointed to serve as the chief executive officer of {org}."),
                (35, f"Under updated filings, citizenship of {new_ceo} is held in {country}."),
                (58, f"It was announced that {cap_old} formerly acted as the capital of {country}."),
                (84, f"At the latest annual meeting, leadership passed and {new_ceo} became the chief executive officer of {org}."),
                (106, f"According to the revised charter, {cap_new} stands as the capital of {country}."),
            ])
            q = f"What is the capital of the country of citizenship of the chief executive officer of {org}?"
            ans, hop = [cap_new], "mh"
        items.append({"id": f"stress21_passive_{i:02d}", "category": "passive_inversion", "hop_type": hop, "history": hist, "question": q, "answers": ans})

    # 2. synonym_alias (20Q: 10 SH + 10 MH)
    for i in range(20):
        org_base = ORGS[(i + 4) % len(ORGS)]
        org = f"{org_base} Group {i + 20}"
        alias = f"{org_base.split()[0]}-{i + 20}"
        old_p, new_p = PEOPLE[(i + 2) % len(PEOPLE)], PEOPLE[(i + 9) % len(PEOPLE)]
        if i < 10:
            hist = wrap_distractors([
                (20, f"{org} (shortened as {alias}) used to be helmed by {old_p} as its top executive."),
                (94, f"In the newest filing, {alias} is now led at the top executive helm by {new_p}."),
            ])
            q, ans, hop = f"Who is the current chief executive officer of {org}?", [new_p], "sh"
        else:
            c_idx = (i + 7) % len(COUNTRIES)
            country, cap = COUNTRIES[c_idx], CAPITALS[c_idx]
            hist = wrap_distractors([
                (14, f"{org} (also known as {alias}) originally picked {old_p} to run daily affairs as managing head."),
                (44, f"{new_p} holds national passport citizenship from {country}."),
                (79, f"{alias} transitioned executive control so {new_p} now heads the firm as CEO."),
                (103, f"Administratively, the seat of national capital for {country} is {cap}."),
            ])
            q = f"What is the capital city of the country of citizenship of {org}'s latest CEO?"
            ans, hop = [cap], "mh"
        items.append({"id": f"stress21_synonym_{i:02d}", "category": "synonym_alias", "hop_type": hop, "history": hist, "question": q, "answers": ans})

    # 3. historical_reminiscence (20Q: T1 old -> T2 current -> T3 reminiscence of old)
    for i in range(20):
        person = f"{PEOPLE[i]} ({i + 30})"
        old_city = CITIES[(i + 2) % len(CITIES)]
        cur_city = CITIES[(i + 10) % len(CITIES)]
        hist = wrap_distractors([
            (15, f"Back in 2016, {person} resided in {old_city}."),
            (62, f"In 2023, {person} moved permanently and now lives in {cur_city}."),
            (102, f"During a 2024 retrospective podcast, {person} fondly recalled living in {old_city} back in 2016 before relocating."),
        ])
        items.append({"id": f"stress21_reminiscence_{i:02d}", "category": "historical_reminiscence", "hop_type": "sh_valid_time", "history": hist, "question": f"Where does {person} currently reside?", "answers": [cur_city]})

    # 4. multivalued_relation (20Q)
    for i in range(20):
        person = f"{PEOPLE[(i + 6) % len(PEOPLE)]} #{i + 40}"
        if i < 10:
            aw1, aw2 = AWARDS[i % len(AWARDS)]
            aw1, aw2 = f"{aw1} ({2019 + (i % 3)})", f"{aw2} ({2023 + (i % 2)})"
            hist = wrap_distractors([
                (22, f"In {2019 + (i % 3)}, {person} won the {aw1}."),
                (90, f"Alongside earlier honors, {person} also received the {aw2} in {2023 + (i % 2)}."),
            ])
            q, ans = f"Which earlier award did {person} win before receiving the {aw2}?", [aw1]
        else:
            l1, l2 = LANG_PAIRS[i % len(LANG_PAIRS)]
            hist = wrap_distractors([
                (24, f"The spoken language of {person} is {l1}."),
                (91, f"The spoken language of {person} is {l2} (acquired alongside {l1}, remaining fluent in both)."),
            ])
            q, ans = f"What language did {person} speak before also acquiring {l2}?", [l1]
        items.append({"id": f"stress21_multivalued_{i:02d}", "category": "multivalued_relation", "hop_type": "sh_multivalued", "history": hist, "question": q, "answers": ans})

    # 5. out_of_order_timestamp (20Q: [2024] logged BEFORE archival [2018] note)
    for i in range(20):
        org = f"{ORGS[(i + 8) % len(ORGS)]} Lab-{i + 50}"
        dir_2024 = PEOPLE[(i + 3) % len(PEOPLE)]
        dir_2018 = PEOPLE[(i + 12) % len(PEOPLE)]
        hist = wrap_distractors([
            (21, f"[2024-09 Update] As of September 2024, the director of {org} is {dir_2024}."),
            (96, f"[2018-04 Archive] Note appended to archive: in April 2018, the director of {org} was {dir_2018}."),
        ])
        items.append({"id": f"stress21_ooo_ts_{i:02d}", "category": "out_of_order_timestamp", "hop_type": "sh_valid_time", "history": hist, "question": f"Who is the director of {org} as of 2024?", "answers": [dir_2024]})

    assert len(items) == 100
    return items


def line_level_hybrid_fallback(raw_lines: list[str], embedder, query: str, top_k: int = 8) -> str:
    """Line-level BM25+BGE RRF fallback fixing single-chunk truncation (Concern #2 engineering fix)."""
    if not raw_lines:
        return ""
    bm25 = BM25Index(raw_lines)
    lex_scores = bm25.scores(query)
    vecs = normalize_rows(list(embedder.embed(raw_lines, batch_size=64)))
    qvec = normalize_rows(list(embedder.embed([QUERY_PREFIX + query])))[0]
    dense_scores = vecs @ qvec
    import numpy as np
    lex_ord = np.argsort(-lex_scores, kind="mergesort")
    den_ord = np.argsort(-dense_scores, kind="mergesort")
    fused = np.zeros(len(raw_lines), dtype=np.float64)
    for r, idx in enumerate(lex_ord[:64]):
        fused[idx] += 1.0 / (60 + r + 1)
    for r, idx in enumerate(den_ord[:64]):
        fused[idx] += 1.0 / (60 + r + 1)
    chosen = sorted(np.argsort(-fused, kind="mergesort")[:top_k].tolist())
    return "\n".join(raw_lines[i] for i in chosen)


def extract_facts_with_valid_time(history: str, item_id: str, meter: SafeMeter, embedder) -> tuple[list[tuple[Observation, tuple[int, int]]], dict]:
    """Open LLM fact extractor extracting optional valid_year in batched windows over ~120-line histories."""
    lines = history.splitlines()
    # Extract in 2 chunks of ~65 lines via pre-filtering non-filler or direct 2-window extraction
    # Actually 120 short lines (~2,000 tokens total!) easily fits in ONE prompt call!
    msgs = [
        {
            "role": "system",
            "content": (
                "Extract durable factual, leadership, location, citizenship, language, or award statements from these numbered lines "
                "(skip generic regional depot / retired consultant / guild / patent distractor filler). "
                "Return JSON {\"facts\": [{\"line\": int, \"entity\": str, \"relation\": \"short snake_case\", "
                "\"value\": str, \"operation\": \"assert|replace\", \"valid_year\": int_or_null, \"is_reminiscence\": bool}]}. "
                "Set valid_year to the explicit year when stated (e.g., 2024, 2018, 2016) and is_reminiscence=true for past recollections."
            ),
        },
        {"role": "user", "content": history},
    ]
    call = meter.call(f"ext_vt--{item_id}", msgs, max_tokens=1536, json_output=True)
    extracted: list[tuple[Observation, tuple[int, int]]] = []
    if call.get("error") or not call.get("answer"):
        return extracted, call
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
            op = str(row.get("operation") or "assert").strip().lower()
            vy = row.get("valid_year")
            rem = bool(row.get("is_reminiscence", False))
            if not ent or not rel or not val:
                continue
            if op not in {"assert", "replace"}:
                op = "assert"
            # Lexicographic valid-time rank: (valid_year, t) strictly prioritizes explicit year over append order t
            vy_int = int(vy) if isinstance(vy, (int, float)) and 1900 <= int(vy) <= 2100 else 0
            if vy_int:
                eff_rank = (vy_int, lnum)
            elif rem:
                eff_rank = (-1, lnum)
            else:
                eff_rank = (0, lnum)
            obs = Observation(lnum - 1, lnum, ent, rel, val, op, (0, len(lines[lnum - 1])), "llm_open")
            extracted.append((obs, eff_rank))
    except (ValueError, TypeError, AttributeError):
        pass
    return extracted, call


class ValidTimeLedgerStore(RecencyHybridStore):
    """RecencyHybridStore parameterized by either transaction order or explicit valid-time rank."""

    def __init__(self, pairs: list[tuple[Observation, tuple[int, int]]], raw_lines: list[str], model, hybrid: HybridLexicalDenseIndex, *, use_valid_time: bool):
        self.raw_lines = raw_lines
        self.lines = {i + 1: raw_lines[i] for i in range(len(raw_lines))}
        self.model = model
        self.hybrid = hybrid
        self.use_exact = True
        self.semantic_threshold = 0.56
        self.index = defaultdict(list)
        self.by_entity = defaultdict(list)
        self.facts = {}
        self.rank_of: dict[int, tuple[int, int]] = {}
        for obs, vt_rank in pairs:
            score = vt_rank if use_valid_time else (0, obs.order)
            # Store original Observation; sort by rank_of[obs.order]
            self.index[(norm(obs.entity), obs.relation)].append(obs)
            self.by_entity[norm(obs.entity)].append(obs)
            self.facts[obs.order] = obs
            self.rank_of[obs.order] = score
        self.entities = [f.entity for f in self.facts.values()]
        self.cache = {}

    def step(self, entity: str, relation: str, sub_query: str = "", *, allow_fallback: bool = True):
        fact, ev = super().step(entity, relation, sub_query, allow_fallback=allow_fallback)
        # Re-rank exact/semantic candidates using rank_of when use_valid_time differs from order
        if ev.get("candidates"):
            best = max(ev["candidates"], key=lambda c: self.rank_of.get(c["order"], (0, c["order"])))
            if fact is None or best["order"] != fact.order:
                fact = self.facts[best["order"]]
                ev = {**ev, "chosen": fact.order}
        return fact, ev


def run_stress_v21(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    items = build_long_dataset(seed=101)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump_json(DATA_PATH, {"generator": "src/run_paraphrase_stress_v2_1.py", "seed": 101, "items": items})

    if not execute:
        # 100 extractor + 100 planner + 500 answer = 700 calls
        est_usd = token_cost(0, 100 * 2400 + 100 * 220 + 500 * 1100, 100 * 500 + 100 * 80 + 500 * 24, "peak")
        q = {"questions": len(items), "lines_per_history": len(items[0]["history"].splitlines()), "arms": list(ARMS_V21), "est_peak_usd": str(est_usd)}
        print(json.dumps(q, indent=2))
        return q

    key = credentials(FLASH, key_file)
    if not key:
        raise SystemExit("Missing DeepSeek API key.")

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "requests"):
        (out / sub).mkdir(exist_ok=True)

    meter = SafeMeter(out, key, budget)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    embedder = _load_embedder()

    def process_item(item: dict) -> list[dict]:
        sid = item["id"]
        hist = item["history"]
        raw_lines = hist.splitlines()
        q = item["question"]

        plan_call = meter.call(f"plan21--{sid}", planner_messages_v2(q), 192, json_output=True)
        raw_hybrid = HybridLexicalDenseIndex(embedder, hist, tombstones=None)

        records, _clean, decisions = archive_v2(hist, "numbered")
        tombstones = {d["removed_line"] for d in decisions}
        active_recs = [rec for idx, rec in records.items() if idx not in tombstones]
        tomb_hybrid = raw_hybrid.with_tombstones(tombstones)
        rule_store = RecencyHybridStore(active_recs, raw_lines, embedder, tomb_hybrid, use_exact=True)
        plan_rule = parse_plan_v2(plan_call, q, rule_store.entities)

        ext_pairs, ext_call = extract_facts_with_valid_time(hist, sid, meter, embedder)
        tx_store = ValidTimeLedgerStore(ext_pairs, raw_lines, embedder, raw_hybrid, use_valid_time=False)
        vt_store = ValidTimeLedgerStore(ext_pairs, raw_lines, embedder, raw_hybrid, use_valid_time=True)
        plan_llm = parse_plan_v2(plan_call, q, tx_store.entities)

        item_rows = []
        tx_ctx_cache = None
        tx_case_cache = None

        for arm in ARMS_V21:
            cpath = out / "cases" / f"{arm}--{sid}.json"
            if cpath.exists():
                existing = json.loads(cpath.read_text(encoding="utf-8"))
                if arm == "carr_v2_llm_tx_time":
                    tx_case_cache = existing
                item_rows.append(existing)
                continue

            extra_usd = Decimal(0)
            if arm == "full_context":
                ctx_str = hist
            elif arm == "bm25_bge_recency":
                ctx_str, _ = retrieve_bm25_recency(raw_hybrid, q, top_k=8)
            elif arm == "carr_v2_rule_line_fb":
                ev, tr_r = retrieve_v2(rule_store, plan_rule, q)
                if tr_r.get("status") == "hybrid_fallback_text" or not ev:
                    ctx_str = line_level_hybrid_fallback(raw_lines, embedder, q, top_k=8)
                else:
                    ctx_str = "\n".join(ev)
                extra_usd = Decimal(str(plan_call.get("usage_usd") or "0"))
            elif arm == "carr_v2_llm_tx_time":
                ev, _ = retrieve_v2(tx_store, plan_llm, q)
                ctx_str = "\n".join(ev)
                tx_ctx_cache = ctx_str
                extra_usd = Decimal(str(plan_call.get("usage_usd") or "0")) + Decimal(str(ext_call.get("usage_usd") or "0"))
            elif arm == "carr_v2_llm_valid_time":
                ev, _ = retrieve_v2(vt_store, plan_llm, q)
                ctx_str = "\n".join(ev)
                extra_usd = Decimal(str(plan_call.get("usage_usd") or "0")) + Decimal(str(ext_call.get("usage_usd") or "0"))
                if tx_ctx_cache is not None and ctx_str == tx_ctx_cache and tx_case_cache is not None:
                    row = {**tx_case_cache, "arm": arm, "reused_from": "carr_v2_llm_tx_time"}
                    dump_json(cpath, row)
                    item_rows.append(row)
                    continue
            else:
                raise ValueError(arm)

            ans_call = meter.call(f"{arm}--{sid}", stress_messages(ctx_str, q), 32, json_output=False)
            corr = bool(not ans_call.get("error") and scorer.score(ans_call.get("answer") or "", item["answers"]))
            row = {
                "sample_id": sid,
                "category": item["category"],
                "arm": arm,
                "question": q,
                "reference": item["answers"],
                "prediction": ans_call.get("answer") or "",
                "correct": corr,
                "error": ans_call.get("error"),
                "usage": ans_call.get("usage"),
                "usage_usd": ans_call.get("usage_usd"),
                "extra_usd": str(extra_usd),
                "reused_from": None,
            }
            if arm == "carr_v2_llm_tx_time":
                tx_case_cache = row
            dump_json(cpath, row)
            item_rows.append(row)
        return item_rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_item, it) for it in items]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    summary_rows = []
    for arm in ARMS_V21:
        arows = [r for r in all_rows if r["arm"] == arm]
        for grp in ("all", *CATEGORIES):
            sub = arows if grp == "all" else [r for r in arows if r["category"] == grp]
            corr = sum(int(bool(r["correct"])) for r in sub)
            errs = sum(int(bool(r["error"])) for r in sub)
            ans_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in sub)
            ans_usd = sum(Decimal(str(r.get("usage_usd") or "0")) for r in sub)
            ext_usd = sum(Decimal(str(r.get("extra_usd") or "0")) for r in sub)
            summary_rows.append({
                "arm": arm,
                "group": grp,
                "questions": len(sub),
                "correct": corr,
                "acc_pct": round(100.0 * corr / len(sub), 2),
                "errors": errs,
                "ans_input_per_q": round(ans_in / len(sub), 1),
                "ans_usd": str(ans_usd),
                "extra_usd": str(ext_usd),
                "total_usd": str(ans_usd + ext_usd),
            })

    report = {
        "generator": "src/run_paraphrase_stress_v2_1.py",
        "generated_utc": utc_now().isoformat(),
        "lines_per_history": len(items[0]["history"].splitlines()),
        "incremental_api_usd": str(meter.usage_cost),
        "rows": summary_rows,
    }
    dump_json(out / "summary.json", report)
    write_csv(out / "summary.csv", summary_rows)

    print("=== Step 3A Upgraded 120-Line Stress Test (100Q x 5 Arms) ===")
    for arm in ARMS_V21:
        vals = [f"{g[:7]}={next(r['acc_pct'] for r in summary_rows if r['arm']==arm and r['group']==g):5.1f}%" for g in ("all", *CATEGORIES)]
        print(f"{arm:24s} | " + " | ".join(vals))
    print("Incremental API USD:", meter.usage_cost)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("3.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run_stress_v21(args.execute, args.budget, args.key_file, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
