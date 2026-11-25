"""Controlled Non-Templated Paraphrase & Complex Temporal Stress Test (100Q x 4 arms).

Addresses Peer Review §3.3 (template overfitting beyond 'The <rel> of <ent> is <val>')
and §3.9 (Transaction Time vs. Valid Time, historical reminiscence, multi-valued relations,
and out-of-order timestamps).

Categories (20Q each, 100Q total, each with 36-line numbered history containing distractors):
  1. passive_inversion       (20Q: 10 SH + 10 MH passive/inverted natural language paraphrases)
  2. synonym_alias           (20Q: 10 SH + 10 MH synonymous predicates & entity surface forms)
  3. historical_reminiscence (20Q: T1 old -> T2 current -> T3 reminiscence of old state)
  4. multivalued_relation    (20Q: additive non-overwriting attributes / awards / languages)
  5. out_of_order_timestamp  (20Q: [2024] logged before archival [2019] record)

Arms (all on DeepSeek Flash, T=0):
  - full_context         : Full ~40-line history passed directly
  - bm25_bge_recency     : BM25+BGE RRF + log-order recency
  - carr_v2_rule         : parse_numbered_v2 + Unified Planner v2 + 3-channel relay
  - carr_v2_llm_extract  : Open-domain LLM atomic fact extractor + Unified Planner/Recency Store
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
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.carr_v2.open_parser import archive_v2
from src.carr_v2.semantic_router import RecencyHybridStore, parse_plan_v2, planner_messages_v2, retrieve_v2
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex
from src.evaluator import LongContext, dump_json, token_cost, utc_now
from src.factual_v2 import Observation, Record, canon, norm
from src.run_carr_v2 import SafeMeter
from src.run_v2 import BGE_FOLDER
from src.v2_report import write_csv
from src.v2_retrieval import BGE
from src.v2_transport import FLASH, credentials

DATA_PATH = ROOT / "data" / "paraphrase_stress_v2" / "dataset.json"
DEFAULT_OUT = ROOT / "results" / "test" / "paraphrase-stress-v2"

CATEGORIES = (
    "passive_inversion",
    "synonym_alias",
    "historical_reminiscence",
    "multivalued_relation",
    "out_of_order_timestamp",
)
STRESS_ARMS = ("full_context", "bm25_bge_recency", "carr_v2_rule", "carr_v2_llm_extract")

CITIES = [
    "Zurich", "Kyoto", "Lisbon", "Oslo", "Prague", "Montreal", "Valencia", "Tbilisi",
    "Bergen", "Nagoya", "Salzburg", "Porto", "Gdansk", "Tallinn", "Dunedin", "Ghent",
    "Innsbruck", "Turin", "Aarhus", "Brno", "Krakow", "Malmo", "Fukuoka", "Bologna",
]
PEOPLE = [
    "Elena Rostova", "Marcus Vance", "Kenji Takahashi", "Sora Lindqvist", "Amara Okafor",
    "Tariq Al-Mansoor", "Nadia Kowalski", "Henrik Solberg", "Priya Subramanian", "Lucas Ferreira",
    "Mei-Ling Zhou", "Dario Conti", "Amina Diallo", "Viktor Novak", "claire Beaumont",
    "Mateo Herrera", "Hana Varga", "Rohan Kulkarni", "Freja Lind", "Samir Haddad",
]
PEOPLE = [p.title() if p[0].islower() else p for p in PEOPLE]
ORGS = [
    "Helios Labs", "Nordic Rail", "Veridian Press", "Caspian Bio", "Asteria Robotics",
    "Solstice Energy", "Boreal Grid", "Mariner Media", "Aegis Cloud", "Kestrel Works",
    "Orion Micro", "Halcyon Bio", "Vertex Transit", "Vanguard Optics", "Summit Aero",
    "Stratos Dynamics", "Meridian Pharma", "Cobalt Telecom", "Crescent Logistics", "Pulse Systems",
]
COUNTRIES = [
    "Norway", "Portugal", "Estonia", "Austria", "Slovenia", "Croatia", "Finland", "Denmark",
    "Belgium", "Czechia", "Hungary", "Uruguay", "Chile", "Morocco", "Tunisia", "Georgia",
    "Iceland", "Slovakia", "Lithuania", "Latvia",
]
CAPITALS = [
    "Oslo", "Lisbon", "Tallinn", "Vienna", "Ljubljana", "Zagreb", "Helsinki", "Copenhagen",
    "Brussels", "Prague", "Budapest", "Montevideo", "Santiago", "Rabat", "Tunis", "Tbilisi",
    "Reykjavik", "Bratislava", "Vilnius", "Riga",
]
AWARDS = [
    ("Turing Bronze Medal", "Aurora Research Prize"),
    ("Edison Innovation Cup", "Lovelace Fellowship"),
    ("Kepler Science Award", "Maxwell Systems Medal"),
    ("Curie Discovery Prize", "Helmholtz Honor"),
    ("Gauss Computing Medal", "Fourier Signal Prize"),
]
LANG_PAIRS = [
    ("Mandarin", "Portuguese"), ("Arabic", "Finnish"), ("Japanese", "Polish"),
    ("Turkish", "Swedish"), ("Korean", "Dutch"), ("Swahili", "Hungarian"),
    ("Greek", "Vietnamese"), ("Czech", "Indonesian"), ("Norwegian", "Thai"), ("Estonian", "Welsh"),
]


def _make_distractors(rng: random.Random, count: int) -> list[str]:
    out = []
    for k in range(count):
        p = f"Distractor Person {k + 101}"
        c = CITIES[(k * 7 + 3) % len(CITIES)]
        o = ORGS[(k * 5 + 1) % len(ORGS)]
        t = k % 4
        if t == 0:
            out.append(f"By administrative decree, {c} serves as the regional depot for {o} Annex-{k}.")
        elif t == 1:
            out.append(f"{p} briefly consulted at {o} Division-{k} before retiring.")
        elif t == 2:
            out.append(f"A historical note records that {o} Unit-{k} collaborated with {c} Guild.")
        else:
            out.append(f"In an unrelated filing, {p} registered a patent in {c}.")
    return out


def build_dataset(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []

    # 1. passive_inversion (20Q: 10 SH + 10 MH)
    for i in range(20):
        org = f"{ORGS[i]} {i + 10}"
        old_city, new_city = CITIES[i], CITIES[(i + 7) % len(CITIES)]
        old_ceo, new_ceo = PEOPLE[i], PEOPLE[(i + 5) % len(PEOPLE)]
        distractors = _make_distractors(rng, 28)
        if i < 10:
            lines = (
                distractors[:8]
                + [f"Originally, {old_city} served as the headquarters city where {org} operated."]
                + distractors[8:20]
                + [f"Following a board vote, {new_city} was formally designated as the headquarters city of {org}."]
                + distractors[20:]
            )
            q = f"What is the latest headquarters city of {org}?"
            ans = [new_city]
            hop = "sh"
        else:
            c_idx = (i + 3) % len(COUNTRIES)
            country, cap_old, cap_new = COUNTRIES[c_idx], CITIES[(i + 11) % len(CITIES)], CAPITALS[c_idx]
            lines = (
                distractors[:6]
                + [f"{old_ceo} was initially appointed to serve as the chief executive officer of {org}."]
                + distractors[6:14]
                + [f"Under updated filings, citizenship of {new_ceo} is held in {country}."]
                + distractors[14:20]
                + [f"It was announced that {cap_old} formerly acted as the capital of {country}."]
                + distractors[20:24]
                + [f"At the latest annual meeting, leadership passed and {new_ceo} became the chief executive officer of {org}."]
                + distractors[24:]
                + [f"According to the revised charter, {cap_new} stands as the capital of {country}."]
            )
            q = f"What is the capital of the country of citizenship of the chief executive officer of {org}?"
            ans = [cap_new]
            hop = "mh"
        numbered = "\n".join(f"{idx + 1}. {txt}" for idx, txt in enumerate(lines))
        items.append({
            "id": f"stress_passive_{i:02d}",
            "category": "passive_inversion",
            "hop_type": hop,
            "history": numbered,
            "question": q,
            "answers": ans,
        })

    # 2. synonym_alias (20Q: 10 SH + 10 MH)
    for i in range(20):
        org_base = ORGS[(i + 4) % len(ORGS)]
        org = f"{org_base} Group {i + 20}"
        alias = f"{org_base.split()[0]}-{i + 20}"
        old_p, new_p = PEOPLE[(i + 2) % len(PEOPLE)], PEOPLE[(i + 9) % len(PEOPLE)]
        old_c, new_c = CITIES[(i + 1) % len(CITIES)], CITIES[(i + 13) % len(CITIES)]
        distractors = _make_distractors(rng, 28)
        if i < 10:
            lines = (
                distractors[:9]
                + [f"{org} (shortened as {alias}) used to be helmed by {old_p} as its top executive."]
                + distractors[9:21]
                + [f"In the newest filing, {alias} is now led at the top executive helm by {new_p}."]
                + distractors[21:]
            )
            q = f"Who is the current chief executive officer of {org}?"
            ans = [new_p]
            hop = "sh"
        else:
            c_idx = (i + 7) % len(COUNTRIES)
            country, cap = COUNTRIES[c_idx], CAPITALS[c_idx]
            lines = (
                distractors[:7]
                + [f"{org} (also known as {alias}) originally picked {old_p} to run daily affairs as managing head."]
                + distractors[7:15]
                + [f"{new_p} holds national passport citizenship from {country}."]
                + distractors[15:22]
                + [f"{alias} transitioned executive control so {new_p} now heads the firm as CEO."]
                + distractors[22:]
                + [f"Administratively, the seat of national capital for {country} is {cap}."]
            )
            q = f"What is the capital city of the country of citizenship of {org}'s latest CEO?"
            ans = [cap]
            hop = "mh"
        numbered = "\n".join(f"{idx + 1}. {txt}" for idx, txt in enumerate(lines))
        items.append({
            "id": f"stress_synonym_{i:02d}",
            "category": "synonym_alias",
            "hop_type": hop,
            "history": numbered,
            "question": q,
            "answers": ans,
        })

    # 3. historical_reminiscence (20Q: T1 old -> T2 current -> T3 reminiscence of old)
    for i in range(20):
        person = f"{PEOPLE[i]} ({i + 30})"
        old_city = CITIES[(i + 2) % len(CITIES)]
        cur_city = CITIES[(i + 10) % len(CITIES)]
        distractors = _make_distractors(rng, 28)
        lines = (
            distractors[:6]
            + [f"Back in 2016, {person} resided in {old_city}."]
            + distractors[6:16]
            + [f"In 2023, {person} moved permanently and now lives in {cur_city}."]
            + distractors[16:25]
            + [f"During a 2024 retrospective podcast, {person} fondly recalled living in {old_city} back in 2016 before relocating."]
            + distractors[25:]
        )
        numbered = "\n".join(f"{idx + 1}. {txt}" for idx, txt in enumerate(lines))
        items.append({
            "id": f"stress_reminiscence_{i:02d}",
            "category": "historical_reminiscence",
            "hop_type": "sh_valid_time",
            "history": numbered,
            "question": f"Where does {person} currently reside?",
            "answers": [cur_city],
        })

    # 4. multivalued_relation (20Q: additive multi-valued facts; earlier item must NOT be erased)
    for i in range(20):
        person = f"{PEOPLE[(i + 6) % len(PEOPLE)]} #{i + 40}"
        distractors = _make_distractors(rng, 28)
        if i < 10:
            aw1, aw2 = AWARDS[i % len(AWARDS)]
            aw1, aw2 = f"{aw1} ({2019 + (i % 3)})", f"{aw2} ({2023 + (i % 2)})"
            lines = (
                distractors[:8]
                + [f"In {2019 + (i % 3)}, {person} won the {aw1}."]
                + distractors[8:20]
                + [f"Alongside earlier honors, {person} also received the {aw2} in {2023 + (i % 2)}."]
                + distractors[20:]
            )
            q = f"Which earlier award did {person} win before receiving the {aw2}?"
            ans = [aw1]
        else:
            l1, l2 = LANG_PAIRS[i % len(LANG_PAIRS)]
            lines = (
                distractors[:8]
                + [f"The spoken language of {person} is {l1}."]
                + distractors[8:20]
                + [f"The spoken language of {person} is {l2} (acquired alongside {l1}, remaining fluent in both)."]
                + distractors[20:]
            )
            q = f"What language did {person} speak before also acquiring {l2}?"
            ans = [l1]
        numbered = "\n".join(f"{idx + 1}. {txt}" for idx, txt in enumerate(lines))
        items.append({
            "id": f"stress_multivalued_{i:02d}",
            "category": "multivalued_relation",
            "hop_type": "sh_multivalued",
            "history": numbered,
            "question": q,
            "answers": ans,
        })

    # 5. out_of_order_timestamp (20Q: [2024] logged BEFORE archival [2018] note)
    for i in range(20):
        org = f"{ORGS[(i + 8) % len(ORGS)]} Lab-{i + 50}"
        dir_2024 = PEOPLE[(i + 3) % len(PEOPLE)]
        dir_2018 = PEOPLE[(i + 12) % len(PEOPLE)]
        distractors = _make_distractors(rng, 28)
        lines = (
            distractors[:8]
            + [f"[2024-09 Update] As of September 2024, the director of {org} is {dir_2024}."]
            + distractors[8:20]
            + [f"[2018-04 Archive] Note appended to archive: in April 2018, the director of {org} was {dir_2018}."]
            + distractors[20:]
        )
        numbered = "\n".join(f"{idx + 1}. {txt}" for idx, txt in enumerate(lines))
        items.append({
            "id": f"stress_ooo_ts_{i:02d}",
            "category": "out_of_order_timestamp",
            "hop_type": "sh_valid_time",
            "history": numbered,
            "question": f"Who is the director of {org} as of 2024?",
            "answers": [dir_2024],
        })

    assert len(items) == 100
    return items


def llm_extract_open_facts(history: str, item_id: str, meter: SafeMeter) -> tuple[list[Observation], list[dict]]:
    """Open-domain atomic fact extraction mirroring LongMemEval session extraction."""
    msgs = [
        {
            "role": "system",
            "content": (
                "Extract key entity-relation-value facts from these numbered statements (ignore generic filler). "
                "Return JSON {\"facts\": [{\"line\": int, \"entity\": str, \"relation\": \"short snake_case label\", "
                "\"value\": str, \"operation\": \"assert|replace\"}]}. Use replace when updating a prior role/location."
            ),
        },
        {"role": "user", "content": history},
    ]
    call = meter.call(f"extract--{item_id}", msgs, max_tokens=1024, json_output=True)
    raw_lines = history.splitlines()
    obs_list: list[Observation] = []
    if call.get("error") or not call.get("answer"):
        return obs_list, [call]
    try:
        data = json.loads(call["answer"])
        for row in data.get("facts", []):
            if not isinstance(row, dict):
                continue
            line_num = row.get("line")
            if not isinstance(line_num, int) or not (1 <= line_num <= len(raw_lines)):
                continue
            ent = str(row.get("entity") or "").strip()
            rel = canon(str(row.get("relation") or "").strip())
            val = str(row.get("value") or "").strip()
            op = str(row.get("operation") or "assert").strip().lower()
            if not ent or not rel or not val:
                continue
            if op not in {"assert", "replace"}:
                op = "assert"
            obs_list.append(Observation(line_num - 1, line_num, ent, rel, val, op, (0, len(raw_lines[line_num - 1])), "llm_open"))
    except (ValueError, TypeError, AttributeError):
        pass
    return obs_list, [call]


class OpenLedgerStore(RecencyHybridStore):
    """RecencyHybridStore populated with LLM-extracted Observations instead of regex parse_numbered_v2."""

    def __init__(self, observations: list[Observation], raw_lines: list[str], model, hybrid: HybridLexicalDenseIndex):
        records = [Record(i, raw_lines[i], 0, i + 1) for i in range(len(raw_lines))]
        # Initialize parent attributes without regex parsing
        self.raw_lines = raw_lines
        self.lines = {r.order: raw_lines[r.index] for r in records}
        self.model = model
        self.hybrid = hybrid
        self.use_exact = True
        self.semantic_threshold = 0.56
        from src.factual_v2 import norm as _norm
        self.index = defaultdict(list)
        self.by_entity = defaultdict(list)
        self.facts = {}
        for f in observations:
            self.index[(_norm(f.entity), f.relation)].append(f)
            self.by_entity[_norm(f.entity)].append(f)
            self.facts[f.order] = f
        self.entities = [f.entity for f in self.facts.values()]
        self.cache = {}


def stress_messages(history_ctx: str, question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "Answer the question concisely based strictly on the provided context. Give only the final answer.",
        },
        {
            "role": "user",
            "content": f"Context:\n{history_ctx}\n\nQuestion: {question}\nAnswer:",
        },
    ]


def run_stress(execute: bool, budget: Decimal, key_file: Path | None, out: Path, workers: int) -> dict:
    items = build_dataset(seed=42)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump_json(DATA_PATH, {"generator": "src/run_paraphrase_stress_v2.py", "seed": 42, "items": items})

    if not execute:
        # 100 extractor + 100 planner + 400 answer = 600 calls
        est_usd = token_cost(0, 100 * 900 + 100 * 220 + 400 * 500, 100 * 400 + 100 * 80 + 400 * 24, "peak")
        q = {"questions": len(items), "categories": list(CATEGORIES), "arms": list(STRESS_ARMS), "est_peak_usd": str(est_usd)}
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
    from threading import Lock
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
        local_files_only=True, specific_model_path=str(BGE_FOLDER),
    )
    embed_lock = Lock()

    class LockedEmbed:
        def embed(self, texts, batch_size=32):
            with embed_lock:
                return list(embedder.embed(texts, batch_size=batch_size))

    safe_emb = LockedEmbed()

    def process_item(item: dict) -> list[dict]:
        sid = item["id"]
        hist = item["history"]
        raw_lines = hist.splitlines()
        q = item["question"]

        # Shared planner call
        plan_call = meter.call(f"plan_stress--{sid}", planner_messages_v2(q), 192, json_output=True)
        # Rule store
        records, _clean, decisions = archive_v2(hist, "numbered")
        tombstones = {d["removed_line"] for d in decisions}
        active_recs = [rec for idx, rec in records.items() if idx not in tombstones]
        raw_hybrid = HybridLexicalDenseIndex(safe_emb, hist, tombstones=None)
        tomb_hybrid = raw_hybrid.with_tombstones(tombstones)
        rule_store = RecencyHybridStore(active_recs, raw_lines, safe_emb, tomb_hybrid, use_exact=True)
        plan = parse_plan_v2(plan_call, q, rule_store.entities)

        # LLM open extractor store
        obs_llm, ext_calls = llm_extract_open_facts(hist, sid, meter)
        llm_store = OpenLedgerStore(obs_llm, raw_lines, safe_emb, raw_hybrid)
        plan_llm = parse_plan_v2(plan_call, q, llm_store.entities)

        item_rows = []
        for arm in STRESS_ARMS:
            cpath = out / "cases" / f"{arm}--{sid}.json"
            if cpath.exists():
                item_rows.append(json.loads(cpath.read_text(encoding="utf-8")))
                continue

            extra_usd = Decimal(0)
            if arm == "full_context":
                ctx_str = hist
            elif arm == "bm25_bge_recency":
                ctx_str, _ = retrieve_bm25_recency(raw_hybrid, q)
            elif arm == "carr_v2_rule":
                ev, _ = retrieve_v2(rule_store, plan, q)
                ctx_str = "\n".join(ev)
                extra_usd = Decimal(str(plan_call.get("usage_usd") or "0"))
            elif arm == "carr_v2_llm_extract":
                ev, _ = retrieve_v2(llm_store, plan_llm, q)
                ctx_str = "\n".join(ev)
                extra_usd = Decimal(str(plan_call.get("usage_usd") or "0")) + sum(
                    Decimal(str(c.get("usage_usd") or "0")) for c in ext_calls
                )
            else:
                raise ValueError(arm)

            ans_call = meter.call(f"{arm}--{sid}", stress_messages(ctx_str, q), 32, json_output=False)
            correct = bool(not ans_call.get("error") and scorer.score(ans_call.get("answer") or "", item["answers"]))
            row = {
                "sample_id": sid,
                "category": item["category"],
                "hop_type": item["hop_type"],
                "arm": arm,
                "question": q,
                "reference": item["answers"],
                "prediction": ans_call.get("answer") or "",
                "correct": correct,
                "error": ans_call.get("error"),
                "usage": ans_call.get("usage"),
                "usage_usd": ans_call.get("usage_usd"),
                "extra_usd": str(extra_usd),
                "elapsed_ms": ans_call.get("elapsed_ms"),
            }
            dump_json(cpath, row)
            item_rows.append(row)
        return item_rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_item, it) for it in items]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    summary_rows = []
    for arm in STRESS_ARMS:
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
        "generator": "src/run_paraphrase_stress_v2.py",
        "generated_utc": utc_now().isoformat(),
        "total_api_usd": str(meter.usage_cost),
        "rows": summary_rows,
    }
    dump_json(out / "summary.json", report)
    write_csv(out / "summary.csv", summary_rows)

    print("=== Phase 3 Paraphrase & Complex Temporal Stress Test (100Q) ===")
    for arm in STRESS_ARMS:
        vals = [f"{g[:7]}={next(r['acc_pct'] for r in summary_rows if r['arm']==arm and r['group']==g):5.1f}%" for g in ("all", *CATEGORIES)]
        print(f"{arm:20s} | " + " | ".join(vals))
    print("Total API USD:", meter.usage_cost)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget", type=Decimal, default=Decimal("2.0"))
    ap.add_argument("--key-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    run_stress(args.execute, args.budget, args.key_file, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
