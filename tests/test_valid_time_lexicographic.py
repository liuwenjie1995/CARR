"""Tests verifying lexicographic valid-time ranking above 10,000 lines and stress-test equivalence."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from src.factual_v2 import Observation, canon, norm
from src.run_paraphrase_stress_v2_1 import ValidTimeLedgerStore, extract_facts_with_valid_time

ROOT = Path(__file__).resolve().parent.parent


class _CachedMeter:
    def __init__(self, req_dir: Path):
        self.req_dir = req_dir

    def call(self, request_id: str, messages, max_tokens: int, *, json_output: bool = False):
        return json.loads((self.req_dir / f"{request_id}.json").read_text(encoding="utf-8"))


class _DummyEmbedder:
    def embed(self, texts, batch_size=32):
        raise AssertionError("Offline test should not invoke embeddings.")


class _DummyHybrid:
    pass


def _legacy_numeric_rank(vy, rem: bool, lnum: int) -> int:
    vy_int = int(vy) if isinstance(vy, (int, float)) and 1900 <= int(vy) <= 2100 else 0
    eff = (vy_int * 10000 + lnum) if vy_int else lnum
    if rem and not vy_int:
        eff = -1000 + lnum
    return eff


def test_lexicographic_valid_time_exceeds_10k_lines():
    # At t > 10,000, numeric vy*10000+t could overflow year differences (e.g. 2018*10000 + 80000 > 2024*10000 + 21).
    # Lexicographic tuple (tau, t) strictly prioritizes valid year for arbitrarily long histories.
    obs_2024 = Observation(20, 21, "Lab-50", "director", "Alice", "assert", (0, 10), "llm_open")
    obs_2018 = Observation(79999, 80000, "Lab-50", "director", "Bob", "assert", (0, 10), "llm_open")
    obs_rem = Observation(99998, 99999, "Lab-50", "director", "Carol", "assert", (0, 10), "llm_open")

    pairs = [
        (obs_2024, (2024, 21)),
        (obs_2018, (2018, 80000)),
        (obs_rem, (-1, 99999)),
    ]
    vt_store = ValidTimeLedgerStore(pairs, [""] * 80000, _DummyEmbedder(), _DummyHybrid(), use_valid_time=True)
    chosen, _ = vt_store.step("Lab-50", "director", allow_fallback=False)
    assert chosen is not None
    assert chosen.value == "Alice"
    assert vt_store.rank_of[21] > vt_store.rank_of[80000] > vt_store.rank_of[99999]

    tx_store = ValidTimeLedgerStore(pairs, [""] * 80000, _DummyEmbedder(), _DummyHybrid(), use_valid_time=False)
    chosen_tx, _ = tx_store.step("Lab-50", "director", allow_fallback=False)
    assert chosen_tx is not None
    assert chosen_tx.value == "Carol"


def test_stress_v21_lexicographic_matches_logged_winners_offline():
    dataset = json.loads((ROOT / "data" / "paraphrase_stress_v2_1" / "dataset.json").read_text(encoding="utf-8"))
    meter = _CachedMeter(ROOT / "results" / "test" / "paraphrase-stress-v2-1" / "requests")
    summary = json.loads((ROOT / "results" / "test" / "paraphrase-stress-v2-1" / "summary.json").read_text(encoding="utf-8"))
    s_map = {(r["arm"], r["group"]): r for r in summary["rows"]}
    assert s_map["carr_v2_llm_valid_time", "all"]["correct"] == 81
    assert s_map["carr_v2_llm_valid_time", "out_of_order_timestamp"]["correct"] == 20

    for item in dataset["items"]:
        sid = item["id"]
        lines = item["history"].splitlines()
        pairs, raw_call = extract_facts_with_valid_time(item["history"], sid, meter, _DummyEmbedder())
        data = json.loads(raw_call["answer"])

        by_key_new = defaultdict(list)
        by_key_old = defaultdict(list)
        for obs, lex_rank in pairs:
            by_key_new[(norm(obs.entity), obs.relation)].append((lex_rank, obs.order))

        for row in data.get("facts", []):
            if not isinstance(row, dict):
                continue
            lnum = row.get("line")
            if not isinstance(lnum, int) or not (1 <= lnum <= len(lines)):
                continue
            ent = str(row.get("entity") or "").strip()
            rel = canon(str(row.get("relation") or "").strip())
            val = str(row.get("value") or "").strip()
            if not ent or not rel or not val:
                continue
            old_r = _legacy_numeric_rank(row.get("valid_year"), bool(row.get("is_reminiscence", False)), lnum)
            by_key_old[(norm(ent), rel)].append((old_r, lnum))

        assert set(by_key_new) == set(by_key_old)
        for k in by_key_new:
            win_new = max(by_key_new[k])[1]
            win_old = max(by_key_old[k])[1]
            assert win_new == win_old, f"Mismatch on {sid} key={k}: {win_new} vs {win_old}"
