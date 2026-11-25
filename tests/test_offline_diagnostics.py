"""Unit tests for offline diagnostics, title-safe parsing, and frozen-source integrity."""

from pathlib import Path
import json

from src.factual_v2 import Record
from src.freeze_formal_v7 import DEST, hashes
from src.offline_diagnostics import (
    hardened_parse_record,
    has_alias,
    mcnemar_exact,
)
from src.v2_report import load


def test_frozen_formal_v7_hashes_unchanged():
    """Ensure optimization work never mutates any file locked in formal-v7 freeze.json."""
    frozen = load(DEST / "freeze.json")
    assert hashes() == frozen["source_sha256"]


def test_title_safe_negation_recovers_song_title_without_breaking_predicate_negation():
    song_rec = Record(index=10, text="Tomorrow Never Knows was performed by Madonna.", start=0, order=100)
    parsed = hardened_parse_record(song_rec)
    assert parsed is not None
    assert parsed.entity == "Tomorrow Never Knows"
    assert parsed.relation == "performer"
    assert parsed.value == "Madonna"
    assert parsed.origin == "hardened_title_safe"

    neg_rec = Record(index=11, text="Alice never lives in Paris.", start=0, order=101)
    assert hardened_parse_record(neg_rec) is None

    lang_rec = Record(index=12, text="The official language of Brazil is Portuguese.", start=0, order=102)
    parsed_lang = hardened_parse_record(lang_rec)
    assert parsed_lang is not None
    assert parsed_lang.entity == "Brazil"
    assert parsed_lang.relation == "official_language"
    assert parsed_lang.value == "Portuguese"


def test_has_alias_word_boundaries():
    assert has_alias("10. The capital of Australia is Oderzo.", ["Oderzo"])
    assert not has_alias("10. The capital of Australia is Oderzoville.", ["Oderzo"])
    assert has_alias("10. The sport is pesäpallo.", ["Pesäpallo"])


def test_mcnemar_exact_values():
    assert mcnemar_exact(13, 13) == "1"
    assert mcnemar_exact(0, 0) == "1"
    assert mcnemar_exact(98, 52) == "0.000215"


def test_diagnostics_summary_reconciles_with_formal_logs():
    summary_path = Path("results/analysis/formal_v7_diagnostics/diagnostics_summary.json")
    assert summary_path.exists(), "Run `python -m src.offline_diagnostics` first."
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # Zero false deletions on both FactConsolidation and EventQA
    assert data["chunk_displacement"]["bge_top8"]["raw_win_gold_deleted_by_archive"] == 0
    assert data["chunk_displacement"]["relay"]["raw_win_gold_deleted_by_archive"] == 0
    assert data["hardening_counterfactual"]["eventqa_removed_lines"] == 0

    # Prompt diff accounting
    assert data["chunk_displacement"]["relay"]["identical_prompts"] == 405
    assert data["chunk_displacement"]["relay"]["prompt_diffs"] == 157

    # Single-hop route sum = 286
    assert sum(data["sh_route_breakdown"].values()) == 286

    # Cross-backbone agreement check
    ag = data["cross_backbones"]["carr_ds_vs_glm_562"]
    assert ag["n"] == 562
    assert ag["both_correct"] + ag["both_wrong"] + ag["ds_only"] + ag["glm_only"] == 562
