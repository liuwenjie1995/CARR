"""Unit tests for locked LongMemEval-S* partition and scoring integration."""

import json
from pathlib import Path

from src.longmemeval_data import LME_DIR, read_longmemeval_locked
from src.run_longmemeval import score_longmemeval


def test_longmemeval_split_lock_integrity():
    lock = json.loads((LME_DIR / "split-lock.json").read_text(encoding="utf-8"))
    assert lock["test_locked"] is True
    assert not any(lock["cross_split_intersections"].values())

    dev_samples, dev_ctx = read_longmemeval_locked("dev")
    test_samples, test_ctx = read_longmemeval_locked("test")

    assert len(dev_ctx) == 1 and len(dev_samples) == 60
    assert len(test_ctx) == 4 and len(test_samples) == 240
    assert not (set(dev_ctx) & set(test_ctx))
    assert not ({s["id"] for s in dev_samples} & {s["id"] for s in test_samples})


def test_longmemeval_official_scorer():
    m = score_longmemeval("The answer is June 3rd.", ["June 3rd"])
    assert m["substring_exact_match"] is True
    assert m["f1"] > 0.0
