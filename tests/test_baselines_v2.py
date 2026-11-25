"""Unit tests for controlled external baselines (BM25+Recency, IRCoT, Mem0-style)."""

import json

from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.ircot_baseline import run_ircot
from src.baselines_v2.mem0_style import Mem0Store, consolidate_mem0_history
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex, TombstoneDenseIndex
from tests.test_carr_v2 import DummyEmbedder


def test_bm25_recency_orders_chronologically():
    history = (
        "1. Bob lives in Madrid.\n"
        "2. Weather is sunny today.\n"
        "15. Bob now lives in Lisbon.\n"
    )
    idx = HybridLexicalDenseIndex(DummyEmbedder(), history, chunk_tokens=30, top_k=3)
    ctx, meta = retrieve_bm25_recency(idx, "Where does Bob live?", top_k=3, max_lines=2)
    lines = ctx.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1.") and lines[1].startswith("15.")
    assert meta["status"] == "bm25_bge_recency"


def test_ircot_stops_on_answer_phrase_and_caps_chunks():
    history = "1. Fact A.\n2. Fact B.\n3. Fact C.\n4. Fact D.\n"
    idx = TombstoneDenseIndex(DummyEmbedder(), history, chunk_tokens=8)
    calls = []

    def mock_chat(tag, msgs, mtok):
        calls.append(tag)
        if len(calls) == 1:
            return {"id": tag, "error": None, "answer": "Fact B points to Fact C.", "finish": "stop"}
        return {"id": tag, "error": None, "answer": "So the answer is: Fact C", "finish": "stop"}

    ctx, trace = run_ircot(idx, "Question about Fact A?", mock_chat, "q1", max_steps=4, init_k=1, step_k=1, max_chunks=3)
    assert len(calls) == 2
    assert trace["status"] == "ircot_completed"
    assert len(trace["chunk_ids"]) <= 3
    assert ctx


def test_mem0_style_add_update_delete():
    emb = DummyEmbedder()
    stmts = [
        {"idx": 0, "order": 1, "text": "1. Alice lives in Beijing."},
        {"idx": 1, "order": 2, "text": "2. Alice moved to Shanghai."},
    ]

    def mock_chat(tag, msgs, mtok, json_output=True):
        return {
            "id": tag,
            "error": None,
            "finish": "stop",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "cache_hit_tokens": 0},
            "usage_usd": "0.00001",
            "answer": json.dumps({
                "ops": [
                    {"idx": 0, "op": "ADD", "target_id": None, "text": "1. Alice lives in Beijing."},
                    {"idx": 1, "op": "UPDATE", "target_id": 1, "text": "2. Alice moved to Shanghai."},
                ]
            }),
        }

    store, usage = consolidate_mem0_history(stmts, emb, mock_chat, batch_size=10)
    assert usage["op_counts"]["ADD"] == 1
    assert usage["op_counts"]["UPDATE"] == 1
    assert len(store.memories) == 1
    ctx, _ = store.search_context("Where does Alice live?")
    assert "Shanghai" in ctx and "Beijing" not in ctx
