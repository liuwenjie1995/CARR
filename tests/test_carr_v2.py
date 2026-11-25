"""Unit tests for CARR v2 tombstone index, open parser, and 3-channel hybrid relay."""

import numpy as np

from src.carr_v2.open_parser import archive_v2, parse_longmemeval_sessions, parse_numbered_v2
from src.carr_v2.semantic_router import RecencyHybridStore, parse_plan_v2, retrieve_v2
from src.carr_v2.tombstone_index import (
    BM25Index,
    HybridLexicalDenseIndex,
    TombstoneDenseIndex,
    chunk_with_line_ids,
)
from src.factual_v2 import Record, records_for


class DummyEmbedder:
    """Deterministic bag-of-word-hash embedder for fast, offline unit tests."""

    def embed(self, texts, batch_size=32):
        dim = 64
        out = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for tok in text.lower().split():
                h = sum(ord(c) for c in tok) % dim
                vec[h] += 1.0
            if vec.sum() == 0:
                vec[0] = 1.0
            out.append(vec)
        return out


def test_tombstone_index_preserves_chunk_boundaries():
    lines = [
        "0. Header line",
        "1. Alice lives in Beijing.",
        "2. Bob lives in Tokyo.",
        "3. Alice now lives in Shanghai.",
        "4. Carol works in London.",
    ]
    history = "\n".join(lines)
    emb = DummyEmbedder()
    raw_idx = TombstoneDenseIndex(emb, history, tombstones=None, chunk_tokens=16)
    tomb_idx = raw_idx.with_tombstones({1})

    # Chunk count and chunk boundaries MUST be strictly identical
    assert len(raw_idx.chunks) == len(tomb_idx.chunks)
    assert [c["line_ids"] for c in raw_idx.chunks] == [c["line_ids"] for c in tomb_idx.chunks]

    vis, cids, kept_ids = tomb_idx.search("Alice lives", top_k=4)
    assert 1 not in kept_ids
    assert "Beijing" not in vis
    assert "Shanghai" in vis


def test_bm25_and_hybrid_rrf_masking():
    history = (
        "1. Xyzzyq is located in AlphaCity.\n"
        "2. Something unrelated about weather.\n"
        "3. Xyzzyq is located in BetaCity.\n"
    )
    emb = DummyEmbedder()
    hybrid = HybridLexicalDenseIndex(emb, history, tombstones={0}, chunk_tokens=20, top_k=2)
    vis, _, kept = hybrid.search("Xyzzyq")
    assert 0 not in kept
    assert "AlphaCity" not in vis
    assert "BetaCity" in vis


def test_open_parser_v2_templates_and_title_safe_negation():
    r1 = Record(0, "Tomorrow Never Knows was performed by Madonna.", 0, 10)
    f1 = parse_numbered_v2(r1)
    assert f1 is not None and f1.entity == "Tomorrow Never Knows" and f1.value == "Madonna"

    r2 = Record(1, "Alice never lives in Paris.", 0, 11)
    assert parse_numbered_v2(r2) is None

    r3 = Record(2, "Tony Soprano was created by David Chase.", 0, 12)
    f3 = parse_numbered_v2(r3)
    assert f3 is not None and f3.relation == "creator" and f3.value == "David Chase"

    r4 = Record(3, "The Prime Minister of Sweden is Stefan Löfven.", 0, 13)
    f4 = parse_numbered_v2(r4)
    assert f4 is not None and f4.relation == "head_of_government"


def test_semantic_router_open_relation_and_3channel_relay():
    history = (
        "0. Facts:\n"
        "1. Acme Corp was founded by Alice.\n"
        "2. Alice is a citizen of France.\n"
        "3. Alice is a citizen of Canada.\n"
        "4. The capital of Canada is Ottawa.\n"
    )
    recs, _ = records_for(history, "numbered")
    emb = DummyEmbedder()
    hybrid = HybridLexicalDenseIndex(emb, history, chunk_tokens=60)
    store = RecencyHybridStore(list(recs.values()), history.splitlines(), emb, hybrid, use_exact=True, semantic_threshold=0.0)

    plan = parse_plan_v2(
        {
            "error": None,
            "finish": "stop",
            "answer": '{"mode":"multi","start":"Acme Corp","steps":[{"relation":"founder_person","query":"who founded Acme Corp"},{"relation":"citizenship","query":"citizenship"},{"relation":"capital","query":"capital"}]}',
        },
        "What is the capital of the citizenship of who founded Acme Corp?",
        store.entities,
    )
    assert plan["mode"] == "multi" and plan["start"] == "Acme Corp"
    ev, trace = retrieve_v2(store, plan, "What is the capital of the citizenship of who founded Acme Corp?")
    assert len(ev) == 3
    assert "Ottawa" in ev[-1]
    assert trace["events"][1]["channel"] == "exact_latest"


def test_parse_longmemeval_sessions_roundtrip():
    raw = str([
        "Chat Time: 2023/01/10 (Tue) 09:00",
        [{"role": "user", "content": "I live in Austin."}, {"role": "assistant", "content": "Got it."}],
        "Chat Time: 2023/02/14 (Tue) 14:30",
        [{"role": "user", "content": "I just moved to Seattle."}],
    ])
    sessions = parse_longmemeval_sessions(raw)
    assert len(sessions) == 2
    assert sessions[0]["chat_time"].startswith("2023/01/10")
    assert sessions[1]["turns"][0]["content"] == "I just moved to Seattle."
