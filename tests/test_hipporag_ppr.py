"""Unit tests for HippoRAG-style PPR graph retrieval and tombstone pruning."""

import numpy as np

from src.baselines_v2.hipporag_ppr import GraphFact, HippoGraphIndex


class _HashEmbedder:
    """Deterministic lightweight embedder for offline PPR graph tests."""

    def embed(self, texts, batch_size=32):
        vecs = []
        for t in texts:
            v = np.zeros(32, dtype=np.float32)
            for tok in t.lower().split():
                v[hash(tok) % 32] += 1.0
            n = float(np.linalg.norm(v))
            vecs.append(v / n if n > 0 else v)
        return vecs


def test_hipporag_ppr_bridges_two_hop_entity_chain():
    facts = [
        GraphFact(1, "Alice", "works_at", "Acme Corp", 1, "1. Alice works at Acme Corp."),
        GraphFact(2, "Acme Corp", "headquarter", "Zurich", 2, "2. The headquarter of Acme Corp is Zurich."),
        GraphFact(3, "Bob", "lives_in", "Paris", 3, "3. Bob lives in Paris."),
    ]
    idx = HippoGraphIndex(facts, _HashEmbedder(), damping=0.5, num_iters=20)
    ctx, meta = idx.search("Where is Alice's employer headquartered?", top_k=2)
    assert "Alice works at Acme Corp." in ctx
    assert "Acme Corp is Zurich." in ctx
    assert "alice" in meta["seed_entities"]


def test_hipporag_tombstone_prunes_stale_conflict_edge():
    raw_facts = [
        GraphFact(1, "Alice", "city", "Boston", 1, "1. Alice lives in Boston."),
        GraphFact(2, "Alice", "city", "Seattle", 2, "2. Alice lives in Seattle."),
    ]
    active_facts = [raw_facts[1]]
    idx_raw = HippoGraphIndex(raw_facts, _HashEmbedder())
    idx_tomb = HippoGraphIndex(active_facts, _HashEmbedder())

    ctx_raw, _ = idx_raw.search("Which city does Alice live in?", top_k=2)
    ctx_tomb, _ = idx_tomb.search("Which city does Alice live in?", top_k=2)
    assert "Boston" in ctx_raw and "Seattle" in ctx_raw
    assert "Boston" not in ctx_tomb and "Seattle" in ctx_tomb
