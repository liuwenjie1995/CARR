"""Controlled HippoRAG-style Personalized PageRank (PPR) graph retriever baseline.

Builds a bipartite entity-fact knowledge graph (without external graph DBs),
seeds query entity nodes (via exact lexical span matching + dense BGE cosine)
plus optional dense fact seeds, and runs Personalized PageRank power iteration
to score and retrieve Top-k fact statements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np

from src.carr_v2.tombstone_index import tokenize_lexical
from src.factual_v2 import norm
from src.v2_retrieval import QUERY_PREFIX, normalize_rows


@dataclass(frozen=True)
class GraphFact:
    fact_id: int
    entity: str
    relation: str
    value: str
    order: int
    text: str


class HippoGraphIndex:
    """Bipartite entity-fact graph with HippoRAG-style Personalized PageRank retrieval."""

    def __init__(
        self,
        facts: list[GraphFact],
        embedder,
        *,
        damping: float = 0.5,
        num_iters: int = 25,
        fact_seed_weight: float = 0.05,
    ) -> None:
        self.facts = list(facts)
        self.embedder = embedder
        self.damping = damping
        self.num_iters = num_iters
        self.fact_seed_weight = fact_seed_weight

        # Build entity vocabulary from normalized subject & object surfaces
        entity_surfaces: dict[str, str] = {}
        for f in self.facts:
            for raw_ent in (f.entity, f.value):
                key = norm(raw_ent)
                if key and len(key) >= 2 and key not in entity_surfaces:
                    entity_surfaces[key] = raw_ent.strip()

        self.entity_keys = sorted(entity_surfaces.keys())
        self.entity_to_id = {k: idx for idx, k in enumerate(self.entity_keys)}
        self.num_entities = len(self.entity_keys)
        self.num_facts = len(self.facts)
        self.num_nodes = self.num_entities + self.num_facts

        # Build fast O(query_tokens^2) entity span lookup index (handles possessives & punctuation)
        self._tok_span_to_ei: dict[str, int] = {}
        max_len = 1
        for ei, ek in enumerate(self.entity_keys):
            if len(ek) < 3:
                continue
            toks = tokenize_lexical(ek)
            if toks:
                span = " ".join(toks)
                self._tok_span_to_ei.setdefault(span, ei)
                if len(toks) > max_len:
                    max_len = len(toks)
        self._max_ent_tokens = min(max_len, 10)

        # Build undirected bipartite adjacency edges: entity node [0..E) <-> fact node [E..E+F)
        src_list: list[int] = []
        dst_list: list[int] = []
        ent_deg = np.zeros(self.num_entities, dtype=np.float32)

        for fi, f in enumerate(self.facts):
            f_node = self.num_entities + fi
            seen_e: set[int] = set()
            for raw_ent in (f.entity, f.value):
                key = norm(raw_ent)
                ei = self.entity_to_id.get(key)
                if ei is not None and ei not in seen_e:
                    seen_e.add(ei)
                    src_list.extend((ei, f_node))
                    dst_list.extend((f_node, ei))
                    ent_deg[ei] += 1.0

        self.ent_deg = ent_deg
        if src_list:
            self.edge_src = np.asarray(src_list, dtype=np.int32)
            self.edge_dst = np.asarray(dst_list, dtype=np.int32)
            deg = np.bincount(self.edge_src, minlength=self.num_nodes).astype(np.float32)
            deg[deg == 0] = 1.0
            self.edge_weight = 1.0 / deg[self.edge_src]
        else:
            self.edge_src = np.zeros(0, dtype=np.int32)
            self.edge_dst = np.zeros(0, dtype=np.int32)
            self.edge_weight = np.zeros(0, dtype=np.float32)

        # Embed unique entity surfaces and fact texts for dense seeding
        if self.entity_keys:
            ent_texts = [entity_surfaces[k] for k in self.entity_keys]
            self.entity_vecs = normalize_rows(list(embedder.embed(ent_texts, batch_size=64)))
        else:
            self.entity_vecs = None

        if self.facts:
            fact_texts = [f.text for f in self.facts]
            self.fact_vecs = normalize_rows(list(embedder.embed(fact_texts, batch_size=64)))
        else:
            self.fact_vecs = None

    def _seed_vector(self, query: str) -> tuple[np.ndarray, list[str], list[int]]:
        p = np.zeros(self.num_nodes, dtype=np.float32)
        if self.num_nodes == 0:
            return p, [], []

        qtoks = tokenize_lexical(query)
        matched_entities: list[str] = []
        used_ranges: list[tuple[int, int]] = []

        # 1. Exact lexical n-gram matching on entity nodes (longest match first, O(|q|^2) fast lookup)
        n_q = len(qtoks)
        for length in range(min(self._max_ent_tokens, n_q), 0, -1):
            for start in range(0, n_q - length + 1):
                end = start + length
                if any(not (end <= s0 or start >= e0) for s0, e0 in used_ranges):
                    continue
                span = " ".join(qtoks[start:end])
                ei = self._tok_span_to_ei.get(span)
                if ei is not None:
                    w = 1.0 / float(np.log2(2.0 + self.ent_deg[ei]))
                    p[ei] += max(w, 0.2)
                    used_ranges.append((start, end))
                    matched_entities.append(self.entity_keys[ei])
                    if len(matched_entities) >= 6:
                        break
            if len(matched_entities) >= 6:
                break

        qvec = normalize_rows(list(self.embedder.embed([QUERY_PREFIX + query])))[0]

        # 2. Dense BGE entity linking (adds synonym/paraphrase seed nodes)
        if self.entity_vecs is not None and self.num_entities > 0:
            escores = self.entity_vecs @ qvec
            top_e = np.argsort(-escores)[:6]
            for ei in top_e:
                sc = float(escores[ei])
                if sc >= 0.48:
                    w = (sc * 0.6) / float(np.log2(2.0 + self.ent_deg[ei]))
                    p[ei] += w
                    if self.entity_keys[ei] not in matched_entities:
                        matched_entities.append(self.entity_keys[ei])

        # 3. HippoRAG 2-style light dense fact passage seeding
        top_fact_seeds: list[int] = []
        if self.fact_vecs is not None and self.num_facts > 0 and self.fact_seed_weight > 0:
            fscores = self.fact_vecs @ qvec
            top_f = np.argsort(-fscores)[:5]
            for fi in top_f:
                sc = float(fscores[fi])
                if sc > 0:
                    p[self.num_entities + int(fi)] += self.fact_seed_weight * sc
                    top_fact_seeds.append(int(fi))

        total = float(p.sum())
        if total <= 0:
            p[: self.num_entities] = 1.0 / max(1, self.num_entities)
        else:
            p /= total
        return p, matched_entities, top_fact_seeds

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        max_bytes: int = 4800,
        chronological_output: bool = True,
    ) -> tuple[str, dict]:
        if not self.facts:
            return "", {"seed_entities": [], "seed_facts": [], "selected_fact_ids": []}

        p, seed_entities, seed_facts = self._seed_vector(query)
        pi = p.copy()
        alpha = self.damping
        if self.edge_src.size > 0:
            for _ in range(self.num_iters):
                contrib = pi[self.edge_src] * self.edge_weight
                propagated = np.bincount(self.edge_dst, weights=contrib, minlength=self.num_nodes).astype(np.float32)
                pi = (1.0 - alpha) * p + alpha * propagated

        fact_scores = pi[self.num_entities :]
        ranked_idx = np.argsort(-fact_scores, kind="mergesort")[:top_k]
        chosen = [self.facts[int(i)] for i in ranked_idx if fact_scores[int(i)] > 0]
        if not chosen:
            chosen = [self.facts[int(i)] for i in ranked_idx[:top_k]]

        if chronological_output:
            chosen.sort(key=lambda f: f.order)

        lines: list[str] = []
        used = 0
        kept_ids: list[int] = []
        for f in chosen:
            b = len((f.text + "\n").encode("utf-8"))
            if lines and used + b > max_bytes:
                break
            lines.append(f.text)
            used += b
            kept_ids.append(f.fact_id)

        return "\n".join(lines), {
            "seed_entities": seed_entities,
            "seed_facts": seed_facts,
            "selected_fact_ids": kept_ids,
            "top_ppr_scores": [round(float(fact_scores[int(i)]), 6) for i in ranked_idx[: len(kept_ids)]],
        }
