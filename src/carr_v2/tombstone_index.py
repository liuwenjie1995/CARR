"""Tombstone-masked stable chunking and BM25 + BGE hybrid retrieval."""

from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from src.v2_retrieval import PASSAGE_TOKENS, QUERY_PREFIX, TOP_K, normalize_rows, proxy_tokens

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize_lexical(text: str) -> list[str]:
    return [tok.casefold() for tok in WORD_RE.findall(text)]


def chunk_with_line_ids(history: str, chunk_tokens: int = PASSAGE_TOKENS) -> list[dict]:
    """Pack non-empty lines into stable chunks with original 0-based line indices."""
    chunks: list[dict] = []
    cur_ids: list[int] = []
    cur_lines: list[str] = []
    cur_tokens = 0
    for idx, raw_line in enumerate(history.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            continue
        toks = proxy_tokens(raw_line + "\n")
        if cur_lines and cur_tokens + toks > chunk_tokens:
            chunks.append({
                "id": len(chunks),
                "line_ids": cur_ids,
                "lines": cur_lines,
                "text": "\n".join(cur_lines),
            })
            cur_ids, cur_lines, cur_tokens = [], [], 0
        cur_ids.append(idx)
        cur_lines.append(raw_line.rstrip("\r\n"))
        cur_tokens += toks
    if cur_lines:
        chunks.append({
            "id": len(chunks),
            "line_ids": cur_ids,
            "lines": cur_lines,
            "text": "\n".join(cur_lines),
        })
    if not chunks:
        raise ValueError("Dense retrieval needs at least one nonempty chunk.")
    return chunks


class BM25Index:
    """Pure-Python/NumPy BM25 scorer over chunk texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(texts)
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lens = np.zeros(self.n_docs, dtype=np.float32)
        df: Counter[str] = Counter()
        for i, text in enumerate(texts):
            tokens = tokenize_lexical(text)
            self.doc_lens[i] = len(tokens)
            tf = Counter(tokens)
            self.doc_freqs.append(tf)
            df.update(tf.keys())
        self.avgdl = float(self.doc_lens.mean()) if self.n_docs else 1.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tf in enumerate(self.doc_freqs):
            for term, count in tf.items():
                self.postings.setdefault(term, []).append((doc_id, count))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out
        q_terms = set(tokenize_lexical(query))
        avgdl = max(self.avgdl, 1e-6)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                dl = float(self.doc_lens[doc_id])
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                out[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom
        return out


class TombstoneDenseIndex:
    """Dense index over invariant raw chunk boundaries with post-retrieval tombstone masking."""

    def __init__(
        self,
        model,
        history: str,
        tombstones: set[int] | None = None,
        *,
        chunk_tokens: int = PASSAGE_TOKENS,
        top_k: int = TOP_K,
        _chunks: list[dict] | None = None,
        _vectors: np.ndarray | None = None,
    ):
        self.model = model
        self.history = history
        self.tombstones = frozenset(tombstones or ())
        self.chunk_tokens = chunk_tokens
        self.top_k = top_k
        self.chunks = _chunks if _chunks is not None else chunk_with_line_ids(history, chunk_tokens)
        self.texts = [c["text"] for c in self.chunks]
        if _vectors is not None:
            self.vectors = _vectors
        else:
            self.vectors = normalize_rows(list(model.embed(self.texts, batch_size=32)))

    def with_tombstones(self, tombstones: set[int] | None) -> "TombstoneDenseIndex":
        return TombstoneDenseIndex(
            self.model,
            self.history,
            tombstones,
            chunk_tokens=self.chunk_tokens,
            top_k=self.top_k,
            _chunks=self.chunks,
            _vectors=self.vectors,
        )

    def _format_masked(self, chunk_ids: list[int]) -> tuple[str, list[int], list[int]]:
        kept_lines: list[str] = []
        kept_line_ids: list[int] = []
        for cid in sorted(chunk_ids):
            chunk = self.chunks[cid]
            for lid, line in zip(chunk["line_ids"], chunk["lines"]):
                if lid not in self.tombstones:
                    kept_lines.append(line)
                    kept_line_ids.append(lid)
        return "\n".join(kept_lines), sorted(chunk_ids), kept_line_ids

    def rank_scores(self, query: str) -> np.ndarray:
        qvec = normalize_rows(list(self.model.embed([QUERY_PREFIX + query])))[0]
        return self.vectors @ qvec

    def search(self, query: str, top_k: int | None = None) -> tuple[str, list[int], list[int]]:
        k = top_k or self.top_k
        scores = self.rank_scores(query)
        ids = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return self._format_masked(ids)


class HybridLexicalDenseIndex(TombstoneDenseIndex):
    """Reciprocal Rank Fusion (BM25 + BGE) over invariant chunks with post-retrieval tombstone masking."""

    def __init__(
        self,
        model,
        history: str,
        tombstones: set[int] | None = None,
        *,
        chunk_tokens: int = PASSAGE_TOKENS,
        top_k: int = TOP_K,
        rrf_k: int = 60,
        _chunks: list[dict] | None = None,
        _vectors: np.ndarray | None = None,
        _bm25: BM25Index | None = None,
    ):
        super().__init__(
            model,
            history,
            tombstones,
            chunk_tokens=chunk_tokens,
            top_k=top_k,
            _chunks=_chunks,
            _vectors=_vectors,
        )
        self.rrf_k = rrf_k
        self.bm25 = _bm25 if _bm25 is not None else BM25Index(self.texts)

    def with_tombstones(self, tombstones: set[int] | None) -> "HybridLexicalDenseIndex":
        return HybridLexicalDenseIndex(
            self.model,
            self.history,
            tombstones,
            chunk_tokens=self.chunk_tokens,
            top_k=self.top_k,
            rrf_k=self.rrf_k,
            _chunks=self.chunks,
            _vectors=self.vectors,
            _bm25=self.bm25,
        )

    def search(self, query: str, top_k: int | None = None) -> tuple[str, list[int], list[int]]:
        k = top_k or self.top_k
        dense_scores = self.rank_scores(query)
        lex_scores = self.bm25.scores(query)

        dense_order = sorted(range(len(dense_scores)), key=lambda i: (-float(dense_scores[i]), i))
        lex_order = sorted(range(len(lex_scores)), key=lambda i: (-float(lex_scores[i]), i))

        dense_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(dense_order)}
        lex_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(lex_order)}

        rrf_scores = {
            doc_id: (1.0 / (self.rrf_k + dense_rank[doc_id]))
            + (1.0 / (self.rrf_k + lex_rank[doc_id]) if lex_scores[doc_id] > 0 else 0.0)
            for doc_id in range(len(self.chunks))
        }
        ids = sorted(range(len(self.chunks)), key=lambda i: (-rrf_scores[i], -float(dense_scores[i]), i))[:k]
        return self._format_masked(ids)
