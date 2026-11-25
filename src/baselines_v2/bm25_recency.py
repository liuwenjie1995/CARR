"""Zero-LLM BM25 + BGE RRF + recency-ordered retrieval baseline."""

from __future__ import annotations

import re

from src.carr_v2.tombstone_index import HybridLexicalDenseIndex, tokenize_lexical
from src.factual_v2 import norm

SERIAL_RE = re.compile(r"^(\d+)\.")
SESSION_RE = re.compile(r"^\[S(\d+)T(\d+)\b")


def _recency_key(line: str) -> tuple[int, int]:
    if m := SERIAL_RE.match(line):
        return (int(m.group(1)), 0)
    if m := SESSION_RE.match(line):
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def retrieve_bm25_recency(
    hybrid_index: HybridLexicalDenseIndex,
    question: str,
    *,
    top_k: int = 8,
    max_lines: int = 16,
    max_bytes: int = 3600,
) -> tuple[str, dict]:
    """Retrieve Top-k hybrid chunks, score lines by lexical overlap + recency, and order chronologically."""
    visible, chunk_ids, _ = hybrid_index.search(question, top_k=top_k)
    q_terms = set(tokenize_lexical(question))
    lines = [ln for ln in visible.splitlines() if ln.strip()]

    # Score each line by query-term overlap, breaking ties by newest order first,
    # then serialize selected lines in ascending chronological order for the reader.
    scored = []
    for idx, line in enumerate(lines):
        ltoks = set(tokenize_lexical(line))
        overlap = len(q_terms & ltoks)
        rec = _recency_key(line)
        scored.append((overlap, rec, idx, line))

    scored.sort(key=lambda item: (-item[0], -item[1][0], -item[1][1], item[2]))
    selected = []
    used = 0
    for overlap, rec, idx, line in scored:
        b = len(line.encode("utf-8")) + 1
        if selected and used + b > max_bytes:
            continue
        selected.append((rec, idx, line))
        used += b
        if len(selected) >= max_lines:
            break

    selected.sort(key=lambda item: (item[0], item[1]))
    context = "\n".join(item[2] for item in selected)
    return context, {
        "status": "bm25_bge_recency",
        "chunk_ids": chunk_ids,
        "selected_lines": len(selected),
        "recency_keys": [item[0] for item in selected],
    }
