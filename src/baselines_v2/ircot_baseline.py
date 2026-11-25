"""Controlled same-backbone IRCoT reproduction (Trivedi et al., 2023)."""

from __future__ import annotations

import re
from typing import Callable

from src.carr_v2.tombstone_index import TombstoneDenseIndex


def _ircot_step_messages(question: str, chunks: list[str], cot_history: list[str]) -> list[dict]:
    context = "\n---\n".join(chunks)
    if len(context) > 9000:
        context = context[:9000] + "\n..."
    prev = " ".join(cot_history).strip()
    return [
        {
            "role": "system",
            "content": (
                "You perform iterative retrieval and chain-of-thought reasoning over the retrieved notes. "
                "Given the notes and question, write ONE concise next reasoning sentence to guide the next search "
                "(mentioning any newly discovered bridge entity or missing fact). "
                "If the retrieved notes already contain enough information to answer the question, finish with "
                "'So the answer is: <short answer>'. Keep your reply under 60 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieved notes:\n{context}\n\n"
                f"Question: {question}\n"
                f"Reasoning so far: {prev if prev else '(none)'}\n"
                "Next thought:"
            ),
        },
    ]


def run_ircot(
    dense_index: TombstoneDenseIndex,
    question: str,
    chat_fn: Callable[[str, list[dict], int], dict],
    call_prefix: str,
    *,
    max_steps: int = 4,
    init_k: int = 4,
    step_k: int = 3,
    max_chunks: int = 8,
) -> tuple[str, dict]:
    """Interleave BGE retrieval with CoT reasoning steps up to max_steps and max_chunks."""
    seen_ids: set[int] = set()
    ordered_ids: list[int] = []

    def add_chunks(query: str, k: int) -> list[int]:
        scores = dense_index.rank_scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        added = []
        for cid in ranked:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ordered_ids.append(cid)
            added.append(cid)
            if len(added) >= k or len(ordered_ids) >= max_chunks:
                break
        return added

    init_added = add_chunks(question, init_k)
    cot_steps: list[str] = []
    step_traces: list[dict] = [{"step": 0, "query": question, "added_chunks": init_added}]
    calls: list[dict] = []

    for step in range(1, max_steps + 1):
        chunk_texts = [dense_index.chunks[cid]["text"] for cid in sorted(ordered_ids)]
        msgs = _ircot_step_messages(question, chunk_texts, cot_steps)
        res = chat_fn(f"{call_prefix}_ircot_s{step}", msgs, 96)
        calls.append(res)
        thought = (res.get("answer") or "").strip()
        if thought:
            cot_steps.append(thought)
        step_traces.append({
            "step": step,
            "thought": thought,
            "error": res.get("error"),
            "request_id": res.get("id"),
        })
        if res.get("error") or not thought or re.search(r"so the answer is\s*:", thought, re.I):
            break
        if len(ordered_ids) >= max_chunks or step == max_steps:
            break
        added = add_chunks(thought, step_k)
        step_traces[-1]["added_chunks"] = added
        if not added:
            break

    context, final_ids, _ = dense_index._format_masked(ordered_ids)
    return context, {
        "status": "ircot_completed",
        "chunk_ids": final_ids,
        "steps": step_traces,
        "reasoning_calls": calls,
    }
