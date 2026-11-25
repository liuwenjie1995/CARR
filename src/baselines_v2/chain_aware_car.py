"""Controlled Chain-Aware Resolution (CAR-style) baseline for same-backbone comparison.

Implements iterative hop-by-hop entity-relation resolution with deterministic recency selection
(max order among active/parsed facts matching each intermediate hop), contrasting with CARR v2's
single-pass ~200-token decomposition + local O(0.01 ms) deterministic lookup.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from src.carr_v2.semantic_router import CANONICAL_RELATIONS, _entity_variants
from src.factual_v2 import norm


def car_hop_messages(
    question: str,
    resolved_lines: list[str],
    hop_idx: int,
) -> list[dict]:
    """Prompt LLM for the next unresolved (entity, relation) hop or termination."""
    rel_hint = ", ".join(CANONICAL_RELATIONS[:24])
    state_block = "\n".join(resolved_lines) if resolved_lines else "(none resolved yet)"
    return [
        {
            "role": "system",
            "content": (
                "You perform hop-by-hop Chain-Aware Resolution over a temporal knowledge store. "
                "Given the question and facts resolved so far, decide the NEXT single atomic (entity, relation) lookup needed, "
                "or mark done=true if the resolved facts already contain enough information to answer the question.\n"
                'Return JSON strictly: {"done": true} or {"done": false, "entity": "exact entity name", '
                '"relation": "concise snake_case relation label", "sub_query": "short lookup phrase"}.\n'
                f"Prefer canonical relation names when applicable (e.g. {rel_hint}). "
                "NEVER output an entity that is not either in the question or an object value in the resolved facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Hop step: {hop_idx}/4\n"
                f"Resolved facts so far:\n{state_block}\n"
                "Next lookup JSON:"
            ),
        },
    ]


def parse_car_decision(raw_answer: str) -> dict:
    try:
        data = json.loads(raw_answer or "{}")
        if not isinstance(data, dict):
            return {"done": True}
        if bool(data.get("done")):
            return {"done": True}
        ent = str(data.get("entity") or "").strip()
        rel = re.sub(r"\s+", "_", str(data.get("relation") or "").strip().lower())
        sub_q = str(data.get("sub_query") or "").strip()
        if not ent or not rel:
            return {"done": True}
        return {"done": False, "entity": ent, "relation": rel, "sub_query": sub_q}
    except (ValueError, TypeError, AttributeError):
        return {"done": True}


def run_car_chain(
    store,
    question: str,
    chat_fn: Callable[[str, list[dict], int, bool], dict],
    *,
    call_prefix: str,
    max_hops: int = 4,
) -> tuple[str, dict]:
    """Run iterative Chain-Aware Resolution (up to 4 hop calls + deterministic latest selection)."""
    resolved_lines: list[str] = []
    visited: set[int] = set()
    hop_calls: list[dict] = []
    hop_events: list[dict] = []
    status = "completed"

    for h in range(1, max_hops + 1):
        msgs = car_hop_messages(question, resolved_lines, h)
        call = chat_fn(f"car_hop{h}--{call_prefix}", msgs, 96, True)
        hop_calls.append(call)
        dec = parse_car_decision(call.get("answer") or "")
        if dec.get("done"):
            status = "stopped_done" if resolved_lines else "empty_done"
            hop_events.append({"hop": h, "decision": "done", "chosen": None})
            break

        ent, rel, sub_q = dec["entity"], dec["relation"], dec["sub_query"]
        # Deterministic freshness resolution via store.step (exact recency -> semantic recency -> hybrid fallback)
        fact, ev = store.step(ent, rel, sub_q, allow_fallback=True)
        hop_events.append({"hop": h, "entity": ent, "relation": rel, "channel": ev["channel"], "chosen": ev["chosen"]})
        if not fact or fact.order in visited:
            status = "miss_or_cycle"
            break
        visited.add(fact.order)
        resolved_lines.append(store.lines[fact.order])

    if not resolved_lines:
        from src.carr_v2.semantic_router import whole_question_hybrid_fallback
        ev, fb = whole_question_hybrid_fallback(store, question)
        return "\n".join(ev), {
            "status": "fallback_" + status,
            "fact_ids": fb.get("fact_ids", []),
            "hop_events": hop_events,
            "hop_calls": hop_calls,
        }

    return "\n".join(resolved_lines), {
        "status": status,
        "fact_ids": sorted(visited),
        "hop_events": hop_events,
        "hop_calls": hop_calls,
    }
