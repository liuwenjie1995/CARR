"""Controlled same-backbone Mem0-style (ADD/UPDATE/DELETE/NOOP) memory baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.evaluator import dump_json
from src.v2_retrieval import QUERY_PREFIX, TOP_K, normalize_rows


class Mem0Store:
    """In-memory active memory store supporting ADD / UPDATE / DELETE / NOOP with incremental BGE vectors."""

    def __init__(self, embedder):
        self.model = embedder
        self.memories: dict[int, dict] = {}
        self.next_id = 1
        self._vec_by_id: dict[int, np.ndarray] = {}
        self._vec_ids: list[int] = []
        self._vectors: np.ndarray | None = None
        self.op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}

    def _rebuild_index(self) -> None:
        active = sorted(self.memories.items())
        self._vec_ids = [mid for mid, _ in active]
        if not active:
            self._vectors = None
            return
        missing = [(mid, item["text"]) for mid, item in active if mid not in self._vec_by_id]
        if missing:
            vecs = normalize_rows(list(self.model.embed([t for _, t in missing], batch_size=64)))
            for (mid, _), vec in zip(missing, vecs):
                self._vec_by_id[mid] = vec
        self._vectors = np.stack([self._vec_by_id[mid] for mid in self._vec_ids], axis=0)

    def search_raw(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not self.memories:
            return []
        if self._vectors is None or len(self._vec_ids) != len(self.memories):
            self._rebuild_index()
        assert self._vectors is not None
        qvec = normalize_rows(list(self.model.embed([QUERY_PREFIX + query])))[0]
        scores = self._vectors @ qvec
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), -self._vec_ids[i]))[:top_k]
        return [{**self.memories[self._vec_ids[i]], "score": float(scores[i])} for i in ranked]

    def search_context(self, query: str, top_k: int = TOP_K) -> tuple[str, list[int]]:
        hits = self.search_raw(query, top_k=top_k)
        # Preserve chronological insertion/order for reader consistency
        hits.sort(key=lambda item: item["id"])
        return "\n".join(item["text"] for item in hits), [item["id"] for item in hits]

    def apply_operations(self, statements: list[dict], ops: list[dict]) -> None:
        stmt_map = {s["idx"]: s for s in statements}
        changed = False
        for op_item in ops:
            if not isinstance(op_item, dict):
                continue
            idx = op_item.get("idx")
            action = str(op_item.get("op", "ADD")).upper().strip()
            text = str(op_item.get("text", "")).strip()
            target_id = op_item.get("target_id")
            src = stmt_map.get(idx)
            if src is None:
                continue
            if not text:
                text = src["text"]

            if action == "UPDATE" and isinstance(target_id, int) and target_id in self.memories:
                self.memories[target_id] = {
                    "id": target_id,
                    "text": text,
                    "order": src["order"],
                    "raw": src["text"],
                }
                self._vec_by_id.pop(target_id, None)
                self.op_counts["UPDATE"] += 1
                changed = True
            elif action == "DELETE" and isinstance(target_id, int) and target_id in self.memories:
                del self.memories[target_id]
                self._vec_by_id.pop(target_id, None)
                self.op_counts["DELETE"] += 1
                changed = True
            elif action == "NOOP":
                self.op_counts["NOOP"] += 1
            else:
                mid = self.next_id
                self.next_id += 1
                self.memories[mid] = {
                    "id": mid,
                    "text": text,
                    "order": src["order"],
                    "raw": src["text"],
                }
                self._vec_by_id.pop(mid, None)
                self.op_counts["ADD"] += 1
                changed = True
        if changed:
            self._vectors = None

    def to_dict(self) -> dict:
        return {
            "next_id": self.next_id,
            "op_counts": self.op_counts,
            "active_count": len(self.memories),
            "memories": [self.memories[k] for k in sorted(self.memories)],
        }

    @classmethod
    def from_dict(cls, embedder, data: dict) -> "Mem0Store":
        store = cls(embedder)
        store.next_id = int(data["next_id"])
        store.op_counts = dict(data["op_counts"])
        for item in data["memories"]:
            store.memories[int(item["id"])] = item
        store._rebuild_index()
        return store


def _mem0_batch_messages(batch: list[dict], existing: list[dict]) -> list[dict]:
    existing_block = "\n".join(f"[{m['id']}] {m['text']}" for m in existing) if existing else "(empty)"
    new_block = "\n".join(f"[{s['idx']}] {s['text']}" for s in batch)
    return [
        {
            "role": "system",
            "content": (
                "You are a memory manager maintaining a factual memory store. "
                "For each new observation, compare against Existing Memories and decide ONE operation: "
                "ADD (new non-conflicting fact), UPDATE (supersedes/corrects an existing memory ID), "
                "DELETE (invalidates an existing memory ID), or NOOP (redundant/non-factual). "
                'Return JSON: {"ops": [{"idx": <int>, "op": "ADD|UPDATE|DELETE|NOOP", '
                '"target_id": <int or null>, "text": "<concise memory statement preserving number/time>"}]}.'
            ),
        },
        {
            "role": "user",
            "content": f"Existing Memories:\n{existing_block}\n\nNew Observations:\n{new_block}",
        },
    ]


def consolidate_mem0_history(
    statements: list[dict],
    embedder,
    chat_fn: Callable[[str, list[dict], int, bool], dict],
    *,
    cache_path: Path | None = None,
    batch_size: int = 16,
) -> tuple[Mem0Store, dict]:
    """Incrementally consolidate chronological statements using Mem0-style ADD/UPDATE/DELETE/NOOP."""
    if cache_path is not None and cache_path.exists():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved.get("complete") and saved.get("total_statements") == len(statements):
            return Mem0Store.from_dict(embedder, saved["store"]), saved["write_usage"]

    store = Mem0Store(embedder)
    start_pos = 0
    write_calls: list[dict] = []

    partial_path = cache_path.with_suffix(".partial.json") if cache_path is not None else None
    if partial_path is not None and partial_path.exists():
        part = json.loads(partial_path.read_text(encoding="utf-8"))
        if part.get("total_statements") == len(statements):
            store = Mem0Store.from_dict(embedder, part["store"])
            start_pos = int(part["next_pos"])
            write_calls = list(part.get("write_calls", []))

    for pos in range(start_pos, len(statements), batch_size):
        batch = statements[pos:pos + batch_size]
        # Gather top-2 existing similar memories per statement in this batch
        if store.memories:
            store._rebuild_index()
            assert store._vectors is not None
            qvecs = normalize_rows(list(embedder.embed([QUERY_PREFIX + s["text"] for s in batch], batch_size=32)))
            sim_mat = qvecs @ store._vectors.T
            seen_mids: set[int] = set()
            for r_idx in range(len(batch)):
                row_scores = sim_mat[r_idx]
                top_cols = sorted(range(len(row_scores)), key=lambda c: -float(row_scores[c]))[:2]
                for c in top_cols:
                    if float(row_scores[c]) >= 0.45:
                        seen_mids.add(store._vec_ids[c])
            existing = [store.memories[mid] for mid in sorted(seen_mids)[:24]]
        else:
            existing = []

        msgs = _mem0_batch_messages(batch, existing)
        res = chat_fn(f"mem0_w_{pos:05d}", msgs, 768, True)
        write_calls.append({
            "id": res.get("id"),
            "error": res.get("error"),
            "usage": res.get("usage"),
            "usage_usd": res.get("usage_usd"),
        })
        parsed_ops = []
        if not res.get("error") and res.get("answer"):
            try:
                payload = json.loads(res["answer"])
                if isinstance(payload, dict) and isinstance(payload.get("ops"), list):
                    parsed_ops = payload["ops"]
            except (ValueError, TypeError):
                parsed_ops = []
        if not parsed_ops:
            # Fallback to conservative ADD if JSON parsing failed
            parsed_ops = [{"idx": s["idx"], "op": "ADD", "target_id": None, "text": s["text"]} for s in batch]
        store.apply_operations(batch, parsed_ops)

        if partial_path is not None and ((pos // batch_size) % 10 == 9 or pos + batch_size >= len(statements)):
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            dump_json(partial_path, {
                "total_statements": len(statements),
                "next_pos": min(pos + batch_size, len(statements)),
                "store": store.to_dict(),
                "write_calls": write_calls,
            })

    usage_summary = {
        "calls": len(write_calls),
        "errors": sum(bool(c.get("error")) for c in write_calls),
        "prompt_tokens": sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in write_calls),
        "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0) for c in write_calls),
        "cache_hit_tokens": sum((c.get("usage") or {}).get("cache_hit_tokens", 0) for c in write_calls),
        "usage_usd": str(sum(float(c.get("usage_usd") or 0) for c in write_calls)),
        "op_counts": dict(store.op_counts),
        "active_memories": len(store.memories),
        "raw_statements": len(statements),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(cache_path, {
            "complete": True,
            "total_statements": len(statements),
            "store": store.to_dict(),
            "write_usage": usage_summary,
        })
        if partial_path is not None and partial_path.exists():
            partial_path.unlink()
    return store, usage_summary
