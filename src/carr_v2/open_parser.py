"""Open-relation title-safe parser, conservative archiver v2, and dialogue fact extractor."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from typing import Callable

from src.evaluator import dump_json
from src.factual_v2 import (
    AMBIGUOUS, NEGATED, Observation, Record, canon, norm, projection, records_for, rule_observation,
)
from src.hop_retrieval import structured
from src.run_unified_dev import PINNED_FACT_PATTERNS

# Case-sensitive lowercase predicate negation: catches "never lived", "might be",
# while preserving capitalized proper titles like "Tomorrow Never Knows" or "May Day".
PREDICATE_NEGATED = re.compile(r"\b(?:not|never|might|may|possibly|rumou?r(?:ed)?)\b")
CAP_NEGATED_WORD = re.compile(r"\b(Not|Never|Might|May|Possibly|Rumou?r(?:ed)?)\b")

V2_EXTRA_PATTERNS: tuple[tuple[str, str], ...] = (
    *PINNED_FACT_PATTERNS,
    (r"(.+?) was created by (.+)\.", "creator"),
    (r"(.+?) worked in the city of (.+)\.", "work_city"),
    (r"The head coach of (.+?) is (.+)\.", "head_coach"),
    (r"The original language of (.+?) is (.+)\.", "original_language"),
    (r"The official language of (.+?) is (.+)\.", "official_language"),
    (r"The company that produced (.+?) is (.+)\.", "manufacturer"),
    (r"The (?:Prime Minister|President|Governor|Mayor) of (.+?) is (.+)\.", "head_of_government"),
    (r"The religion of (.+?) is (.+)\.", "religion"),
    (r"(.+?) works in the area of (.+)\.", "field"),
    (r"(.+?) is a member of (.+)\.", "member_of"),
)


def _strip_cap_negation(record: Record) -> tuple[Record, bool]:
    if not NEGATED.search(record.text) or PREDICATE_NEGATED.search(record.text):
        return record, False
    masked = CAP_NEGATED_WORD.sub(lambda m: f"__CAP_{m.group(1)}__", record.text)
    return Record(record.index, masked, record.start, record.order), True


def _restore_cap(text: str) -> str:
    return re.sub(r"__CAP_([A-Za-z]+)__", r"\1", text)


def parse_numbered_v2(record: Record) -> Observation | None:
    """Parse numbered FactConsolidation line with title-safe negation and extended templates."""
    if PREDICATE_NEGATED.search(record.text):
        return None
    work_rec, masked = _strip_cap_negation(record)

    fact = structured(work_rec)
    if fact is None:
        for regex, relation in V2_EXTRA_PATTERNS:
            if match := re.fullmatch(regex, work_rec.text):
                entity, value = match.groups()
                if entity.strip() and value.strip():
                    op = "unknown" if AMBIGUOUS.search(value) else "assert"
                    fact = Observation(
                        work_rec.index, work_rec.order, entity, relation, value, op,
                        (work_rec.start, work_rec.start + len(work_rec.text)), "v2_rule",
                    )
                    break
    if fact is None:
        return None
    if masked:
        return Observation(
            fact.index, fact.order, _restore_cap(fact.entity), fact.relation,
            _restore_cap(fact.value), fact.operation, fact.span, "v2_title_safe",
        )
    return fact


def archive_v2(history: str, protocol: str) -> tuple[dict[int, Record], str, list[dict]]:
    """Conservative archiving v2: preserves non-whitelisted relations and zero-deletes EventQA."""
    records, duplicates = records_for(history, protocol)
    parse = parse_numbered_v2 if protocol == "numbered" else rule_observation
    facts = [fact for record in records.values() if (fact := parse(record))]
    clean, decisions = projection(history, facts, protocol, duplicate_ids=duplicates)
    return records, clean, decisions


def parse_longmemeval_sessions(context: str) -> list[dict]:
    """Parse raw MemoryAgentBench LongMemEval context string into chronological sessions."""
    raw = ast.literal_eval(context)
    if not isinstance(raw, list):
        raise ValueError("Expected list representation for LongMemEval context.")
    sessions: list[dict] = []
    cur_time = "Unknown"
    for item in raw:
        if isinstance(item, str) and item.startswith("Chat Time:"):
            cur_time = item[len("Chat Time:"):].strip()
        elif isinstance(item, list):
            turns = []
            for turn in item:
                if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                    turns.append({
                        "role": str(turn.get("role", "user")),
                        "content": turn["content"].strip(),
                    })
            sessions.append({
                "session_idx": len(sessions),
                "chat_time": cur_time,
                "turns": turns,
            })
    return sessions


def format_longmemeval_transcript(sessions: list[dict]) -> str:
    """Format chronological sessions into line-per-turn text for indexing & retrieval."""
    lines: list[str] = []
    for sess in sessions:
        sid = sess["session_idx"]
        ts = sess["chat_time"]
        for t_idx, turn in enumerate(sess["turns"]):
            role = turn["role"].capitalize()
            clean_body = " ".join(turn["content"].split())
            if clean_body:
                lines.append(f"[S{sid:03d}T{t_idx:02d} | {ts}] {role}: {clean_body}")
    return "\n".join(lines)


def _extract_session_prompt(session: dict) -> list[dict]:
    sid = session["session_idx"]
    ts = session["chat_time"]
    turn_lines = []
    for t_idx, turn in enumerate(session["turns"]):
        role = turn["role"].capitalize()
        body = " ".join(turn["content"].split())
        if len(body) > 1200:
            body = body[:1200] + " ..."
        turn_lines.append(f"[T{t_idx:02d}] {role}: {body}")
    transcript = "\n".join(turn_lines)
    if len(transcript) > 6500:
        transcript = transcript[:6500] + "\n..."
    return [
        {
            "role": "system",
            "content": (
                "Extract durable factual, preference, event, count, or state-update statements from this conversation session. "
                "Return JSON {\"facts\": [{\"entity\": \"user or subject\", \"attribute\": \"short snake_case attribute\", "
                "\"value\": \"concise value\", \"operation\": \"assert|replace\", \"quote\": \"exact substring from session\"}]}. "
                "Rules: (1) Only include concrete user/world facts, dates, counts, plans, or updated preferences; skip generic assistant advice. "
                "(2) Mark operation='replace' if the user updates/changes a prior state, otherwise 'assert'. "
                "(3) quote MUST be copied verbatim from the session text. Return {\"facts\": []} if none."
            ),
        },
        {
            "role": "user",
            "content": f"Session {sid} | Chat Time: {ts}\n{transcript}",
        },
    ]


def _verify_session_facts(raw_answer: str, session: dict) -> list[dict]:
    try:
        data = json.loads(raw_answer)
        items = data.get("facts", [])
        if not isinstance(items, list):
            return []
    except (ValueError, TypeError, AttributeError):
        return []
    full_norm = norm(" ".join(turn["content"] for turn in session["turns"]))
    verified = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ent = str(item.get("entity", "")).strip()
        attr = str(item.get("attribute", "")).strip()
        val = str(item.get("value", "")).strip()
        op = str(item.get("operation", "assert")).strip().lower()
        quote = str(item.get("quote", "")).strip()
        if not ent or not attr or not val or not quote:
            continue
        if op not in {"assert", "replace"}:
            op = "assert"
        # Span verification: normalized quote (or its first 32 chars) must ground in session turns
        nq = norm(quote)
        if not nq or (nq not in full_norm and (len(nq) < 20 or nq[:32] not in full_norm)):
            continue
        verified.append({
            "session_idx": session["session_idx"],
            "chat_time": session["chat_time"],
            "entity": ent,
            "attribute": canon(attr),
            "value": val,
            "operation": op,
            "quote": quote,
        })
    return verified


def extract_dialogue_facts_batch(
    sessions: list[dict],
    chat_fn: Callable[[str, list[dict], int, bool], dict],
    *,
    cache_path: Path | None = None,
    max_workers: int = 10,
) -> tuple[list[dict], dict]:
    """Concurrently extract and span-verify structured facts from LongMemEval sessions."""
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("session_count") == len(sessions):
            return cached["facts"], cached["usage"]

    per_session: dict[int, list[dict]] = {}
    calls: list[dict] = []

    def run_one(sess: dict) -> tuple[int, list[dict], dict]:
        msgs = _extract_session_prompt(sess)
        res = chat_fn(f"lme_extract_s{sess['session_idx']:03d}", msgs, 512, True)
        facts = _verify_session_facts(res.get("answer") or "", sess) if not res.get("error") else []
        return sess["session_idx"], facts, res

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(run_one, s) for s in sessions]
        for fut in as_completed(futs):
            sid, facts, res = fut.result()
            per_session[sid] = facts
            calls.append(res)

    ordered_facts = [fact for sid in sorted(per_session) for fact in per_session[sid]]
    usage_summary = {
        "calls": len(calls),
        "errors": sum(bool(c.get("error")) for c in calls),
        "prompt_tokens": sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in calls),
        "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls),
        "cache_hit_tokens": sum((c.get("usage") or {}).get("cache_hit_tokens", 0) for c in calls),
        "usage_usd": str(sum(float(c.get("usage_usd") or 0) for c in calls)),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(cache_path, {
            "session_count": len(sessions),
            "facts": ordered_facts,
            "usage": usage_summary,
        })
    return ordered_facts, usage_summary
