"""Local relation coverage, entity-gated fallback, and traceable fact projection."""

import json
import re
from collections import Counter
from dataclasses import asdict

from src.factual_v2 import (
    AMBIGUOUS, NEGATED, Observation, norm, projection, records_for, rule_observation, validate_model,
)
from src.v2_extract import BATCH, OUTPUT_CAP, messages
from src.v2_transport import FLASH

MARKERS = {
    " was founded by ": "founder",
    " is affiliated with the religion of ": "religion",
    " was performed by ": "performer",
    " is employed by ": "employer",
    " speaks the language of ": "spoken_language",
    " is famous for ": "known_for",
    " was founded in the city of ": "founding_city",
    " works in the field of ": "field",
    " was written in the language of ": "written_language",
}
HEADQUARTERS = re.compile(r"The headquarters of (.+?) is located in the city of (.+)\.")
UNIVERSITY = re.compile(r"The (?:university|univeristy) where (.+?) was educated is (.+)\.")
POSSESSIVE = re.compile(r"The [^\n]+? of (.+?) (?:is|was|are|were) .+")
PREDICATE = re.compile(r"(.+?) (?:is|was|are|were|has|had|works|worked|speaks|plays|lives|resides|died|changed|moved|uses) .+")


def expanded_rule(record):
    if fact := rule_observation(record):
        return fact
    text = record.text
    if not text.endswith(".") or NEGATED.search(text):
        return None
    candidates = [
        (*text[:-1].split(marker), relation) for marker, relation in MARKERS.items() if text.count(marker) == 1
    ]
    for pattern, relation in ((HEADQUARTERS, "headquarters_city"), (UNIVERSITY, "educated_at")):
        if match := pattern.fullmatch(text):
            candidates.append((*match.groups(), relation))
    if len(candidates) != 1:
        return None
    entity, value, relation = candidates[0]
    if not entity.strip() or not value.strip():
        return None
    # Recognition does not declare an attribute single-valued. The archive policy
    # remains the existing conservative allowlist; these new relations are retained.
    return Observation(record.index, record.order, entity, relation, value,
                       "unknown" if AMBIGUOUS.search(value) else "assert",
                       (record.start, record.start + len(text)), "rule")


def subject(text):
    pattern = POSSESSIVE if text.startswith("The ") and " of " in text else PREDICATE
    if not (match := pattern.fullmatch(text)):
        return None
    name = norm(match[1]).strip(" \t'\"")
    if len(name) < 2 or len(name.split()) > 12 or name in {"he", "she", "it", "they", "we", "you", "this", "that"}:
        return None
    return name


def plan_extraction(history, *, entity_gate=True):
    records, duplicate = records_for(history, "numbered")
    found = {index: fact for index, record in records.items() if (fact := expanded_rule(record))}
    misses = [record for index, record in records.items() if index not in found]
    candidates = {record.index: subject(record.text) for record in misses}
    terms = {name for name in candidates.values() if name is not None}
    # Match whole entity names and count each record once per entity.
    frequencies = Counter()
    if terms:
        patterns = {name: re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)") for name in terms}
        texts = [norm(record.text) for record in records.values()]
        frequencies.update({name: sum(bool(pattern.search(text)) for text in texts)
                            for name, pattern in patterns.items()})
    reasons, pending = {}, []
    for record in misses:
        name = candidates[record.index]
        count = frequencies.get(name, 0)
        if duplicate:
            reason = "duplicate_numbering"
        elif not entity_gate:
            reason = "eligible_gate_disabled"
        elif name is None:
            reason = "unknown_subject"
        elif count < 2:
            reason = "too_few_records"
        else:
            reason = "eligible"
        reasons[record.index] = {"subject": name, "records_mentioning_entity": count, "decision": reason}
        if reason.startswith("eligible"):
            pending.append(record)
    return records, duplicate, found, pending, reasons


def batches(pending):
    return (pending[start:start + BATCH] for start in range(0, len(pending), BATCH))


def parse_batch(task, batch, reasons, entity_gate):
    try:
        if task["error"] or task["finish"] != "stop":
            return {}, task["error"] or "truncated_response"
        data = json.loads(task["answer"])
        keys = [key for key in ("records", "facts") if isinstance(data, dict) and key in data]
        if len(keys) != 1 or not isinstance(data[keys[0]], list):
            return {}, "invalid_schema"
    except (TypeError, ValueError):
        return {}, "invalid_json"
    items, duplicates = {}, set()
    for item in data[keys[0]]:
        if isinstance(item, dict) and type(index := item.get("id")) is int:
            if index in items:
                duplicates.add(index)
            items[index] = item
    accepted = {}
    for record in batch:
        if record.index in duplicates:
            continue
        fact = validate_model(items.get(record.index), record)
        if fact and (not entity_gate or norm(fact.entity) == reasons[record.index]["subject"]):
            accepted[record.index] = fact
    return accepted, "ok"


def prepare(history, plan, chat=None, save=None, *, entity_gate=True):
    records, duplicate, rules, pending, reasons = plan
    resolved = dict(rules)
    calls = []
    if chat:
        for batch in batches(pending):
            task = chat(FLASH, messages(batch), OUTPUT_CAP, "extract", json_output=True)
            parsed, status = parse_batch(task, batch, reasons, entity_gate)
            resolved.update(parsed)
            task.update({"extraction_status": status, "accepted": [asdict(fact) for fact in parsed.values()],
                         "submitted_lines": [item.index for item in batch]})
            save(task)
            calls.append(task["id"])
    rule_view, rule_archive = projection(history, list(rules.values()), "numbered", duplicate_ids=duplicate)
    hybrid_view, hybrid_archive = projection(history, list(resolved.values()), "numbered", duplicate_ids=duplicate)
    return rule_view, hybrid_view, {
        "numbered_records": len(records), "rule_hits": len(rules), "rule_misses": len(records) - len(rules),
        "eligible_for_model": len(pending), "model_accepted": len(resolved) - len(rules),
        "model_executed": chat is not None, "extraction_calls": calls, "entity_gate": entity_gate,
        "dispositions": dict(Counter(item["decision"] for item in reasons.values())),
        "line_decisions": reasons, "rule_archive": rule_archive, "hybrid_archive": hybrid_archive,
    }
