"""Evidence-backed extraction and conservative projection of original fact lines."""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from src.memory_state import parse_fact

NUMBERED = re.compile(r"(?P<order>[0-9]+)\.\s+(?P<text>.+)")
DATED = re.compile(r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\s*[:|]\s*(?P<text>.+)")
RESIDENCE = re.compile(r"(?P<entity>.+?) (?P<marker>(?:now )?(?:lives in|resides in)) (?P<value>.+?)\.")
REPLACEMENT = re.compile(r"\b(?:now (?:lives in|resides in|is|uses|has)|changed to|updated to|replaced (?:by|with))\s+", re.I)
AMBIGUOUS = re.compile(r"\b(?:or|and|as well as|among|one of)\b|;| / ", re.I)
NEGATED = re.compile(r"\b(?:not|never|might|may|possibly|rumou?r(?:ed)?)\b", re.I)
ALIASES = {
    "lives in": "residence", "resides in": "residence", "residence": "residence",
    "current address": "address", "address": "address",
    "capital city": "capital", "country of citizenship": "citizenship", "nationality": "citizenship",
    "birthplace": "birth_city", "place of birth": "birth_city", "death place": "death_city",
    "playing position": "playing_position", "position": "playing_position", "plays the position of": "playing_position",
    "sport": "sport", "associated sport": "sport", "country of origin": "origin_country",
    "origin country": "origin_country", "official language": "official_language",
    "chief executive officer": "chief_executive_officer", "ceo": "chief_executive_officer",
    "married to": "spouse", "favorite hobbies": "hobby", "hobbies": "hobby",
}
SINGLE_SLOTS = frozenset({
    "address", "residence", "capital", "birth_city", "death_city", "continent", "origin_country",
    "playing_position", "sport", "chairperson", "chief_executive_officer",
})
NARRATIVE_SINGLE_SLOTS = frozenset({"address", "residence", "chief_executive_officer"})


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canon(text):
    key = norm(text).replace("_", " ")
    return ALIASES.get(key, key.replace(" ", "_"))


def replacement_supports(evidence, value):
    return any(re.match(re.escape(value) + r"(?=$|[\s.,;!?])", evidence[marker.end():])
               for marker in REPLACEMENT.finditer(evidence))


@dataclass(frozen=True)
class Record:
    index: int
    text: str
    start: int
    order: int


@dataclass(frozen=True)
class Observation:
    index: int
    order: int
    entity: str
    relation: str
    value: str
    operation: str
    span: tuple[int, int]
    origin: str

    @property
    def key(self):
        return norm(self.entity), self.relation


def records_for(history, protocol):
    result, orders, duplicate = {}, set(), False
    offset = 0
    for index, line in enumerate(history.splitlines(keepends=True)):
        text = line.rstrip("\r\n")
        match = (NUMBERED if protocol == "numbered" else DATED).fullmatch(text)
        if match:
            try:
                order = int(match["order"]) if protocol == "numbered" else date.fromisoformat(match["date"]).toordinal()
            except ValueError:
                match = None
        if match:
            duplicate |= protocol == "numbered" and order in orders
            orders.add(order)
            result[index] = Record(index, match["text"], offset + match.start("text"), order)
        offset += len(line)
    return result, duplicate


def rule_observation(record):
    body = record.text
    if NEGATED.search(body):
        return None
    if match := RESIDENCE.fullmatch(body):
        entity, relation, value = match["entity"], "residence", match["value"]
    elif fact := parse_fact(f"{record.order}. {body}"):
        entity, relation, value = fact.entity, canon(fact.attribute), fact.value
    else:
        return None
    operation = "unknown" if AMBIGUOUS.search(value) else ("replace" if replacement_supports(body, value) else "assert")
    return Observation(record.index, record.order, entity, relation, value, operation,
                       (record.start, record.start + len(body)), "rule")


def validate_model(item, record):
    if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] != record.index:
        return None
    fields = ("entity", "relation", "value", "operation", "evidence")
    if not all(isinstance(item.get(key), str) and item[key].strip() for key in fields):
        return None
    entity, relation, value, operation, evidence = (item[key].strip() for key in fields)
    if record.text.count(evidence) != 1 or entity not in evidence or value not in evidence:
        return None
    if operation not in {"assert", "replace", "add", "remove", "unknown"}:
        return None
    if operation == "replace" and not replacement_supports(evidence, value):
        return None
    if NEGATED.search(evidence) or AMBIGUOUS.search(value):
        operation = "unknown"
    start = record.start + record.text.index(evidence)
    return Observation(record.index, record.order, entity, canon(relation), value, operation,
                       (start, start + len(evidence)), "model")


def projection(history, facts, protocol, *, archive=True, duplicate_ids=False):
    groups = defaultdict(list)
    for item in facts:
        groups[item.key].append(item)
    decisions = []
    slots = SINGLE_SLOTS if protocol == "numbered" else NARRATIVE_SINGLE_SLOTS
    if archive and not duplicate_ids:
        for key, group in sorted(groups.items()):
            if key[1] not in slots or any(f.operation not in {"assert", "replace"} for f in group):
                continue
            order = max(f.order for f in group)
            latest = [f for f in group if f.order == order]
            if len({norm(f.value) for f in latest}) != 1:
                continue
            if protocol != "numbered" and not any(f.operation == "replace" for f in latest):
                continue
            winner = min(latest, key=lambda f: f.index)
            for old in group:
                if old.order < order and norm(old.value) != norm(winner.value):
                    decisions.append({
                        "removed_line": old.index, "winner_line": winner.index, "key": key,
                        "old_span": old.span, "new_span": winner.span,
                        "rule": "dataset_numbering" if protocol == "numbered" else "explicit_dated_replacement",
                    })
    removed = {row["removed_line"] for row in decisions}
    visible = "".join(line for i, line in enumerate(history.splitlines(keepends=True)) if i not in removed)
    return visible, decisions
