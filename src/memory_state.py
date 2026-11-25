"""Parse conservative FactConsolidation observations and resolve state updates."""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

RELATION_MARKERS = {
    " is associated with the sport of ": "sport",
    " plays the position of ": "playing_position",
    " is a citizen of ": "citizenship",
    " is married to ": "spouse",
    " was born in the city of ": "birth_city",
    " died in the city of ": "death_city",
    " is located in the continent of ": "continent",
    " was created in the country of ": "origin_country",
}
PREFIX_RELATIONS = (
    "capital", "chairperson", "chief executive officer", "director", "author", "official language",
)
NUMBERED = re.compile(r"(?P<id>\d+)\.\s+(?P<statement>.+)")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


@dataclass(frozen=True)
class Fact:
    record_id: int
    entity: str
    attribute: str
    value: str
    effective_at: int
    confirmed: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return normalize(self.entity), normalize(self.attribute)


def parse_fact(line: str) -> Fact | None:
    """Accept only explicitly supported surface forms. Leave other lines intact."""
    match = NUMBERED.fullmatch(line)
    if not match or not match["statement"].endswith("."):
        return None
    statement = match["statement"][:-1]
    candidate = []
    for marker, attribute in RELATION_MARKERS.items():
        if statement.count(marker) == 1:
            entity, value = statement.split(marker)
            candidate.append((entity, attribute, value))
    for attribute in PREFIX_RELATIONS:
        prefix = f"The {attribute} of "
        if statement.startswith(prefix) and statement[len(prefix):].count(" is ") == 1:
            entity, value = statement[len(prefix):].split(" is ")
            candidate.append((entity, attribute, value))
    if len(candidate) != 1:
        return None
    entity, attribute, value = candidate[0]
    if not entity.strip() or not value.strip():
        return None
    # FactConsolidation defines larger record numbers as later effective order.
    index = int(match["id"])
    return Fact(index, entity, attribute, value, index)


@dataclass(frozen=True)
class Conflict:
    key: tuple[str, str]
    kept_id: int
    archived_ids: tuple[int, ...]


def resolve_conflicts(
    facts: tuple[Fact, ...], *, min_distinct: int = 2, as_of: int | None = None,
) -> tuple[Conflict, ...]:
    """Archive older contradictory confirmed records when the latest value is unambiguous."""
    if min_distinct < 2:
        raise ValueError("A conflict requires at least two distinct values.")
    groups = defaultdict(list)
    for fact in facts:
        if fact.confirmed and (as_of is None or fact.effective_at <= as_of):
            groups[fact.key].append(fact)
    decisions = []
    for key, group in sorted(groups.items()):
        if len({normalize(fact.value) for fact in group}) < min_distinct:
            continue
        latest_date = max(fact.effective_at for fact in group)
        latest = [fact for fact in group if fact.effective_at == latest_date]
        if len({normalize(fact.value) for fact in latest}) != 1:
            continue
        winner = max(latest, key=lambda fact: fact.record_id)
        archived = tuple(sorted(
            fact.record_id for fact in group
            if fact.effective_at < latest_date and normalize(fact.value) != normalize(winner.value)
        ))
        if archived:
            decisions.append(Conflict(key, winner.record_id, archived))
    return tuple(decisions)
