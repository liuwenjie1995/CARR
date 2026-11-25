"""Unified query planner and 3-channel recency-prioritized hybrid hop relay (CARR v2)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import re

from src.carr_v2.open_parser import parse_numbered_v2
from src.carr_v2.tombstone_index import HybridLexicalDenseIndex, TombstoneDenseIndex
from src.factual_v2 import Observation, norm, records_for
from src.hop_retrieval import ALIASES, BRIDGES, CUES, MAX_FACTS, anchor_from
from src.run_unified_dev import DIRECT
from src.v2_retrieval import QUERY_PREFIX, normalize_rows

CANONICAL_RELATIONS = sorted(
    set(DIRECT)
    | {
        "creator",
        "work_city",
        "head_coach",
        "original_language",
        "manufacturer",
        "head_of_government",
        "founder",
        "employer",
        "spoken_language",
        "founding_city",
        "written_language",
        "headquarters_city",
        "birth_city",
        "member_of",
    }
)
CANONICAL_SET = frozenset(CANONICAL_RELATIONS)

ALIASES_V2 = {
    **ALIASES,
    "created_by": "creator",
    "produced_by": "manufacturer",
    "producer": "manufacturer",
    "company": "manufacturer",
    "language": "official_language",
    "spoken_language": "spoken_language",
    "languages_spoken": "spoken_language",
    "work_location": "work_city",
    "worked_in": "work_city",
    "coach": "head_coach",
    "manager": "head_coach",
    "prime_minister": "head_of_government",
    "president": "head_of_government",
    "governor": "head_of_government",
    "mayor": "head_of_government",
    "ceo": "chief_executive_officer",
    "headquarters": "headquarters_city",
    "head_office": "headquarters_city",
    "location_of_head_office": "headquarters_city",
    "birthplace": "birth_city",
    "place_of_birth": "birth_city",
    "school": "educated_at",
    "university": "educated_at",
    "college": "educated_at",
    "notable_work": "known_for",
    "religion_or_worldview": "religion",
    "music_genre": "music_type",
    "genre": "music_type",
    "type_of_music": "music_type",
    "founded_by": "founder",
    "established_by": "founder",
    "nationality": "citizenship",
    "country_of_citizenship": "citizenship",
    "first_came_from": "origin_country",
}


def planner_messages_v2(question: str) -> list[dict]:
    """Compact unified planner prompt supporting single-hop, multi-hop, and open relations."""
    rel_list = ", ".join(CANONICAL_RELATIONS)
    return [
        {
            "role": "system",
            "content": (
                "Decompose the question into 1 to 4 sequential lookup steps without answering it. "
                'Return JSON: {"mode":"single"|"multi"|"open","start":"entity copied verbatim from question or null",'
                '"steps":[{"relation":"label","query":"short lookup phrase"}]}. '
                f"Use these canonical labels whenever applicable: {rel_list}; otherwise use a concise snake_case label. "
                "Rules: (1) Start at the named subject entity and list steps outward to the requested answer "
                "(e.g., 'job title of the chairperson of X' -> ['chairperson', 'field']; "
                "'capital of the country where X first came from' -> ['origin_country', 'capital']). "
                "(2) Do NOT append redundant location steps after 'capital', 'birth_city', or 'death_city' (they already return a city). "
                "(3) Use 'author' for books/written works and 'creator' for characters/shows/games. "
                "(4) Keep plans minimal so the retriever can expand short bridges for missing direct edges."
            ),
        },
        {"role": "user", "content": question},
    ]


def parse_plan_v2(response: dict, question: str, known_entities: list[str] | None = None) -> dict:
    """Validate unified plan permissively without rejecting open-vocabulary relations."""
    if response.get("error") or response.get("finish") != "stop":
        fallback_start = anchor_from(question, known_entities) if known_entities else None
        return {
            "mode": "open",
            "start": fallback_start,
            "steps": [],
            "reason": response.get("error") or "incomplete_plan",
        }
    try:
        data = json.loads(response["answer"])
        mode = str(data.get("mode", "single")).lower()
        if mode not in {"single", "multi", "open"}:
            mode = "single"
        start = data.get("start")
        if isinstance(start, str):
            start = start.strip() or None
        if start is not None and norm(start) not in norm(question):
            # Try grounding via longest known entity rather than throwing away the plan
            grounded = anchor_from(question, known_entities) if known_entities else None
            start = grounded
        elif start is None and known_entities:
            start = anchor_from(question, known_entities)

        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []
        steps = []
        for s in raw_steps[:MAX_FACTS]:
            if isinstance(s, dict):
                rel = norm(str(s.get("relation", ""))).replace(" ", "_")
                rel = ALIASES_V2.get(rel, rel)
                q = str(s.get("query", "")).strip() or rel.replace("_", " ")
                if rel:
                    steps.append({"relation": rel, "query": q})
            elif isinstance(s, str) and s.strip():
                rel = ALIASES_V2.get(norm(s).replace(" ", "_"), norm(s).replace(" ", "_"))
                steps.append({"relation": rel, "query": rel.replace("_", " ")})

        if not steps or not start:
            return {"mode": "open", "start": start, "steps": steps, "reason": "open_fallback"}
        if len(steps) >= 2:
            mode = "multi"
        return {"mode": mode, "start": start, "steps": steps, "reason": "ok"}
    except (ValueError, TypeError, KeyError):
        fallback_start = anchor_from(question, known_entities) if known_entities else None
        return {"mode": "open", "start": fallback_start, "steps": [], "reason": "invalid_json_plan"}


def _entity_variants(entity: str) -> list[str]:
    n = norm(entity)
    stripped = n.removeprefix("the ").strip()
    variants = [n]
    if stripped and stripped != n:
        variants.append(stripped)
    elif n and not n.startswith("the "):
        variants.append("the " + n)
    return variants


class RecencyHybridStore:
    """3-channel recency-prioritized store with invariant-boundary tombstone hybrid fallback."""

    def __init__(
        self,
        records: list,
        original_lines: list[str],
        embedder,
        hybrid_index: HybridLexicalDenseIndex,
        *,
        use_exact: bool = True,
        semantic_threshold: float = 0.50,
    ):
        self.records = {r.order: r for r in records}
        self.lines = {r.order: original_lines[r.index] for r in records}
        self.model = embedder
        self.hybrid = hybrid_index
        self.use_exact = use_exact
        self.semantic_threshold = semantic_threshold

        self.index: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        self.by_entity: dict[str, list[Observation]] = defaultdict(list)
        self.facts: dict[int, Observation] = {}
        for record in records:
            if fact := parse_numbered_v2(record):
                self.index[fact.key].append(fact)
                self.by_entity[norm(fact.entity)].append(fact)
                self.facts[fact.order] = fact
        self.entities = [fact.entity for fact in self.facts.values()]
        self.cache: dict[tuple[str, str, str], tuple[Observation | None, dict]] = {}

    def _lookup_exact(self, entity: str, relation: str) -> list[Observation]:
        for ent_key in _entity_variants(entity):
            cands = self.index.get((ent_key, relation), [])
            if cands:
                return cands
        return []

    def _lookup_entity_facts(self, entity: str) -> list[Observation]:
        out: list[Observation] = []
        for ent_key in _entity_variants(entity):
            out.extend(self.by_entity.get(ent_key, []))
        return out

    def step(
        self,
        entity: str,
        relation: str,
        sub_query: str = "",
        *,
        allow_fallback: bool = True,
    ) -> tuple[Observation | None, dict]:
        cache_key = (norm(entity), relation, norm(sub_query), allow_fallback)
        if cache_key in self.cache:
            return self.cache[cache_key]

        exact = self._lookup_exact(entity, relation) if self.use_exact else []
        passages = None
        candidates: list[Observation] = []
        channel = "exact_miss"
        if exact:
            channel = "exact_latest"
            candidates = sorted(exact, key=lambda f: -f.order)
        elif not allow_fallback:
            return None, {
                "entity": entity,
                "relation": relation,
                "sub_query": sub_query,
                "channel": "exact_miss",
                "passages": None,
                "candidates": [],
                "chosen": None,
            }
        elif relation not in CANONICAL_SET or relation in {"creator", "author", "manufacturer", "developer", "original_language", "official_language"}:
            # Channel 2: Entity-anchored semantic-recency lookup for open/unmapped or close-pair relations
            ent_facts = self._lookup_entity_facts(entity) if self.use_exact else []
            matched_rel_cands: list[Observation] = []
            if ent_facts:
                qtext = f"{entity} {sub_query or relation.replace('_', ' ')}"
                vecs = normalize_rows(list(self.model.embed(
                    [QUERY_PREFIX + qtext] + [self.lines[f.order] for f in ent_facts]
                )))
                sims = vecs[1:] @ vecs[0]
                best_idx = max(range(len(ent_facts)), key=lambda i: (float(sims[i]), ent_facts[i].order))
                min_sim = self.semantic_threshold if relation not in CANONICAL_SET else max(self.semantic_threshold, 0.62)
                if float(sims[best_idx]) >= min_sim:
                    best_rel = ent_facts[best_idx].relation
                    matched_rel_cands = sorted(
                        [f for f in ent_facts if f.relation == best_rel],
                        key=lambda f: -f.order,
                    )
            if matched_rel_cands:
                channel = "entity_semantic_recency"
                candidates = matched_rel_cands

        if not candidates and allow_fallback:
            # Channel 3: Tombstone-masked hybrid lexical+dense fallback
            channel = "tombstone_hybrid_fallback"
            query = f"{entity} {sub_query or relation.replace('_', ' ')}"
            visible, passages, _ = self.hybrid.search(query)
            candidates = []
            ent_forms = set(_entity_variants(entity))
            for line in visible.splitlines():
                m = re.match(r"(\d+)\.", line)
                fact = self.facts.get(int(m[1])) if m else None
                if fact and any(v in ent_forms for v in _entity_variants(fact.entity)):
                    if fact.relation == relation:
                        candidates.append(fact)
            if len(candidates) > 1:
                vecs = normalize_rows(list(self.model.embed(
                    [QUERY_PREFIX + query] + [self.lines[f.order] for f in candidates]
                )))
                sims = vecs[1:] @ vecs[0]
                candidates = [
                    candidates[i]
                    for i in sorted(range(len(candidates)), key=lambda i: (-float(sims[i]), -candidates[i].order))
                ]

        chosen = candidates[0] if candidates else None
        detail = {
            "entity": entity,
            "relation": relation,
            "sub_query": sub_query,
            "channel": channel,
            "passages": passages,
            "candidates": [asdict(c) for c in candidates],
            "chosen": chosen.order if chosen else None,
        }
        result = (chosen, detail)
        self.cache[cache_key] = result
        return result


def whole_question_hybrid_fallback(
    store: RecencyHybridStore,
    question: str,
    max_lines: int = 8,
    max_bytes: int = 1800,
) -> tuple[list[str], dict]:
    """Tombstone-masked hybrid fallback prioritized by matched entity and descending recency."""
    visible, passages, _ = store.hybrid.search(question)
    rows = [line for line in visible.splitlines() if re.match(r"^\d+\.", line)]
    entity = anchor_from(question, store.entities)
    if entity:
        needle = r"(?<!\w)" + re.escape(norm(entity)) + r"(?!\w)"
        rows.sort(
            key=lambda line: (
                not bool(re.search(needle, norm(line))),
                -int(line.split(".", 1)[0]),
            )
        )
    evidence: list[str] = []
    used = 0
    for line in rows:
        b = len(line.encode("utf-8")) + 1
        if evidence and used + b > max_bytes:
            continue
        evidence.append(line)
        used += b
        if len(evidence) >= max_lines:
            break
    # Keep ascending order when passing to official prompt for natural serial reading
    evidence.sort(key=lambda line: int(line.split(".", 1)[0]))
    return evidence, {
        "status": "hybrid_fallback_text",
        "passages": passages,
        "fact_ids": [int(line.split(".", 1)[0]) for line in evidence],
        "events": [],
    }


def retrieve_v2(
    store: RecencyHybridStore,
    plan: dict,
    question: str,
    *,
    max_hops: int = MAX_FACTS,
) -> tuple[list[str], dict]:
    """Execute unified plan over RecencyHybridStore with bounded bridging and clean fallback."""
    entity = plan.get("start")
    steps = plan.get("steps") or []
    if not entity or not steps:
        return whole_question_hybrid_fallback(store, question)

    queue: list[tuple[str, str, bool]] = [(s["relation"], s["query"], True) for s in steps]
    events, visited, proof = [], set(), []
    status = "completed_route"
    while queue:
        rel, sub_q, allow_bridge = queue.pop(0)
        can_bridge = allow_bridge and rel in BRIDGES and len(proof) + len(queue) + len(BRIDGES[rel]) <= max_hops
        fact, event = store.step(entity, rel, sub_q, allow_fallback=not can_bridge)
        events.append(event)
        if not fact and can_bridge:
            event["bridge"] = list(BRIDGES[rel])
            queue = [(part, part.replace("_", " "), False) for part in BRIDGES[rel]] + queue
            continue
        if not fact:
            status = "no_matching_fact"
            break
        if len(proof) >= max_hops or fact.order in visited:
            status = "path_limit_or_cycle"
            break
        visited.add(fact.order)
        proof.append(fact.order)
        entity = fact.value

    if not proof:
        ev, fb_trace = whole_question_hybrid_fallback(store, question)
        return ev, {**fb_trace, "relay_status": status, "events": events}

    return [store.lines[num] for num in proof], {
        "status": status,
        "events": events,
        "fact_ids": proof,
        "last_value": entity,
    }
