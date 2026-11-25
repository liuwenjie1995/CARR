"""Automatic query plans and relation-first, locally grounded evidence retrieval."""

import json
import re
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from src.factual_v2 import Observation, norm
from src.factual_v3 import expanded_rule
from src.v2_retrieval import DenseIndex, QUERY_PREFIX, normalize_rows

EXTRA = (
    (r"The name of the current head of state in (.+?) is (.+)\.", "head_of_state"),
    (r"The name of the current head of the (.+?) government is (.+)\.", "head_of_state"),
    (r"The (?:origianl|original) broadcaster of (.+?) is (.+)\.", "broadcaster"),
    (r"(.+?)(?:'s|’s) child is (.+)\.", "child"),
)
CUES = {
    "capital": r"\bcapital\b", "continent": r"\bcontinent\b",
    "citizenship": r"\bcitizen(?:ship)?\b|\bnationality\b|\bbelongs to\b",
    "playing_position": r"\bplaying position\b|\bposition\b",
    "sport": r"\bsport\b", "origin_country": r"\boriginated\b|\borigin\b|\bfirst came from\b",
    "educated_at": r"\beducational institution\b|\beducation\b|\beducat\w*\b",
    "author": r"\bauthor\b", "director": r"\bdirector\b",
    "death_city": r"\bpass(?:ed)? away\b|\bdied\b|\bdeath\b",
    "spouse": r"\bspouse\b|\bpartner\b|\bmarried\b",
    "head_of_state": r"\bchief of state\b|\bhead of state\b",
    "child": r"\bchild\b", "religion": r"\breligion\b",
    "performer": r"\bperformer\b|\bperformed\b", "field": r"\bjob title\b|\boccupation\b",
    "chairperson": r"\bchairperson\b|\bchairman\b", "known_for": r"\bknown for\b|\bfamous for\b",
    "broadcaster": r"\bbroadcaster\b",
}
BRIDGES = {
    "sport": ("playing_position", "sport"),
    "origin_country": ("author", "known_for", "origin_country"),
    "director": ("broadcaster", "director"),
}
ALIASES = {"country": "citizenship", "occupation": "field", "job_title": "field",
           "education": "educated_at", "place_of_death": "death_city", "country_of_origin": "origin_country",
           "capital_city": "capital", "position": "playing_position"}
MAX_FACTS = 4


def structured(record):
    for pattern, relation in EXTRA:
        if match := re.fullmatch(pattern, record.text):
            return Observation(record.index, record.order, *[match[1], relation, match[2]], "assert",
                               (record.start, record.start + len(record.text)), "local")
    return expanded_rule(record)


def anchor_from(question, entities):
    text = norm(question)
    # Prefer the longest known entity, e.g. 'The Birth of Tragedy' over 'Birth'.
    matches = [entity for entity in entities if re.search(r"(?<!\w)" + re.escape(norm(entity)) + r"(?!\w)", text)]
    return max(matches, key=lambda name: (len(norm(name)), name), default=None)


def local_plan(question, entities):
    anchor = anchor_from(question, entities)
    found = [(rel, match.start()) for rel, regex in CUES.items() if (match := re.search(regex, question, re.I))]
    # These questions generally phrase the target outermost. Relative clauses
    # such as "... of the country where X is a citizen" name the intermediate.
    relations = [rel for rel, _ in sorted(found, key=lambda entry: entry[1], reverse=True)]
    if "sport" in relations and "origin_country" in relations:
        relations.remove("origin_country")
        relations.insert(relations.index("sport") + 1, "origin_country")
    if "child" in relations and "author" in relations:
        relations.remove("child")
        relations.append("child")
    return {"start": anchor, "relations": relations[:MAX_FACTS]}


def model_messages(question):
    return [{"role": "system", "content":
             'Decompose this question only. Return JSON {"start":"entity copied from the question",'
             '"relations":["first relation","next relation"]}. Start at the named entity and list up to '
             'four relations in traversal order from it to the requested answer. Each relation starts at '
             'the previous result. Do not answer the question or add inferred names. Use these labels '
             'when applicable: ' + ", ".join(CUES) + ". For a missing direct edge, the local retriever "
             "can try a short bridge; avoid adding steps just because they sound plausible."},
            {"role": "user", "content": question}]


def model_plan(response, question):
    if response["error"] or response["finish"] != "stop":
        return {"start": None, "relations": [], "reason": response["error"] or "incomplete_plan"}
    try:
        value = json.loads(response["answer"])
        start, relations = value["start"], value["relations"]
        if not isinstance(start, str) or not norm(start) or norm(start) not in norm(question):
            raise ValueError("start_not_in_question")
        if not isinstance(relations, list) or not 1 <= len(relations) <= MAX_FACTS:
            raise ValueError("invalid_relation_count")
        if not all(isinstance(rel, str) for rel in relations):
            raise ValueError("invalid_relation_type")
        relations = [norm(rel).replace(" ", "_") for rel in relations]
        relations = [ALIASES.get(rel, rel) for rel in relations]
        if any(rel not in CUES for rel in relations):
            raise ValueError("unsupported_relation")
        return {"start": start, "relations": relations}
    except (ValueError, TypeError, KeyError):
        return {"start": None, "relations": [], "reason": "invalid_model_plan"}


class Store:
    def __init__(self, records, original_lines, embedder):
        self.records = {record.order: record for record in records}
        self.lines = {record.order: original_lines[record.index] for record in records}
        self.index = defaultdict(list)
        self.facts = {}
        self.model = embedder
        for record in records:
            if fact := structured(record):
                self.index[fact.key].append(fact)
                self.facts[fact.order] = fact
        self.dense = DenseIndex(embedder, "\n".join(self.lines.values()))
        self.multiple = {}

    def step(self, entity, relation):
        key = norm(entity), relation
        query = f"{entity} {relation.replace('_', ' ')}"
        if key in self.multiple:
            return self.multiple[key]
        candidates = self.index.get(key, [])
        channel = "entity_relation"
        passage_ids = None
        if not candidates:
            channel = "bge_top8"
            visible, passage_ids = self.dense.search(query)
            for line in visible.splitlines():
                match = re.match(r"(\d+)\.", line)
                if not match:
                    continue
                fact = self.facts.get(int(match[1]))
                if fact and fact.relation == relation and fact.key[0].removeprefix("the ") == key[0].removeprefix("the "):
                    candidates.append(fact)
        if len(candidates) > 1:
            vectors = normalize_rows(list(self.model.embed(
                [QUERY_PREFIX + query] + [self.lines[fact.order] for fact in candidates],
            )))
            scores = vectors[1:] @ vectors[0]
            candidates = [candidates[i] for i in sorted(
                range(len(candidates)), key=lambda i: (-float(scores[i]), -candidates[i].order),
            )]
        detail = {"entity": entity, "relation": relation, "channel": channel, "passages": passage_ids,
                  "candidates": [asdict(fact) for fact in candidates]}
        result = (candidates[0] if candidates else None), detail
        self.multiple[key] = result
        return result


def retrieve(store, plan):
    entity, steps = plan["start"], plan["relations"]
    events, visited, proof = [], set(), []
    queue = [(relation, True) for relation in steps]
    if not entity or not queue:
        return {"status": plan.get("reason", "could_not_decompose"), "events": [], "fact_ids": []}
    status = "completed_route"
    while queue:
        relation, allow_bridge = queue.pop(0)
        fact, event = store.step(entity, relation)
        event = {**event, "chosen": fact.order if fact else None}
        events.append(event)
        if not fact and allow_bridge and relation in BRIDGES and len(proof) + len(queue) + len(BRIDGES[relation]) <= MAX_FACTS:
            event["bridge"] = list(BRIDGES[relation])
            queue = [(part, False) for part in BRIDGES[relation]] + queue
            continue
        if not fact:
            status = "no_matching_fact"
            break
        if len(proof) >= MAX_FACTS or fact.order in visited:
            status = "path_limit_or_cycle"
            break
        visited.add(fact.order)
        proof.append(fact.order)
        entity = fact.value
    return {"status": status, "events": events, "fact_ids": proof, "last_value": entity}
