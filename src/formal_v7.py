"""Frozen local preparation for the A-group formal experiment."""

from collections import defaultdict
from hashlib import sha256

from src.factual_v2 import projection, records_for, rule_observation
from src.factual_v3 import expanded_rule
from src.run_unified_dev import augment_single_index, route, unified_evidence
from src.run_hop_recency_dev import RecencyStore
from src.v2_retrieval import DenseIndex, matched_recent

FACT_ARMS = ("official_full_context", "length_matched_recent_tail", "plain_bge_top8",
             "rule_archive_bge_top8", "raw_exact_latest_relay", "unified_a")
EVENT_ARMS = ("eventqa_original", "eventqa_rule_archive")
WARMUP = 2
CONCURRENCY = 8


def archive(history, protocol):
    records, duplicates = records_for(history, protocol)
    parse = expanded_rule if protocol == "numbered" else rule_observation
    facts = [fact for record in records.values() if (fact := parse(record))]
    clean, decisions = projection(history, facts, protocol, duplicate_ids=duplicates)
    return records, clean, decisions


class FactHistory:
    def __init__(self, history, embedder):
        self.history = history
        records, self.clean, self.archive = archive(history, "numbered")
        removed = {item["removed_line"] for item in self.archive}
        lines = history.splitlines()
        self.raw = RecencyStore(list(records.values()), lines, embedder, use_exact=True)
        self.pruned = (RecencyStore([record for idx, record in records.items() if idx not in removed],
                                    lines, embedder, use_exact=True) if removed else self.raw)
        for store in (self.raw,) if self.raw is self.pruned else (self.raw, self.pruned):
            augment_single_index(store)
        self.entities = [fact.entity for fact in self.raw.facts.values()]
        self.raw_bge = DenseIndex(embedder, history)
        self.clean_bge = self.raw_bge if self.clean == history else DenseIndex(embedder, self.clean)

    def route(self, question):
        # Share the routing decision with the archive-off ablation.
        return route(question, self.entities)

    def prompts(self, sample, baseline, decision, plan):
        question = sample["question"]
        saved_plan = {"plan": plan} if plan is not None else None
        raw_evidence, raw_trace = unified_evidence(self.raw, question, decision, saved_plan)
        evidence, trace = unified_evidence(self.pruned, question, decision, saved_plan)
        compact = "\n".join(evidence)
        # Match the official formatted input, including its fixed instruction overhead.
        def serialized(text):
            return "\n".join(message["content"] for message in baseline.messages(sample, text))

        recent = matched_recent(self.history, compact, serialized)
        raw_bge, raw_ids = self.raw_bge.search(question)
        clean_bge, clean_ids = self.clean_bge.search(question)
        contexts = (self.history, recent, raw_bge, clean_bge, "\n".join(raw_evidence), compact)
        return dict(zip(FACT_ARMS, (baseline.messages(sample, text) for text in contexts), strict=True)), {
            "route": decision, "plan": plan, "raw_relay": raw_trace, "clean_relay": trace,
            "plain_bge_passages": raw_ids, "clean_bge_passages": clean_ids,
            "evidence": evidence, "recent": recent,
        }


def event_prompts(sample, baseline, original, clean):
    prompts = dict(zip(EVENT_ARMS, (baseline.messages(sample, text) for text in (original, clean)), strict=True))
    # Model, decoding and output cap are identical for the two EventQA arms.
    return prompts, prompts[EVENT_ARMS[0]] == prompts[EVENT_ARMS[1]]


def history_batches(samples):
    """Yield stable groups. Finish each group before warming the next prefix."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["context_id"]].append(sample)
    for history_id, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["id"])
        yield history_id, ordered[:WARMUP], ordered[WARMUP:]


def prefix_id(messages, question):
    """Fingerprint the official prefix before the question."""
    prefix, found, _ = messages[1]["content"].rpartition(question)
    if not question or not found:
        raise ValueError("The official prompt did not contain the question.")
    return sha256((messages[0]["content"] + "\n" + prefix).encode()).hexdigest()
