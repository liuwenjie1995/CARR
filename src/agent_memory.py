"""Compose factual state, conflict resolution and rule writes for a session."""

from dataclasses import asdict, dataclass

from src.memory_state import NUMBERED, Fact, parse_fact, resolve_conflicts
from src.replay_gate import GateDecision, ReplayCase, Rule, evaluate_candidate


@dataclass(frozen=True)
class MemoryConfig:
    dual_layer: bool = True
    conflict_unlearning: bool = True
    validation_gate: bool = True
    conflict_min_distinct: int = 2
    gate_min_gains: int = 1
    gate_max_regressions: int = 0


class Memory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._facts: list[Fact] = []
        self._rules: list[Rule] = []
        self._combined: list[Fact | Rule] = []

    @property
    def facts(self) -> tuple[Fact, ...]:
        if self.config.dual_layer:
            return tuple(self._facts)
        return tuple(item for item in self._combined if isinstance(item, Fact))

    @property
    def rules(self) -> tuple[Rule, ...]:
        if self.config.dual_layer:
            return tuple(self._rules)
        return tuple(item for item in self._combined if isinstance(item, Rule))

    def add_fact(self, fact: Fact) -> None:
        store = self._facts if self.config.dual_layer else self._combined
        store.append(fact)

    def propose_rule(self, rule: Rule, dev_replay: tuple[ReplayCase, ...]) -> GateDecision:
        decision = evaluate_candidate(
            self.rules, rule, dev_replay, enabled=self.config.validation_gate,
            min_gains=self.config.gate_min_gains, max_regressions=self.config.gate_max_regressions,
        )
        if decision.accepted:
            store = self._rules if self.config.dual_layer else self._combined
            store.append(rule)
        return decision


def prepare_factconsolidation(history: str, config: MemoryConfig) -> tuple[str, dict]:
    """Apply state conflict resolution before the unchanged upstream QA template."""
    lines = history.splitlines(keepends=True)
    memory = Memory(config)
    records = {}
    seen_ids = set()
    duplicate_ids = set()
    for position, line in enumerate(lines):
        text = line.rstrip("\r\n")
        if match := NUMBERED.fullmatch(text):
            record_id = int(match["id"])
            if record_id in seen_ids:
                duplicate_ids.add(record_id)
            seen_ids.add(record_id)
        if fact := parse_fact(text):
            memory.add_fact(fact)
            records[position] = fact.record_id
    # Ambiguous source IDs must not create silent omissions.
    decisions = resolve_conflicts(
        memory.facts,
        min_distinct=config.conflict_min_distinct,
    ) if config.conflict_unlearning and not duplicate_ids else ()
    archive = sorted({index for item in decisions for index in item.archived_ids})
    archived_ids = set(archive)
    model_history = "".join(line for pos, line in enumerate(lines) if records.get(pos) not in archived_ids)
    trace = {
        "config": asdict(config), "parsed_facts": len(records), "duplicate_source_ids": sorted(duplicate_ids),
        "conflicts": [asdict(decision) for decision in decisions], "archived_ids": archive,
        "state_records_retained": len(records) - len(archive),
        "rule_admission": {"candidates": 0, "replays": 0, "reason": "dataset_supplies_fact_history_only"},
    }
    return model_history, trace
