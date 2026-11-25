"""Candidate rule admission using paired replay on development fixtures."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    when: tuple[tuple[str, str], ...]
    action: str


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    split: str
    attributes: tuple[tuple[str, str], ...]
    expected_action: str


@dataclass(frozen=True)
class ReplayResult:
    case_id: str
    expected: str
    before: str
    after: str


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    gains: int = 0
    regressions: int = 0
    cases: tuple[ReplayResult, ...] = ()


def decide_action(rules: tuple[Rule, ...], case: ReplayCase) -> str:
    attributes = dict(case.attributes)
    for rule in reversed(rules):
        if all(attributes.get(key) == value for key, value in rule.when):
            return rule.action
    return "standard"


def evaluate_candidate(
    current: tuple[Rule, ...],
    candidate: Rule,
    cases: tuple[ReplayCase, ...],
    *,
    enabled: bool,
    min_gains: int = 1,
    max_regressions: int = 0,
    executor: Callable[[tuple[Rule, ...], ReplayCase], str] = decide_action,
) -> GateDecision:
    if min_gains < 1 or max_regressions < 0:
        raise ValueError("Invalid replay thresholds.")
    if any(case.split != "dev" for case in cases):
        raise ValueError("Replay may only use development fixtures.")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Replay fixture IDs must be unique.")
    if any(rule.rule_id == candidate.rule_id for rule in current):
        return GateDecision(False, "duplicate_rule_id")
    if not enabled:
        return GateDecision(True, "direct_write")
    if not cases:
        return GateDecision(False, "no_dev_replay")

    results = tuple(ReplayResult(
        case.case_id, case.expected_action, executor(current, case), executor((*current, candidate), case),
    ) for case in cases)
    gains = sum(row.before != row.expected and row.after == row.expected for row in results)
    regressions = sum(row.before == row.expected and row.after != row.expected for row in results)
    accepted = gains >= min_gains and regressions <= max_regressions
    return GateDecision(accepted, "passed" if accepted else "rejected", gains, regressions, results)
