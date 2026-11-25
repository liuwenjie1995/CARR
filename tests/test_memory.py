from dataclasses import replace

import pytest

from src.agent_memory import Memory, MemoryConfig, prepare_factconsolidation
from src.memory_state import Fact, parse_fact, resolve_conflicts
from src.replay_gate import ReplayCase, Rule


def test_conflict_uses_effective_order_and_keeps_historical_and_unrelated_records():
    old = Fact(9, "Mira", "City", "Oslo", 1)
    current = Fact(4, "mira", "city", "Lyon", 3)
    tentative = Fact(12, "Mira", "City", "Rome", 5, confirmed=False)
    unrelated = Fact(8, "Mira", "Drink", "Tea", 5)
    facts = current, tentative, unrelated, old
    assert resolve_conflicts(facts, as_of=2) == ()
    resolved, = resolve_conflicts(facts)
    assert resolved.kept_id == current.record_id
    assert resolved.archived_ids == (old.record_id,)


def test_ambiguous_latest_day_and_high_threshold_preserve_records():
    old = Fact(1, "Mira", "city", "Oslo", 1)
    current = Fact(2, "Mira", "city", "Lyon", 3)
    tie = Fact(3, "Mira", "city", "Paris", 3)
    assert resolve_conflicts((old, current, tie)) == ()
    assert resolve_conflicts((old, current), min_distinct=3) == ()
    assert resolve_conflicts((old, replace(current, value=" oslo "))) == ()


def test_parser_preserves_unsupported_data_and_conflict_switch_is_independent():
    history = "Facts:\n1. Lee is a citizen of Canada.\n2. Lee writes about Canada.\n3. Lee is a citizen of France.\n"
    assert parse_fact("2. Lee writes about Canada.") is None
    config = MemoryConfig()
    filtered, trace = prepare_factconsolidation(history, config)
    assert filtered == "Facts:\n2. Lee writes about Canada.\n3. Lee is a citizen of France.\n"
    assert trace["archived_ids"] == [1]
    assert prepare_factconsolidation(history, replace(config, dual_layer=False))[0] == filtered
    assert prepare_factconsolidation(history, replace(config, conflict_unlearning=False))[0] == history
    ambiguous = history + "1. Another unparsed fact.\n"
    assert prepare_factconsolidation(ambiguous, config)[0] == ambiguous


def dev(case_id, expected, **attributes):
    return ReplayCase(case_id, "dev", tuple(attributes.items()), expected)


@pytest.mark.parametrize("dual_layer", [True, False])
def test_admission_rejects_regression_atomically_in_both_storage_layouts(dual_layer):
    memory = Memory(MemoryConfig(dual_layer=dual_layer))
    fact = Fact(1, "parcel", "owner", "Mira", 1)
    memory.add_fact(fact)
    protects = Rule("fragile", (("fragile", "yes"),), "padded")
    assert memory.propose_rule(protects, (dev("fragile", "padded", fragile="yes"),)).accepted
    rush = Rule("rush", (("rush", "yes"),), "express")
    replay = (dev("ordinary", "express", rush="yes"), dev("both", "padded", fragile="yes", rush="yes"))
    report = memory.propose_rule(rush, replay)
    assert (report.accepted, report.gains, report.regressions) == (False, 1, 1)
    assert report.cases[1].before == "padded" and report.cases[1].after == "express"
    assert memory.rules == (protects,) and memory.facts == (fact,)


def test_gate_controls_threshold_and_blocks_holdout_replay():
    rule = Rule("special", (), "special")
    memory = Memory(MemoryConfig(gate_min_gains=2))
    assert memory.propose_rule(rule, (dev("one", "special"),)).reason == "rejected"
    assert memory.propose_rule(rule, ()).reason == "no_dev_replay"
    with pytest.raises(ValueError, match="development"):
        memory.propose_rule(rule, (replace(dev("holdout", "special"), split="test"),))
    assert memory.rules == ()
    assert memory.propose_rule(rule, (dev("one", "special"), dev("two", "special"))).accepted


def test_disabled_gate_writes_without_replay_and_protects_duplicate_ids():
    memory = Memory(MemoryConfig(validation_gate=False))
    original = Rule("r", (), "stored")
    assert memory.propose_rule(original, ()).reason == "direct_write"
    assert not memory.propose_rule(replace(original, action="overwrite"), ()).accepted
    assert memory.rules == (original,)
