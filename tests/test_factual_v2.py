import json

from src.factual_v2 import projection, records_for, rule_observation, validate_model
from src.v2_extract import extract_history
from src.v2_retrieval import matched_recent, proxy_tokens


def via_rules(history, protocol):
    records, duplicate = records_for(history, protocol)
    return projection(history, [fact for r in records.values() if (fact := rule_observation(r))],
                      protocol, duplicate_ids=duplicate)


def item(record, **changes):
    return {"id": record.index, "entity": "Mira", "relation": "residence", "value": "Lyon",
            "operation": "replace", "evidence": record.text} | changes


def test_alias_and_effective_order_with_evidence_offsets():
    history = "2026-02-01: Mira now resides in Lyon.\r\n2026-01-01: Mira lives in Oslo.\r\n"
    visible, decisions = via_rules(history, "dated")
    assert visible == history.splitlines(keepends=True)[0]
    move, = decisions
    assert move["removed_line"] == 1 and move["winner_line"] == 0
    start, end = move["old_span"]
    assert history[start:end] == "Mira lives in Oslo."


def test_no_time_multivalue_and_ambiguous_same_date_are_preserved():
    undated = "Mira lives in Oslo.\nMira now resides in Lyon.\n"
    assert via_rules(undated, "dated") == (undated, [])
    same_day = "2026-02-01: Mira now lives in Lyon.\n2026-02-01: Mira now resides in Bern.\n"
    assert via_rules(same_day, "dated") == (same_day, [])
    multi = "1. Mira is a citizen of France.\n2. Mira is a citizen of Switzerland.\n"
    assert via_rules(multi, "numbered") == (multi, [])
    no_signal = "2026-01-01: Mira lives in Lyon.\n2026-02-01: Mira resides in Bern.\n"
    assert via_rules(no_signal, "dated") == (no_signal, [])


def test_validates_original_quote_and_operator_target():
    records, _ = records_for("2026-02-01: Mira updated to Lyon from Oslo.\n", "dated")
    record = records[0]
    assert validate_model(item(record), record).relation == "residence"
    assert validate_model(item(record, value="Oslo"), record) is None
    assert validate_model(item(record, evidence="Mira relocated to Lyon."), record) is None
    assert validate_model(item(record, id=True), record) is None


def test_hybrid_fallback_and_reuse_cache_is_history_level():
    history = "2026-01-01: Mira lives in Oslo.\n2026-02-01: Mira updated to Lyon from Oslo.\n"
    records, _ = records_for(history, "dated")
    calls = []

    def fake_call(model, prompt, _, phase, **kwargs):
        calls.extend(json.loads(prompt[1]["content"]))
        return {"error": None, "finish": "stop", "id": "test", "answer": json.dumps({"facts": [item(records[1])]})}

    logs = []
    visible, trace = extract_history(history, "dated", "hybrid", fake_call, logs.append)
    assert [request["id"] for request in calls] == [1]
    assert visible == history.splitlines(keepends=True)[1]
    assert trace["rule_extracted"] == trace["model_extracted"] == len(logs) == 1


def test_failed_extractor_and_duplicate_ids_keep_original_context():
    history = "1. A known statement without a rule pattern.\n"
    result, trace = extract_history(history, "numbered", "model",
                                    lambda *_a, **_kw: {"error": "timeout", "id": "test"}, lambda _row: None)
    assert result == history and trace["unresolved_lines"] == [0]
    duplicate = "1. Mira lives in Oslo.\n1. Mira lives in Lyon.\n2. Mira now resides in Bern.\n"
    assert via_rules(duplicate, "numbered") == (duplicate, [])


def test_matched_recent_is_a_suffix_with_measured_token_budget():
    raw = "1. Mira lives in a very long sentence in Oslo.\n2. Mira now resides in Lyon.\n"
    filtered = raw.splitlines(keepends=True)[1]
    retained = matched_recent(raw, filtered, lambda value: value)
    assert raw.endswith(retained) and retained
    assert proxy_tokens(retained) <= proxy_tokens(filtered)
