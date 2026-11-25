import json

from src.factual_v3 import plan_extraction, prepare
from src.run_v3_dev import bge_indexes, cohort, inputs_for


def test_new_relations_match_locally_without_treating_values_as_exclusive():
    history = ("1. Mira was performed by Alice.\n2. Mira was performed by Bob.\n"
               "3. Mira is affiliated with the religion of Taoism.\n4. Library was founded by Chen.\n")
    plan = plan_extraction(history)
    assert [fact.relation for fact in plan[2].values()] == ["performer", "performer", "religion", "founder"]
    assert not plan[3]
    view, _, trace = prepare(history, plan)
    assert view == history and trace["rule_archive"] == []


def test_entity_gate_counts_distinct_lines_and_does_not_match_name_substrings():
    history = ("1. Mira was relocated to Lyon.\n2. Mira lives in Oslo.\n"
               "3. Ann was introduced as Ann.\n4. Anna was introduced in Paris.\n"
               "5. She was introduced in Rome.\n")
    plan = plan_extraction(history)
    assert [record.index for record in plan[3]] == [0]
    assert plan[4][0]["records_mentioning_entity"] == 2
    assert plan[4][2]["records_mentioning_entity"] == 1
    assert plan[4][4]["decision"] == "unknown_subject"
    assert prepare(history, plan)[1] == history


def test_hybrid_only_archives_with_evidence_for_the_gated_entity():
    history = "1. Mira lives in Oslo.\n2. Mira was relocated to Lyon with Ada.\n"
    layout = plan_extraction(history)
    requests = []

    def extractor(entity):
        def call(model, prompt, cap, phase, **kwargs):
            requests.extend(json.loads(prompt[1]["content"]))
            records = [{"id": 1, "entity": entity, "relation": "residence", "value": "Lyon",
                        "operation": "assert", "evidence": "Mira was relocated to Lyon with Ada."}]
            return {"id": "fixture", "error": None, "finish": "stop", "answer": json.dumps({"records": records})}
        return call

    _, pruned, trace = prepare(history, layout, extractor("Mira"), lambda event: None)
    assert [row["id"] for row in requests] == [1]
    assert trace["model_accepted"] == 1
    assert pruned == "2. Mira was relocated to Lyon with Ada.\n"
    _, kept, trace = prepare(history, layout, extractor("Ada"), lambda event: None)
    assert trace["model_accepted"] == 0 and kept == history


def test_pruned_rag_builds_from_visible_history_not_original_history():
    class Embedding:
        def embed(self, texts, **kwargs):
            return iter([[1.0, 0.0] for _ in texts])

    class Baseline:
        def messages(self, question, text):
            return [{"role": "user", "content": text}]

    history = "1. Mira lives in Oslo.\n2. Mira resides in Lyon.\n"
    _, pruned, _ = prepare(history, plan_extraction(history))
    raw_index, pruned_index = bge_indexes(Embedding(), history, pruned)
    question = {"question": "Where does Mira live?"}
    raw, _ = inputs_for("rag", question, Baseline(), history, pruned, pruned, raw_index, pruned_index)
    filtered, hits = inputs_for("conflict_pruned_rag", question, Baseline(), history, pruned, pruned, raw_index, pruned_index)
    assert "Oslo" in raw[0]["content"]
    assert "Oslo" not in filtered[0]["content"] and "Lyon" in filtered[0]["content"] and hits == [0]


def test_fixed_cohort_has_both_question_types_and_smoke_is_its_prefix():
    rows = [{"id": f"{hop}-{i}", "source": f"factconsolidation_{hop}_6k"} for hop in ("sh", "mh") for i in range(30)]
    result = cohort(rows, 25)
    assert sum("_sh_" in item["source"] for item in result) == 13
    assert sum("_mh_" in item["source"] for item in result) == 12
    assert cohort(rows, 5) == result[:5]
