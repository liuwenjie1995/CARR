"""Quote the frozen FactConsolidation answering-backbone replay without API calls."""

import argparse
import csv
from decimal import ROUND_CEILING, Decimal
from math import ceil

from src.evaluator import ROOT, dump_json
from src.smoke_ark_backbones import DISCOUNTED, MODELS, native_payload, prepare
from src.v2_report import load

FORMAL = ROOT / "results/test/formal-v7-20261123T134503331644Z"
ARK_DEV = ROOT / "results/dev/ark-backbones-v7-20261124T005359Z/summary.json"
V7_DEV = ROOT / "results/dev/unified-prompt-v7-20261123T133145804847Z-paid/summary.json"
V3_DEV = ROOT / "results/dev/v3-20261123T111806Z-586a10/summary.json"
DEST = ROOT / "results/test_plan_v7/ark_backbone_quote.json"
ARMS = ("official_full_context", "unified_a")
OUTPUT_CAP = 512


def compute(limit):
    ark = load(ARK_DEV)
    if ark["status"] != "complete" or load(FORMAL / "manifest.json")["status"] != "complete":
        raise ValueError("The quote needs complete development and archived formal logs.")
    dev = {(row["model"], row["arm"]): row for row in ark["totals"] if row["type"] == "all"}
    reference_dev = (
        next(row["answer_prompt_tokens"] for row in load(V3_DEV)["arms"] if row["arm"] == "full"),
        next(row["prompt_tokens"] for row in load(V7_DEV)["arms"]
             if row["arm"] == "official_evidence" and row["group"] == "all"),
    )
    with (FORMAL / "billing.csv").open(encoding="utf-8-sig", newline="") as source:
        billed = {row["arm"]: row for row in csv.DictReader(source)}
    for arm in ARMS:
        if int(billed[arm]["paid_requests"]) != 562:
            raise ValueError(f"Unexpected frozen case count for {arm}.")
    if limit == 5:
        cases = list(prepare(limit))
        for sample, arms in cases:
            for model in DISCOUNTED:
                for arm, messages in arms.items():
                    old = next(row for row in ark["results"] if row["sample_id"] == sample["sample_id"]
                               and row["model_requested"] == model and row["arm"] == arm)
                    if load(ROOT / old["source"])["request"] != native_payload(model, messages):
                        raise ValueError(f"Development replay payload drift for {sample['sample_id']}.")
        print(f"Offline smoke passed: {limit} development questions, {limit * 4} saved payloads verified, 0 API calls.")
        return

    rows = []
    for model, rate in DISCOUNTED.items():
        for arm, old_prompt in zip(ARMS, reference_dev, strict=True):
            d = dev[model, arm]
            if d["recorded"] != 25 or not d["prompt_tokens"]:
                raise ValueError("Missing model tokenizer calibration.")
            prompt = ceil(Decimal(billed[arm]["prompt_tokens"]) * d["prompt_tokens"] / old_prompt)
            expected_output = ceil(Decimal(d["completion_tokens"]) * limit / d["recorded"])
            output_cap = limit * OUTPUT_CAP
            rows.append({"model": model, "arm": arm, "calls": limit, "input_estimate": prompt,
                         "output_estimate": expected_output, "output_cap": output_cap,
                         "afp_no_cache_estimate": str((prompt + expected_output) * rate / 10000),
                         "afp_no_cache_output_cap": str((prompt + output_cap) * rate / 10000),
                         "afp_no_cache_without_promotion": str((prompt + output_cap) * MODELS[model] / 10000)})
    totals = {key: str(sum(Decimal(row[key]) for row in rows)) for key in rows[0] if key.startswith("afp_")}
    guard = str((Decimal(totals["afp_no_cache_output_cap"]) * Decimal("1.10"))
                .to_integral_value(rounding=ROUND_CEILING))
    result = {"status": "quote_pending_confirmation", "api_calls_spent": 0,
              "cohort": "frozen FactConsolidation test", "questions": limit,
              "planned_answer_requests": sum(row["calls"] for row in rows), "new_planner_calls": 0,
              "settings": {model: {k: v for k, v in native_payload(model, []).items() if k != "messages"}
                           for model in DISCOUNTED},
              "totals": totals, "recommended_afp_guard": guard, "arms": rows,
              "assumptions": ["Input lengths estimated from frozen formal usage and each arm's development token ratio.",
                              "The quoted guard adds 10% to the no-cache estimate at the output cap.",
                              "No cache billing discount is assumed. Provider charges are authoritative.",
                              "Reuse the exact frozen evidence and answering messages. Only the answering model changes."],
              "sources": [str(path.relative_to(ROOT).as_posix()) for path in
                          (FORMAL / "billing.csv", ARK_DEV, V3_DEV, V7_DEV)]}
    dump_json(DEST, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", required=True, type=int, choices=(5, 562))
    compute(parser.parse_args().limit)
