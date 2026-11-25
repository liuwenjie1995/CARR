"""Run a five-question smoke test for Qwen4-Max and GLM-5.3."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import LongContext, dump_json
from src.smoke_ark_backbones import prepare
from src.v2_transport import usage_of

MODELS = ("qwen4-max", "glm-5.3")
RATES_CNY = {"qwen4-max": (Decimal("10"), Decimal("30")),
             "glm-5.3": (Decimal("8"), Decimal("28"))}
OUTPUT_CAP = 512
BUDGET_GUARD_CNY = Decimal("100")
TIMEOUT_SECONDS = 120


def call(endpoint, key, payload):
    request = Request(endpoint.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": "Bearer " + key,
                               "Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with build_opener().open(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
    except HTTPError as error:
        with error:
            status = error.code
            try:
                body = json.loads(error.read(8192).decode("utf-8", errors="replace"))
            except ValueError:
                body = {"error": "http_error"}
    except (URLError, TimeoutError, OSError, ValueError) as error:
        status, body = None, {"error": type(error).__name__}
    return status, body, round((time.perf_counter() - start) * 1000, 2)


def main():
    endpoint = os.environ.get("BAILIAN_ENDPOINT", "").strip()
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not endpoint or not key:
        raise SystemExit("Set BAILIAN_ENDPOINT and DASHSCOPE_API_KEY in the environment.")

    cases = list(prepare(5))
    jobs = [(model, arm, dev, messages)
            for model in MODELS
            for dev, arms in cases
            for arm, messages in arms.items()]
    reserved = Decimal(0)
    for model, _, _, messages in jobs:
        input_bound = sum(len(message["content"].encode("utf-8")) for message in messages) + 1024
        input_rate, output_rate = RATES_CNY[model]
        reserved += (Decimal(input_bound) * input_rate + OUTPUT_CAP * output_rate) / 1_000_000
    if reserved > BUDGET_GUARD_CNY:
        raise SystemExit(f"Smoke-test reservation {reserved:.2f} CNY exceeds the 100 CNY guard.")

    output = ROOT / "results/dev" / ("bailian-backbones-smoke-" +
             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    rows = []
    for model, arm, dev, messages in jobs:
        payload = {"model": model, "messages": messages, "temperature": 0,
                   "max_tokens": OUTPUT_CAP}
        status, body, elapsed = call(endpoint, key, payload)
        usage = usage_of(body)
        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        answer = (choice.get("message") or {}).get("content")
        answer = answer if isinstance(answer, str) else ""
        correct = scorer.score(answer, dev["reference"]) if status == 200 and answer.strip() else None
        row = {"sample_id": dev["sample_id"], "type": dev["type"], "model": model,
               "arm": arm, "http_status": status, "answer": answer, "correct": correct,
               "usage": usage, "elapsed_ms": elapsed,
               "request": payload, "response": body,
               "served_model": body.get("model") if isinstance(body, dict) else None,
               "finish_reason": choice.get("finish_reason")}
        dump_json(output / f"{model}--{arm}--{dev['sample_id']}.json", row)
        rows.append(row)

    totals = []
    actual_cny = Decimal(0)
    for model in MODELS:
        input_rate, output_rate = RATES_CNY[model]
        model_rows = [row for row in rows if row["model"] == model]
        for row in model_rows:
            if row["usage"]:
                actual_cny += (Decimal(row["usage"]["prompt_tokens"]) * input_rate +
                               Decimal(row["usage"]["completion_tokens"]) * output_rate) / 1_000_000
        for arm in ("official_full_context", "unified_a"):
            group = [row for row in model_rows if row["arm"] == arm]
            totals.append({"model": model, "arm": arm, "questions": len(group),
                           "correct": sum(row["correct"] is True for row in group),
                           "failed_requests": sum(row["http_status"] != 200 for row in group),
                           "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in group if row["usage"]),
                           "completion_tokens": sum(row["usage"]["completion_tokens"] for row in group if row["usage"]),
                           "cache_hit_tokens": sum(row["usage"]["cache_hit_tokens"] for row in group if row["usage"])})
    summary = {"status": "complete" if len(rows) == 20 and all(row["http_status"] == 200 for row in rows) else "partial",
               "split": "dev", "limit": 5, "calls": len(rows),
               "reserved_budget_cny": str(reserved), "estimated_actual_cost_cny": str(actual_cny),
               "totals": totals, "generator": "src/smoke_bailian_backbones.py"}
    dump_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
