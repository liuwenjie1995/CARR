"""Run and log an authorized five-case development smoke test."""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.long_context import CHUNK_SIZE, MAX_TOKENS, UPSTREAM, UPSTREAM_COMMIT, LongContext

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
PRICING_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"
RATES = {
    period: {key: Decimal(value) for key, value in zip(
        ("cache_hit", "cache_miss", "completion"), values, strict=True
    )}
    for period, values in {"off_peak": ("0.003", "0.15", "0.6"), "peak": ("0.006", "0.30", "1.20")}.items()
}
TIMEOUT_SECONDS = 40
MAX_REQUESTS = 5
MAX_RETRIES = 0
MAX_BUDGET_USD = Decimal("0.05")
SEED = 42
MESSAGE_TOKEN_MARGIN = 1024


def utc_now():
    return datetime.now(timezone.utc)


def dump_json(path: Path, data: object):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_dev():
    data = ROOT / "data"
    lock_bytes = (data / "split-lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    if not lock["test_locked"] or any(lock["cross_split_intersections"].values()):
        raise ValueError("Missing valid locked split.")

    def load_checked(name):
        path = f"dev/{name}"
        contents = (data / path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != lock["files"][path]:
            raise ValueError(f"Frozen development file changed: {path}")
        return contents

    samples = [json.loads(line) for line in load_checked("samples.jsonl").splitlines()]
    contexts = json.loads(load_checked("contexts.json"))
    return samples, contexts, hashlib.sha256(lock_bytes).hexdigest()


def load_key(path: Path | None):
    if key := os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return key
    if path is None:
        raise ValueError("Supply DEEPSEEK_API_KEY or --key-file.")
    keys = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if re.search(r"deepseek|\bds\b|深度求索", line, re.IGNORECASE):
            keys.extend(re.findall(r"\bsk-[a-z0-9_-]{12,}\b", line, re.IGNORECASE))
    if len(keys) != 1:
        raise ValueError("Expected one labeled DeepSeek credential.")
    return keys[0]


def token_cost(hit, miss, completion, period):
    rate = RATES[period]
    return (hit * rate["cache_hit"] + miss * rate["cache_miss"] + completion * rate["completion"]) / 1_000_000


def estimated_reservation(messages):
    # Reserve at the peak uncached rate with a conservative text byte bound.
    input_ceiling = sum(len(item["content"].encode("utf-8")) for item in messages) + MESSAGE_TOKEN_MARGIN
    return token_cost(0, input_ceiling, MAX_TOKENS, "peak"), input_ceiling


def usage_counts(response):
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt, completion = (usage.get(name) for name in ("prompt_tokens", "completion_tokens"))
    if not all(type(value) is int and value >= 0 for value in (prompt, completion)):
        return None
    hit = usage.get("prompt_cache_hit_tokens")
    if type(hit) is not int or not 0 <= hit <= prompt:
        # Unknown cache split is costed at the uncached rate.
        return prompt, completion, 0, False
    return prompt, completion, hit, True


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_api(payload: dict, key: str):
    request = Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with build_opener(NoRedirect).open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8").replace(key, "[REDACTED]")
            return response.status, json.loads(body), None
    except HTTPError as error:
        with error:
            body = error.read(8192).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        return error.code, {"error_body": body}, f"http_{error.code}"
    except (URLError, TimeoutError, OSError, ValueError) as error:
        return None, None, type(error).__name__


def summarize(metadata, rows):
    finished = [row for row in rows if row["status"] == "ok"]
    accounted = [row["usage"] for row in rows if row["usage"] is not None]
    prompt = sum(usage["prompt_tokens"] for usage in accounted)
    completion = sum(usage["completion_tokens"] for usage in accounted)
    hit = sum(usage["cache_hit_tokens"] for usage in accounted)
    successes = sum(row["correct"] for row in finished)
    costs = {period: str(token_cost(hit, prompt - hit, completion, period)) for period in RATES}
    unknown_cost_usd = sum(Decimal(row["reserved_usd"]) for row in rows if row["usage"] is None)
    all_dev_count = metadata["dev_size"]
    full_dev_projection = None
    if len(finished) == len(rows) == MAX_REQUESTS and len(accounted) == len(rows):
        # All dev items share one history. A no-cache rate brackets cache variability.
        full_dev_projection = {
            "scope": "selected_dev_source_one_baseline_one_pass",
            "cases": all_dev_count, "estimated_calls": all_dev_count,
            "prompt_tokens": round(prompt / len(rows) * all_dev_count),
            "completion_tokens": round(completion / len(rows) * all_dev_count),
            "estimated_usd_no_cache": {
                period: str(token_cost(0, prompt, completion, period) * all_dev_count / len(rows))
                for period in RATES
            },
        }
    return {
        **metadata, "planned_samples": MAX_REQUESTS, "attempted_samples": len(rows),
        "successes": successes, "completed_samples": len(finished), "errors": len(rows) - len(finished),
        "success_rate": successes / len(rows) if rows else None,
        "accuracy_completed": successes / len(finished) if finished else None,
        "api_calls": len(rows), "prompt_tokens": prompt, "completion_tokens": completion,
        "cache_hit_tokens": hit, "cache_miss_tokens_or_unclassified": prompt - hit,
        "calls_missing_usage": len(rows) - len(accounted),
        "mean_end_to_end_ms": mean(row["end_to_end_ms"] for row in rows) if rows else None,
        "estimated_usd_accounted": costs, "reserved_usd_for_unreported_usage": str(unknown_cost_usd),
        "projection": full_dev_projection,
    }


def run_sample(baseline, sample, messages, key, reserved, input_ceiling, start):
    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
        "thinking": {"type": "disabled"}, "messages": messages,
    }
    started_at = utc_now().isoformat()
    request_start = time.perf_counter()
    status, response, error = call_api(payload, key)
    request_ms = round((time.perf_counter() - request_start) * 1000, 3)
    usage = usage_counts(response)
    answer = None
    if error is None:
        try:
            answer = response["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            error = "invalid_completion"
    correct = error is None and baseline.score(answer, sample["answers"])
    return {
        "sample_id": sample["id"], "source": sample["source"], "context_id": sample["context_id"],
        "question": sample["question"], "reference_answers": sample["answers"],
        "prediction": answer, "correct": correct, "status": "ok" if error is None else error,
        "started_at_utc": started_at, "end_to_end_ms": round((time.perf_counter() - start) * 1000, 3),
        "reserved_input_token_ceiling": input_ceiling, "reserved_usd": str(reserved),
        "usage": None if usage is None else dict(zip(
            ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_split_reported"), usage, strict=True
        )),
        "trajectory": [{"operation": "chat_completion", "request": payload, "http_status": status,
                        "response": response, "elapsed_ms": request_ms}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev"], required=True)
    parser.add_argument("--limit", type=int, choices=[5], required=True)
    parser.add_argument("--baseline", choices=[LongContext.name], default=LongContext.name)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--preflight", action="store_true", help="Build prompts and check cost ceiling without API calls.")
    args = parser.parse_args()
    git_result = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    if git_result.stdout.strip() != UPSTREAM_COMMIT:
        raise SystemExit("Upstream checkout does not match the pinned revision.")
    tracked = subprocess.run(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    )
    if tracked.stdout.strip():
        raise SystemExit("Pinned upstream checkout has modified tracked files.")

    samples, contexts, lock_hash = load_dev()
    eligible = [sample for sample in samples if sample["source"] == "factconsolidation_sh_6k"]
    picked = random.Random(SEED).sample(eligible, args.limit)
    baseline = LongContext(ROOT / ".venv" / "nltk_data")
    metadata = {
        "baseline": args.baseline, "split": args.split, "provider": "deepseek", "model_requested": MODEL,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_templates_sha256": baseline.templates_sha256,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((UPSTREAM.parent / "long_context.py").read_bytes()).hexdigest(),
        "split_lock_sha256": lock_hash, "source_filter": "factconsolidation_sh_6k",
        "seed": SEED, "provider_seed": None, "temperature": 0, "max_tokens": MAX_TOKENS,
        "retrieval_top_k": None, "tools": [], "chunk_size": CHUNK_SIZE, "metric": "substring_exact_match",
        "timeout_seconds": TIMEOUT_SECONDS, "max_retries": MAX_RETRIES, "max_requests": MAX_REQUESTS,
        "budget_cap_usd": str(MAX_BUDGET_USD), "dev_size": len(eligible), "pricing_source": PRICING_SOURCE,
        "pricing_usd_per_million": {period: {kind: str(rate) for kind, rate in rates.items()} for period, rates in RATES.items()},
        "python": sys.version.split()[0], "sample_ids": [sample["id"] for sample in picked],
    }
    if args.preflight:
        reserved = sum(estimated_reservation(baseline.messages(sample, contexts[sample["context_id"]]))[0] for sample in picked)
        if reserved > MAX_BUDGET_USD:
            raise SystemExit("Smoke plan exceeds the configured USD budget ceiling.")
        print(json.dumps({**metadata, "peak_uncached_reserved_usd": str(reserved), "api_calls": 0}, indent=2))
        return

    key = load_key(args.key_file)
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = ROOT / "results" / "raw" / run_id
    folder.mkdir(parents=True)
    metadata["run_id"] = run_id
    metadata["created_at_utc"] = utc_now().isoformat()
    rows = []
    reserved_total = Decimal(0)
    for sample in picked:
        start = time.perf_counter()
        messages = baseline.messages(sample, contexts[sample["context_id"]])
        reservation, input_ceiling = estimated_reservation(messages)
        if len(rows) >= MAX_REQUESTS or reserved_total + reservation > MAX_BUDGET_USD:
            metadata["stopped_reason"] = "budget_guard"
            break
        reserved_total += reservation
        row = run_sample(baseline, sample, messages, key, reservation, input_ceiling, start)
        rows.append(row)
        dump_json(folder / f"{len(rows):02d}-trajectory.json", row)
        if row["status"] in {"http_400", "http_401", "http_402", "http_403", "http_404"}:
            metadata["stopped_reason"] = row["status"]
            break
    metadata["reserved_usd"] = str(reserved_total)
    summary = summarize(metadata, rows)
    dump_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if len(rows) != args.limit or summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
