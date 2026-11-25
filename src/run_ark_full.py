"""Replay locked formal answer requests through Kimi and GLM, with on-disk checkpoints."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from getpass import getpass
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from threading import Lock
import time

from src.evaluator import LongContext, ROOT, dump_json, utc_now
from src.quote_ark_full import ARMS, DEST, FORMAL, compute
from src.smoke_ark_backbones import DISCOUNTED, native_payload, request, reservation
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_transport import usage_of

TIMEOUT = 120
WORKERS_PER_MODEL = 3
MAX_RETRIES = 2
CONTENT_BLOCKS = {"SensitiveContentDetected", "InputSensitiveContentDetected", "OutputSensitiveContentDetected"}


def content_block(record):
    error = record["response"].get("error", {})
    code = str(error.get("code", "")) if isinstance(error, dict) else ""
    return code.split(".")[0] in CONTENT_BLOCKS


def held_after_interrupt(out):
    path = out / "interrupt_allowance.json"
    return Decimal(load(path)["afp_held"]) if path.exists() else Decimal(0)


def summarize(out, cases, stops):
    rows = [load(f) for f in (out / "results").glob("*.json")]
    totals = []
    for model in DISCOUNTED:
        for arm in ARMS:
            selected = [r for r in rows if r["model"] == model and r["arm"] == arm]
            for tag in ("all", "sh", "mh", "32k", "64k", "262k"):
                group = [r for r in selected if tag == "all" or tag in r["sample_id"].split("_")]
                expected = sum(tag == "all" or tag in sample["id"].split("_") for sample in cases)
                totals.append({"model": model, "arm": arm, "stratum": tag, "expected": expected,
                               "recorded": len(group), "correct": sum(r["correct"] for r in group),
                               "empty_or_error": sum(not r["answered"] for r in group),
                               "length_limited": sum(r["finish"] == "length" for r in group),
                               **{key: sum(r["usage"][key] for r in group if r["usage"])
                                  for key in ("prompt_tokens", "completion_tokens", "cache_hit_tokens")},
                               "mean_call_ms": round(mean(r["elapsed_ms"] for r in group), 2) if group else None})
    attempts = [load(f) for f in (out / "attempts").glob("*.json")]
    spent = sum(Decimal(r["afp_for_guard"]) for r in attempts)
    paired = {}
    for model in DISCOUNTED:
        full, a = ({r["sample_id"]: r["correct"] for r in rows if r["model"] == model and r["arm"] == arm}
                   for arm in ARMS)
        ids = full.keys() & a.keys()
        paired[model] = {"paired": len(ids), "a_only_correct": sum(a[i] and not full[i] for i in ids),
                         "full_only_correct": sum(full[i] and not a[i] for i in ids)}
    result = {"status": "complete" if len(rows) == len(cases) * len(ARMS) * len(DISCOUNTED) else "partial",
              "cases": len(cases), "recorded": len(rows), "api_attempts": len(attempts),
              "afp_no_cache_accounted_not_invoice": str(spent),
              "interrupt_allowance_afp_not_invoice": str(held_after_interrupt(out)), "stops": stops,
              "totals": totals, "paired": paired, "generator": "src/run_ark_full.py"}
    dump_json(out / "summary.json", result)
    write_csv(out / "summary.csv", totals)
    return result


def run(out, cap, selected_models=None):
    target_models = tuple(selected_models) if selected_models else tuple(DISCOUNTED)
    for model in target_models:
        if model not in DISCOUNTED:
            raise ValueError(f"Unknown model: {model}")
    quote = load(DEST)
    if cap <= 0 or cap > Decimal(quote["recommended_afp_guard"]):
        raise ValueError("Budget must be positive and at most the confirmed quote guard.")
    for file, digest in load(ROOT / "results/test_plan_v7/freeze.json")["source_sha256"].items():
        if sha256((ROOT / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {file}")
    cases, _ = read_locked("factcon", "test")
    if len(cases) != quote["questions"]:
        raise ValueError("Unexpected locked test size.")
    key = os.environ.get("ARK_AGENT_PLAN_API_KEY") or getpass("Agent Plan API key (hidden): ")
    if not key.strip():
        raise SystemExit("Missing Agent Plan API key.")
    for sub in ("results", "attempts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    config = {"budget_afp": str(cap), "quote_digest": sha256(DEST.read_bytes()).hexdigest(), "scope": "factcon test"}
    if (out / "config.json").exists() and load(out / "config.json") != config:
        raise ValueError("Existing run uses a different budget or quote.")
    dump_json(out / "config.json", config)
    print(out, flush=True)
    groups = defaultdict(list)
    for sample in cases:
        groups[sample["context_id"]].append(sample)
    scorer = LongContext(ROOT / ".venv/nltk_data")
    lock = Lock()
    stopped = {}
    global_stop = False
    spent = sum(Decimal(load(f)["afp_for_guard"]) for f in (out / "attempts").glob("*.json")) + held_after_interrupt(out)
    held = Decimal(0)

    def complete(result_path, attempt_file, record, sample):
        body = record["response"]
        choice = (body.get("choices") or [{}])[0]
        answer = (choice.get("message") or {}).get("content")
        answered = record["http_status"] == 200 and isinstance(answer, str) and bool(answer.strip())
        dump_json(result_path, {**{k: v for k, v in record.items() if k != "response"},
                                "attempt_log": attempt_file.name, "answer": answer, "answered": answered,
                                "provider_blocked": content_block(record),
                                "correct": bool(answered and scorer.score(answer, sample["answers"])),
                                "finish": choice.get("finish_reason"), "served_model": body.get("model")})

    def call(model, arm, sample):
        nonlocal held, spent, global_stop
        stem = f"{model}--{arm}--{sample['id']}"
        result_path = out / "results" / f"{stem}.json"
        if result_path.exists():
            return True
        previous = sorted((out / "attempts").glob(f"{stem}--*.json"))
        if previous:
            record = load(previous[-1])
            if record["http_status"] == 200 or content_block(record):
                complete(result_path, previous[-1], record, sample)
                return True
        original_case = load(FORMAL / "cases" / f"{arm}--{sample['id']}.json")
        if original_case["reference"] != sample["answers"] or original_case["question"] != sample["question"]:
            raise ValueError(f"Saved request membership differs for {sample['id']}.")
        source = FORMAL / "requests" / (original_case["request_id"] + ".json")
        original = load(source)
        payload = native_payload(model, original["request"]["messages"])
        coeff = DISCOUNTED[model]
        maximum = reservation(payload["messages"], coeff, payload["max_tokens"])
        start_number = len(previous)
        for attempt in range(MAX_RETRIES + 1):
            with lock:
                if global_stop or model in stopped:
                    return False
                if spent + held + maximum > cap:
                    stopped[model] = "AFP guard"
                    global_stop = True
                    return False
                held += maximum
            status, body, elapsed = request(payload, key, timeout=TIMEOUT)
            usage = usage_of(body)
            error = body.get("error", {})
            code = str(error.get("code", "")) if isinstance(error, dict) else ""
            transient = status == 429 and code in {"ModelAccountConcurrentRequestsLimitExceeded", "AccountRateLimitExceeded"}
            quota = status in (402, 429) and not transient
            rejected = status is not None and 400 <= status < 500
            amount = ((usage["prompt_tokens"] + usage["completion_tokens"]) * coeff / 10000
                      if usage is not None else Decimal(0) if rejected else maximum)
            attempt_file = out / "attempts" / f"{stem}--{start_number + attempt + 1:04d}.json"
            record = {"model": model, "arm": arm, "sample_id": sample["id"], "http_status": status,
                      "request_source": source.relative_to(ROOT).as_posix(),
                      "request_source_sha256": sha256(source.read_bytes()).hexdigest(),
                      "request_settings": {k: v for k, v in payload.items() if k != "messages"},
                      "response": body, "usage": usage, "elapsed_ms": elapsed, "afp_for_guard": str(amount),
                      "timestamp_utc": utc_now().isoformat()}
            dump_json(attempt_file, record)
            with lock:
                spent += amount
                held -= maximum
                if quota or status in (401, 403):
                    stopped[model] = {"http": status, "code": code, "log": attempt_file.name}
                    global_stop = True
                    return False
                if status in (400, 404) and not content_block(record):
                    stopped[model] = {"http": status, "code": code, "log": attempt_file.name}
                    return False
            if status == 200 or content_block(record):
                complete(result_path, attempt_file, record, sample)
                return True
            if transient and attempt < MAX_RETRIES:
                time.sleep(4 * (attempt + 1))
                continue
            with lock:
                stopped[model] = {"http": status, "code": code, "log": attempt_file.name}
            return False

    def model_run(model):
        with ThreadPoolExecutor(max_workers=WORKERS_PER_MODEL) as pool:
            for arm in reversed(ARMS):
                for context in sorted(groups):
                    ordered = sorted(groups[context], key=lambda row: row["id"])
                    lead = 2 if arm == ARMS[0] else 0
                    for sample in ordered[:lead]:
                        if not call(model, arm, sample):
                            return
                    for i in range(lead, len(ordered), WORKERS_PER_MODEL):
                        tasks = [pool.submit(call, model, arm, s) for s in ordered[i:i + WORKERS_PER_MODEL]]
                        if not all([task.result() for task in tasks]):
                            return

    try:
        with ThreadPoolExecutor(max_workers=len(target_models)) as pool:
            jobs = [pool.submit(model_run, model) for model in target_models]
            for job in jobs:
                job.result()
    finally:
        summary = summarize(out, cases, stopped)
        print({k: v for k, v in summary.items() if k not in {"totals", "paired"}}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(5, 562), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--budget", type=Decimal)
    parser.add_argument("--models", nargs="+", choices=tuple(DISCOUNTED))
    args = parser.parse_args()
    if args.limit == 5 and args.run_dir is None and args.budget is None:
        compute(5)
    elif args.limit == 562 and args.run_dir and args.budget:
        path = args.run_dir.resolve()
        if path.parent != (ROOT / "results/test").resolve() or not path.name.startswith("ark-backbones-"):
            parser.error("Expected a dedicated ark-backbones-* run directory in results/test.")
        run(path, args.budget, args.models)
    else:
        parser.error("Use --limit 5 offline or --limit 562 with the confirmed budget and run folder.")
