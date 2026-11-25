"""Execute the frozen test schedule only after the user authorizes the quoted budget."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import evaluator
from evaluator import ROOT, UPSTREAM, LongContext, dump_json, load_dev, load_key, usage_counts, utc_now
from src.agent_memory import MemoryConfig, prepare_factconsolidation

PLAN = ROOT / "results" / "test_plan"
MAX_PROVIDER_INPUT_TOKENS = 1_048_576
FATAL = {"http_400", "http_401", "http_402", "http_403", "http_404"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_freeze(freeze, estimate):
    if estimate["freeze_sha256"] != file_sha(PLAN / "freeze.json"):
        raise SystemExit("Budget estimate and freeze do not match.")
    for path, expected in freeze["files_sha256"].items():
        if file_sha(ROOT / path) != expected:
            raise SystemExit(f"Frozen file changed: {path}")
    for command, expected in [(["rev-parse", "HEAD"], freeze["upstream_commit"]), (["status", "--porcelain", "-uno"], "")]:
        actual = subprocess.run(["git", "-C", str(UPSTREAM), *command], text=True,
                                capture_output=True, check=True).stdout.strip()
        if actual != expected:
            raise SystemExit("Frozen upstream checkout does not match.")


def effective_config(freeze, variant):
    return MemoryConfig(**(freeze["full_method"] | freeze["variants"][variant]))


def prepare_histories(contexts, freeze, variant):
    if variant == "baseline":
        return contexts, {}
    prepared = {key: prepare_factconsolidation(history, effective_config(freeze, variant))
                for key, history in contexts.items()}
    return {key: value[0] for key, value in prepared.items()}, {key: value[1] for key, value in prepared.items()}


def ordered(samples, seed):
    samples = sorted(samples, key=lambda row: row["id"])
    random.Random(seed).shuffle(samples)
    return sorted(samples, key=lambda row: row["context_id"])


def payload(baseline, model, settings, sample, history):
    return {
        "model": model, "temperature": settings["temperature"], "max_tokens": settings["max_tokens"],
        "thinking": {"type": settings["thinking"]}, "messages": baseline.messages(sample, history),
    }


def run_case(baseline, model, settings, sample, history, key):
    start = time.perf_counter()
    request = payload(baseline, model, settings, sample, history)
    sent_at = utc_now().isoformat()
    call_start = time.perf_counter()
    status, response, error = evaluator.call_api(request, key)
    request_ms = round((time.perf_counter() - call_start) * 1000, 3)
    counts = usage_counts(response)
    answer = None
    if error is None:
        try:
            answer = response["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            error = "invalid_completion"
    return {
        "sample_id": sample["id"], "context_id": sample["context_id"], "source": sample["source"],
        "question": sample["question"], "reference_answers": sample["answers"], "prediction": answer,
        "status": error or "ok", "correct": error is None and baseline.score(answer, sample["answers"]),
        "started_at_utc": sent_at, "end_to_end_ms": round((time.perf_counter() - start) * 1000, 3),
        "usage": None if counts is None else dict(zip(
            ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_split_reported"), counts, strict=True,
        )),
        "trajectory": [{"operation": "chat_completion", "request": request, "response": response,
                        "http_status": status, "elapsed_ms": request_ms}],
    }


def cost(rate, prompt, output, hit=0):
    return ((prompt - hit) * rate["cache_miss"] + hit * rate["cache_hit"] + output * rate["completion"]) / 1_000_000


def load_test(freeze):
    for path, expected in freeze["test_file_sha256"].items():
        if file_sha(ROOT / "data" / path) != expected:
            raise SystemExit(f"Locked test content changed: {path}")
    samples = [json.loads(line) for line in (ROOT / "data/test/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    contexts = read_json(ROOT / "data/test/contexts.json")
    if len(samples) != freeze["test_size"] or len({row["id"] for row in samples}) != len(samples):
        raise SystemExit("Locked test sample count or uniqueness mismatch.")
    return samples, contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5], help="Build five dev prompts per cell offline.")
    parser.add_argument("--confirmation")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.limit is None and args.confirmation != "确认运行":
        parser.error("Formal test requires the user's explicit confirmation.")
    if args.limit and args.confirmation:
        parser.error("Choose offline dev smoke or an explicitly authorized formal test.")
    start = time.perf_counter()
    freeze = read_json(PLAN / "freeze.json")
    estimate = read_json(PLAN / "estimate.json")
    check_freeze(freeze, estimate)
    baseline = LongContext(ROOT / ".venv" / "nltk_data")
    settings = freeze["settings"]
    cells = [(model, variant, seed) for model, variants in freeze["schedule"].items()
             for variant in variants for seed in freeze["execution"]["seeds"]]
    if args.limit:
        dev, histories, _ = load_dev()
        samples = ordered([sample for sample in dev if sample["source"] == "factconsolidation_sh_6k"], 42)[:args.limit]
        checked = 0
        for model, variant, _ in cells:
            material, _ = prepare_histories(histories, freeze, variant)
            for sample in samples:
                request = payload(baseline, model, settings, sample, material[sample["context_id"]])
                if request["messages"][0]["role"] != "system" or len(request["messages"]) != 2:
                    raise ValueError("Unexpected frozen prompt shape.")
                checked += 1
        print(json.dumps({"split": "dev", "smoke": "offline_prompt_construction", "samples": len(samples),
                          "cells": len(cells), "requests_prepared": checked, "api_calls": 0}, indent=2))
        return

    key = load_key(args.key_file)
    samples, contexts = load_test(freeze)
    execution = freeze["execution"]
    evaluator.TIMEOUT_SECONDS = execution["timeout_seconds"]
    cap = Decimal(execution["aggregate_budget_cap_usd"])
    workers = execution["concurrency"]
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = ROOT / "results" / "test" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "run_manifest.json"
    manifest = {
        "split": "test", "status": "running", "run_id": run_id, "created_at_utc": utc_now().isoformat(),
        "freeze_sha256": file_sha(PLAN / "freeze.json"), "budget_plan_sha256": file_sha(PLAN / "estimate.json"),
        "runner_sha256": file_sha(Path(__file__)), "sample_ids": [row["id"] for row in samples], "cells": [],
    }
    spend = Decimal(0)
    attempted = 0
    try:
        dump_json(manifest_path, manifest)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for model, variant, seed in cells:
                rates = {name: Decimal(value) for name, value in estimate["peak_rates_per_million"][model].items()}
                reservation = cost(rates, MAX_PROVIDER_INPUT_TOKENS, settings["max_tokens"])
                context_text, memory = prepare_histories(contexts, freeze, variant)
                cell_dir = folder / f"{model}-{variant}-{seed}"
                cell_dir.mkdir()
                dump_json(cell_dir / "state_diagnostics.json", memory)
                work = ordered(samples, seed)
                for start_at in range(0, len(work), workers):
                    batch = work[start_at:start_at + workers]
                    if spend + reservation * len(batch) > cap:
                        raise RuntimeError("Aggregate USD budget guard reached before the next batch.")
                    # Construct the shared prefixes sequentially before concurrent requests.
                    for sample in batch:
                        baseline.messages(sample, context_text[sample["context_id"]])
                    futures = [pool.submit(run_case, baseline, model, settings, sample,
                                           context_text[sample["context_id"]], key) for sample in batch]
                    statuses = []
                    for position, future in enumerate(futures, start_at + 1):
                        row = future.result()
                        counted = row["usage"]
                        amount = reservation if counted is None else cost(
                            rates, counted["prompt_tokens"], counted["completion_tokens"], counted["cache_hit_tokens"],
                        )
                        spend += amount
                        attempted += 1
                        statuses.append(row["status"])
                        row["budget_accounted_peak_usd"] = str(amount)
                        dump_json(cell_dir / f"{position:04d}-trajectory.json", row)
                    manifest["attempted_calls"] = attempted
                    manifest["accounted_peak_usd"] = str(spend)
                    dump_json(manifest_path, manifest)
                    if spend > cap or any(status in FATAL for status in statuses):
                        raise RuntimeError("Budget or fatal provider error stopped the run. See the saved trajectories.")
                manifest["cells"].append({"model": model, "variant": variant, "seed": seed, "directory": cell_dir.name})
                dump_json(manifest_path, manifest)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest.update({"attempted_calls": attempted, "accounted_peak_usd": str(spend),
                         "elapsed_seconds": round(time.perf_counter() - start, 3)})
        dump_json(manifest_path, manifest)
    print(folder)


if __name__ == "__main__":
    main()
