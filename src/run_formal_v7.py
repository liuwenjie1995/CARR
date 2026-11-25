"""Execute or offline-smoke the frozen A-group protocol."""

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import mean
from threading import Lock

from src.evaluator import ROOT, LongContext, dump_json, token_cost, utc_now
from src.formal_v7 import CONCURRENCY, EVENT_ARMS, FACT_ARMS, FactHistory, archive, event_prompts, history_batches, prefix_id
from src.freeze_formal_v7 import DEST, hashes, smoke
from src.hop_retrieval import model_messages, model_plan
from src.run_hop_dev import bounded, PLANNER_INPUT_LIMIT
from src.run_v2 import BGE_FOLDER, score
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE
from src.v2_transport import BudgetError, FLASH, PROVIDER, TIMEOUT, credentials, request_json, usage_of

D = Decimal
SHANGHAI = timezone(timedelta(hours=8))
PEAK_STARTS = (9 * 60, 14 * 60)
PEAK_ENDS = (12 * 60, 18 * 60)


class OffpeakMeter:
    def __init__(self, out, key, cap):
        self.key, self.cap, self.out = key, cap, out
        self.spent = self.reserved = D(0)
        self.usage_cost = D(0)
        self.lock = Lock()
        self.stopped = False

    def call(self, request_id, messages, max_tokens, *, planner=False, prefix=None):
        payload = {"model": FLASH, "temperature": 0, "max_tokens": max_tokens,
                   "messages": messages, "thinking": {"type": "disabled"}}
        if planner:
            payload["response_format"] = {"type": "json_object"}
        ceiling = min(PROVIDER[FLASH][2], sum(len(m["content"].encode()) for m in messages) + 1024)
        reserve = token_cost(0, ceiling, max_tokens, "off_peak")
        with self.lock:
            now = utc_now().astimezone(SHANGHAI)
            minute = now.hour * 60 + now.minute
            if now.weekday() < 5 and any(start - 3 <= minute < end for start, end in zip(PEAK_STARTS, PEAK_ENDS)):
                self.stopped = True
                raise BudgetError("Paused before a peak-price window.")
            if self.stopped or self.spent + self.reserved + reserve > self.cap:
                self.stopped = True
                raise BudgetError("Stopped before authorizing the next request under the spend cap.")
            self.reserved += reserve
        started = time.perf_counter()
        status, response, error = request_json(FLASH, "/chat/completions", self.key, payload)
        elapsed = (time.perf_counter() - started) * 1000
        use = usage_of(response)
        charge = None if use is None else token_cost(use["cache_hit_tokens"],
                                                    use["prompt_tokens"] - use["cache_hit_tokens"],
                                                    use["completion_tokens"], "off_peak")
        answer, finish = None, None
        if not error:
            try:
                first = response["choices"][0]
                answer, finish = first["message"]["content"], first["finish_reason"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty")
            except (TypeError, KeyError, IndexError, ValueError):
                error = "invalid_completion"
        with self.lock:
            self.reserved -= reserve
            self.spent += reserve if charge is None else charge
            if charge is not None:
                self.usage_cost += charge
            if status in {401, 402, 403, 404}:
                self.stopped = True
        result = {"id": request_id, "started_beijing": now.isoformat(), "elapsed_ms": round(elapsed, 3),
                  "prefix_sha256": prefix, "request": payload, "response": response, "http_status": status,
                  "error": error, "answer": answer, "finish": finish, "usage": use,
                  "usage_usd": str(charge) if charge is not None else None,
                  "reserved_if_usage_missing_usd": str(reserve) if charge is None else None}
        dump_json(self.out / "requests" / (request_id + ".json"), result)
        return result


class Experiment:
    def __init__(self, meter, freeze):
        self.meter, self.freeze = meter, freeze
        self.out = meter.out
        self.baseline = LongContext(ROOT / ".venv/nltk_data")
        self.phase = "initial"

    def case(self, dataset, sample, arm, call, prep_ms=0, reused=False):
        correct, metrics = score(self.baseline, sample, dataset, call)
        row = {"dataset": dataset, "arm": arm, "sample_id": sample["id"], "source": sample["source"],
               "context_id": sample["context_id"], "question": sample["question"], "reference": sample["answers"],
               "prediction": call["answer"], "correct": correct, "metrics": metrics,
               "error": call["error"], "finish": call["finish"], "request_id": call["id"], "reused": reused,
               "usage": call["usage"], "usage_usd": call["usage_usd"],
               "reserved_if_usage_missing_usd": call["reserved_if_usage_missing_usd"],
               "end_to_end_ms": round(prep_ms + (0 if reused else call["elapsed_ms"]), 3)}
        dump_json(self.out / "cases" / (arm + "--" + sample["id"] + ".json"), row)

    def answer(self, dataset, sample, arm, messages, *, prefix=None, prep_ms=0):
        call = self.meter.call(arm + "--" + sample["id"], messages, self.freeze["answer_max_tokens"][dataset],
                               prefix=prefix)
        self.case(dataset, sample, arm, call, prep_ms)
        return call

    def mark(self, phase, context_id=""):
        self.phase = phase
        dump_json(self.out / "progress.json", {"phase": phase, "context_id": context_id, "updated_utc": utc_now().isoformat(),
                                              "budget_accounted_usd": str(self.meter.spent),
                                              "usage_usd": str(self.meter.usage_cost)})

    def long_batch(self, pool, first, rest, task):
        for sample in first:
            task(sample)
        for _ in pool.map(task, rest):
            pass

    def run(self, fact, event):
        rows, contexts = fact
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            for context_id, first, rest in history_batches(rows):
                self.mark(FACT_ARMS[0], context_id)
                history = contexts[context_id]
                prefix = prefix_id(self.baseline.messages(first[0], history), first[0]["question"])

                def full(sample):
                    started = time.perf_counter()
                    messages = self.baseline.messages(sample, history)
                    if prefix_id(messages, sample["question"]) != prefix:
                        raise ValueError("Full-context prefix changed within a history.")
                    self.answer("factcon", sample, FACT_ARMS[0], messages, prefix=prefix,
                                prep_ms=(time.perf_counter() - started) * 1000)

                self.long_batch(pool, first, rest, full)

            self.run_eventqa(pool, event)
            self.run_compact(pool, rows, contexts)

    def run_eventqa(self, pool, event):
        rows, contexts = event
        for context_id, first, rest in history_batches(rows):
            self.mark("eventqa", context_id)
            history = contexts[context_id]
            _, clean, removed = archive(history, "dated")
            dump_json(self.out / "histories" / ("eventqa--" + context_id + ".json"),
                      {"total_lines": len(history.splitlines()), "removed": removed, "identical_context": history == clean})

            def before(sample):
                started = time.perf_counter()
                prompts, equal = event_prompts(sample, self.baseline, history, clean)
                msg = prompts[EVENT_ARMS[0]]
                call = self.answer("eventqa", sample, EVENT_ARMS[0], msg, prefix=prefix_id(msg, sample["question"]),
                                   prep_ms=(time.perf_counter() - started) * 1000)
                if equal:
                    self.case("eventqa", sample, EVENT_ARMS[1], call, reused=True)

            def after(sample):
                if (self.out / "cases" / (EVENT_ARMS[1] + "--" + sample["id"] + ".json")).exists():
                    return
                started = time.perf_counter()
                msg = self.baseline.messages(sample, clean)
                self.answer("eventqa", sample, EVENT_ARMS[1], msg, prefix=prefix_id(msg, sample["question"]),
                            prep_ms=(time.perf_counter() - started) * 1000)

            self.long_batch(pool, first, rest, before)
            if history != clean:
                self.long_batch(pool, first, rest, after)

    def run_compact(self, pool, rows, contexts):
        from fastembed import TextEmbedding

        embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                                 local_files_only=True, specific_model_path=str(BGE_FOLDER))
        for context_id, first, rest in history_batches(rows):
            self.mark("factcon_compact_index", context_id)
            start = time.perf_counter()
            source = contexts[context_id]
            history = FactHistory(source, embedder)
            dump_json(self.out / "histories" / ("factcon--" + context_id + ".json"),
                      {"removed": history.archive, "total_lines": len(source.splitlines()),
                       "index_ms": round((time.perf_counter() - start) * 1000, 3)})
            self.mark("factcon_compact_answers", context_id)

            def prepare(sample):
                start = time.perf_counter()
                route = history.route(sample["question"])
                plan, call = None, None
                if route["route"] == "multi":
                    msg = bounded(model_messages(sample["question"]), PLANNER_INPUT_LIMIT)
                    call = self.meter.call("plan--" + sample["id"], msg, self.freeze["planner"]["max_tokens"], planner=True)
                    plan = model_plan(call, sample["question"])
                return sample, route, plan, call, start

            def dispatch(item):
                sample, route, plan, call, start = item
                prompts, trace = history.prompts(sample, self.baseline, route, plan)
                trace.update({"sample_id": sample["id"], "context_id": context_id,
                              "planner_request": call["id"] if call else None})
                dump_json(self.out / "traces" / (sample["id"] + ".json"), trace)
                prep_ms = (time.perf_counter() - start) * 1000
                return [(sample, arm, prompts[arm], prep_ms if arm != FACT_ARMS[0] else 0) for arm in FACT_ARMS[1:]]

            ordered = first + rest
            for offset in range(0, len(ordered), CONCURRENCY):
                prepared = pool.map(prepare, ordered[offset:offset + CONCURRENCY])
                tasks = [task for item in prepared for task in dispatch(item)]
                for _ in pool.map(lambda task: self.answer("factcon", *task[:3], prep_ms=task[3]), tasks):
                    pass


def run(budget, key_file):
    frozen = load(DEST / "freeze.json")
    if hashes() != frozen["source_sha256"]:
        raise ValueError("Frozen source or data-lock fingerprints differ.")
    key = credentials(FLASH, key_file)
    if not key:
        raise ValueError("DeepSeek credentials unavailable.")
    fact, event = read_locked("factcon", "test"), read_locked("eventqa", "test")
    if [len(fact[0]), len(event[0])] != [562, 400]:
        raise ValueError("Unexpected locked test sizes.")
    out = ROOT / "results/test" / f"formal-v7-{utc_now():%Y%m%dT%H%M%S%fZ}"
    for folder in ("requests", "cases", "traces", "histories"):
        (out / folder).mkdir(parents=True)
    manifest = {"status": "running", "budget_cap_usd": str(budget), "price_period": "off_peak", "split": "test",
                "freeze_sha256": sha256((DEST / "freeze.json").read_bytes()).hexdigest(),
                "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "freeze": frozen, "factcon_questions": len(fact[0]), "eventqa_questions": len(event[0]),
                "started_utc": utc_now().isoformat(), "timeout_seconds": TIMEOUT}
    dump_json(out / "manifest.json", manifest)
    print(out, flush=True)
    meter = OffpeakMeter(out, key, budget)
    experiment = Experiment(meter, frozen)
    try:
        experiment.run(fact, event)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update({"status": "stopped", "error_type": type(error).__name__, "failed_phase": experiment.phase})
        raise
    finally:
        manifest.update({"ended_utc": utc_now().isoformat(), "accounted_usd": str(meter.spent),
                         "usage_usd": str(meter.usage_cost)})
        dump_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(5,))
    parser.add_argument("--confirmed", action="store_true")
    parser.add_argument("--budget", type=D)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.limit == 5 and not any((args.confirmed, args.budget, args.key_file)):
        print({k: v for k, v in smoke(5).items() if k != "source_sha256"})
    elif args.limit is None and args.confirmed and args.budget is not None and D(0) < args.budget <= D(15):
        run(args.budget, args.key_file)
    else:
        parser.error("Run --limit 5 offline, or supply explicit confirmation and an authorized budget.")
