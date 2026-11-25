"""Follow-up factual memory benchmark: dev smoke or separately approved locked test."""

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from src.evaluator import ROOT, UPSTREAM, UPSTREAM_COMMIT, LongContext, dump_json, utc_now
from src.v2_data import DATA, EVENT_DIR, read_locked
from src.v2_extract import extract_history
from src.v2_retrieval import BGE, TOP_K, DenseIndex, matched_recent, proxy_tokens
from src.v2_transport import FLASH, GPT, PROVIDER, Meter, credentials, model_probe

FACT_ARMS = ("full_recency", "matched_recent", "dense_top8", "rule", "model", "hybrid", "hybrid_no_archive")
EVENT_ARMS = ("full_recency", "hybrid")
CROSS_ARMS = EVENT_ARMS
SNAPSHOT = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
BGE_FOLDER = ROOT / ".venv/bge_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots" / SNAPSHOT
PLAN_FOLDER = ROOT / "results" / "test_plan_v2"
SEED = 42
WORKERS = 6
SMOKE_CAP = Decimal("0.75")
SOURCE_PATHS = [
    "src/run_v2.py", "src/v2_data.py", "src/v2_extract.py", "src/v2_retrieval.py", "src/v2_transport.py",
    "src/factual_v2.py", "src/memory_state.py", "src/evaluator.py", "baselines/long_context.py",
    "src/v2_report.py", "data/split-lock.json", "data/eventqa_v2/split-lock.json",
    "baselines/MemoryAgentBench/utils/templates.py", "baselines/MemoryAgentBench/utils/eval_other_utils.py",
]


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def snapshot():
    if not (BGE_FOLDER / "model_optimized.onnx").is_file():
        raise ValueError("Install the pinned BGE snapshot before running.")
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                          text=True, capture_output=True, check=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("Upstream snapshot changed.")
    return {"files": {name: hash_file(ROOT / name) for name in SOURCE_PATHS},
            "bge_revision": SNAPSHOT, "bge_model": BGE,
            "bge_files": {file.name: hash_file(file) for file in sorted(BGE_FOLDER.iterdir()) if file.is_file()},
            "upstream": head, "seed": SEED, "temperature": 0, "top_k": TOP_K,
            "answer_max_tokens": {"factcon": 10, "eventqa": 40}, "gpt_test_items": 20}


def choose(samples, n, balanced=False):
    if not balanced:
        subset = sorted(samples, key=lambda row: row["id"])
        random.Random(SEED).shuffle(subset)
        return subset[:n]
    groups = [[x for x in samples if f"_{hop}_" in x["source"]] for hop in ("sh", "mh")]
    counts = [(n + 1) // 2, n // 2]
    if any(len(group) < count for group, count in zip(groups, counts, strict=True)):
        raise ValueError("Insufficient SH or MH samples for the pre-specified cohort.")
    return [row for group, count in zip(groups, counts, strict=True) for row in choose(group, count)]


def prepare_context(history, dataset, baseline, embedder, meter, persist):
    protocol = "numbered" if dataset == "factcon" else "dated"
    modes = ("rule", "model", "hybrid") if dataset == "factcon" else ("hybrid",)
    derived, reports = {}, {}
    for mode in modes:
        derived[mode], reports[mode] = extract_history(history, protocol, mode, meter.chat, persist)
    common = {"full_recency": history, "hybrid_no_archive": history, **derived}
    local = {}
    index = None
    if dataset == "factcon":
        start = time.perf_counter()
        common["matched_recent"] = matched_recent(
            history, derived["hybrid"], lambda text: baseline._memorize("factconsolidation_sh_6k", text),
        )
        local["matched_recent_preparation_ms"] = round((time.perf_counter() - start) * 1000, 3)
        index = DenseIndex(embedder, history)
        local.update({"dense_index_preparation_ms": index.build_ms, "dense_passages": len(index.texts)})
    return common, reports, index, local


def score(baseline, row, dataset, call):
    if call["error"]:
        return False, {}
    answer = call["answer"]
    if dataset == "factcon":
        return baseline.score(answer, row["answers"]), {}
    result, _ = baseline.scoring["post_process"](
        {"output": answer}, row["answers"], {"sub_dataset": row["source"]},
    )
    reported = {key: float(result[key]) for key in ("eventqa_recall", "substring_exact_match") if key in result}
    return bool(reported["eventqa_recall"]), reported


def run_dataset(dataset, split, gpt_enabled, baseline, embedder, meter, out, limit):
    samples, contexts = read_locked(dataset, split)
    if limit:
        samples = choose(samples, limit, balanced=dataset == "factcon")
    gpt_ids = set()
    if dataset == "factcon" and gpt_enabled:
        eligible = samples if limit else [row for row in samples if row["source"].endswith("_32k")]
        # The same sample IDs are also included in the Flash main matrix.
        gpt_ids = {row["id"] for row in choose(eligible, limit or 20, balanced=True)}
    by_context = defaultdict(list)
    for sample in samples:
        by_context[sample["context_id"]].append(sample)
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for history_id, questions in sorted(by_context.items()):
            history = contexts[history_id]
            work = out / "histories" / (dataset + "-" + history_id)
            work.mkdir(parents=True)
            persist = lambda task: dump_json(out / "requests" / f"{task['id']}.json", task)
            common, report, dense, local = prepare_context(history, dataset, baseline, embedder, meter, persist)
            dump_json(work / "projection.json", {"dataset": dataset, "context_id": history_id, "modes": report,
                                                "local_preparation": local, "raw_history": history})
            print(f"{dataset}: prepared {len(questions)} questions for one history", flush=True)
            for sample in questions:
                arms = FACT_ARMS if dataset == "factcon" else EVENT_ARMS
                schedules = [(FLASH, arms)]
                if sample["id"] in gpt_ids:
                    schedules.append((GPT, CROSS_ARMS))
                novel = {}
                for model, model_arms in schedules:
                    budget = 10 if dataset == "factcon" else 40
                    for arm in model_arms:
                        start = time.perf_counter()
                        hits = None
                        if arm == "dense_top8":
                            visible, hits = dense.search(sample["question"])
                        else:
                            visible = common[arm]
                        request = baseline.messages(sample, visible)
                        token_proxy = sum(proxy_tokens(msg["content"]) for msg in request)
                        if token_proxy + 1024 + budget > PROVIDER[model][2]:
                            raise ValueError(f"Model context size preflight failed for {model}/{sample['id']}.")
                        prep_ms = round((time.perf_counter() - start) * 1000, 3)
                        key = model, json.dumps(request, ensure_ascii=False, sort_keys=True)
                        if key not in novel:
                            novel[key] = [request, []]
                        novel[key][1].append((model, arm, hits, token_proxy, prep_ms))
                futures = [
                    (arms, pool.submit(meter.chat, arms[0][0], request, budget, "answer"))
                    for request, arms in novel.values()
                ]
                first_error = None
                for variants, future in futures:
                    try:
                        task = future.result()
                    except Exception as error:
                        first_error = first_error or error
                        continue
                    persist(task)
                    correct, extra = score(baseline, sample, dataset, task)
                    for index, (model, arm, hits, token_proxy, prep_ms) in enumerate(variants):
                        mode = "hybrid" if arm == "hybrid_no_archive" else arm
                        log = {
                            "dataset": dataset, "split": split, "sample_id": sample["id"], "source": sample["source"],
                            "context_id": history_id, "model": model, "arm": arm, "question": sample["question"],
                            "reference_answers": sample["answers"], "prediction": task["answer"], "correct": correct,
                            "status": task["error"] or "ok", "additional_scores": extra,
                            "physical_request": f"requests/{task['id']}.json", "shared_answer": index != 0,
                            "usage": task["usage"], "token_proxy_o200k_unframed": token_proxy,
                            "milliseconds": task["elapsed_ms"] + prep_ms, "dense_passages": hits,
                            "archive_count": len(report[mode]["archive"]) if mode in report and arm != "hybrid_no_archive" else 0,
                            "extraction_request_count": len(report[mode]["calls"]) if mode in report else 0,
                            "projection_log": (work / "projection.json").relative_to(out).as_posix(),
                        }
                        ident = sha256(f"{dataset}|{sample['id']}|{model}|{arm}".encode()).hexdigest()[:24]
                        dump_json(out / "samples" / f"{ident}.json", log)
                        rows.append(log)
                if first_error:
                    raise first_error
                if limit:
                    print(f"{dataset}: logged {sample['id']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--limit", type=int, choices=[5])
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--confirmation")
    parser.add_argument("--approved-quote", type=Path)
    parser.add_argument("--budget-usd", type=Decimal)
    args = parser.parse_args()
    smoke = args.split == "dev"
    if smoke and (args.limit != 5 or any((args.confirmation, args.approved_quote, args.budget_usd))):
        parser.error("Dev mode allows only --limit 5 at the built-in smoke budget.")
    if not smoke and (args.limit or args.confirmation != "确认运行" or not args.approved_quote
                      or args.budget_usd is None or not args.budget_usd.is_finite() or args.budget_usd <= 0):
        parser.error("Locked test requires the separately confirmed v2 quote and explicit approved budget.")
    code = snapshot()
    keys = {model: credentials(model, args.key_file) for model in PROVIDER}
    if not keys[FLASH]:
        raise SystemExit("DeepSeek credentials unavailable.")
    if smoke:
        probe = model_probe(GPT, keys[GPT])
        gpt_enabled, cap = probe["available"], SMOKE_CAP
    else:
        if not args.approved_quote.resolve().is_relative_to((ROOT / "results/dev").resolve()):
            raise SystemExit("Use a saved dev quote.")
        previous = json.loads(args.approved_quote.read_text(encoding="utf-8"))
        if previous.get("configuration") != code or not previous.get("dev_smoke_complete"):
            raise SystemExit("Quote must come from a completed dev smoke of the exact current configuration.")
        if args.budget_usd > Decimal(previous["proposed_budget_ceiling_usd"]):
            raise SystemExit("Budget exceeds the quoted ceiling.")
        gpt_enabled = previous["cross_model_enabled"]
        if gpt_enabled and not model_probe(GPT, keys[GPT])["available"]:
            raise SystemExit("The approved cross-model snapshot is unavailable.")
        cap = args.budget_usd
        freeze = {**code, "dev_quote_sha256": hash_file(args.approved_quote), "budget_usd": str(cap),
                  "flash_arms": FACT_ARMS, "eventqa_arms": EVENT_ARMS, "cross_gpt": gpt_enabled}
        PLAN_FOLDER.mkdir(parents=True, exist_ok=True)
        content = json.dumps(freeze, indent=2, sort_keys=True) + "\n"
        lock = PLAN_FOLDER / "freeze.json"
        if lock.exists() and lock.read_text(encoding="utf-8") != content:
            raise SystemExit("Existing v2 freeze differs.")
        if not lock.exists():
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(content)
    from fastembed import TextEmbedding
    from src.v2_report import report_run

    embedder = TextEmbedding(BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
                             local_files_only=True, specific_model_path=str(BGE_FOLDER))
    baseline = LongContext(ROOT / ".venv/nltk_data")
    out = ROOT / "results" / args.split / ("v2-" + utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6])
    for name in ("requests", "samples", "histories"):
        (out / name).mkdir(parents=True)
    meter = Meter(keys, cap)
    manifest = {"split": args.split, "limit": args.limit, "configuration": code, "status": "running",
                "cross_model_enabled": gpt_enabled, "started_at": utc_now().isoformat(), "budget_usd": str(cap)}
    if smoke:
        manifest["cross_model_probe"] = probe
    else:
        manifest["freeze_sha256"] = hash_file(PLAN_FOLDER / "freeze.json")
    dump_json(out / "manifest.json", manifest)
    try:
        for dataset in ("factcon", "eventqa"):
            run_dataset(dataset, args.split, gpt_enabled, baseline, embedder, meter, out, args.limit)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "stopped"
        raise
    finally:
        manifest.update({"calls": dict(meter.calls), "budget_accounted_usd": str(meter.spent)})
        dump_json(out / "manifest.json", manifest)
    report_run(out)
    print(out)


if __name__ == "__main__":
    main()
