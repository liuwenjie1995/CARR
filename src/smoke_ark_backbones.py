"""Replay frozen development questions through Ark Agent Plan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from getpass import getpass
import json
import os
from pathlib import Path
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from src.evaluator import LongContext, NoRedirect, ROOT, dump_json, utc_now
from src.v2_report import load
from src.v2_transport import usage_of
from statistics import mean

DEV = ROOT / "results/dev/unified-prompt-v7-20261123T133145804847Z-paid"
BASE = ROOT / "results/dev/v3-20261123T111806Z-586a10"
URL = "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions"
MODELS = {"kimi-k3.1-pro": Decimal("7"), "glm-5.4": Decimal("4.5")}
CAP_AFP = Decimal("200")
OUTPUT_CAP = 10
TIMEOUT = 45
EXTENSION_CAP = Decimal("134")
DISCOUNTED = {"kimi-k3.1-pro": Decimal("3.5"), "glm-5.4": Decimal("3.6")}
SMOKES = {"kimi-k3.1-pro": "ark-backbones-smoke-20261124T004938Z",
          "glm-5.4": "ark-backbones-smoke-20261124T004825Z"}


def native_payload(model, messages):
    return {"model": model, "messages": messages, "temperature": 1 if model == "kimi-k3.1-pro" else 0.6,
            "max_tokens": 512}


def prepare(limit):
    for i in range(1, limit + 1):
        dev = load(DEV / "cases" / f"{i:02d}.json")
        base = load(BASE / "samples" / f"{i:02d}-full.json")
        assert dev["sample_id"] == base["sample_id"] and dev["reference"] == base["reference_answers"]
        full = load(BASE / "requests" / f'{base["answer_request"]}.json')
        assert full["request"]["max_tokens"] == OUTPUT_CAP
        yield dev, {"official_full_context": full["request"]["messages"],
                    "unified_a": dev["variants"]["official_evidence"]["messages"]}


def reservation(messages, coefficient, output_cap):
    tokens = sum(len(item["content"].encode("utf-8")) for item in messages) + 1024
    return (tokens + output_cap) * coefficient / 10000


def request(payload, key, *, timeout=TIMEOUT):
    req = Request(URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with build_opener(NoRedirect).open(req, timeout=timeout) as result:
            status, body = result.status, json.loads(result.read().decode("utf-8").replace(key, "[REDACTED]"))
    except HTTPError as error:
        with error:
            status = error.code
            data = error.read(8192).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        try:
            body = json.loads(data)
        except ValueError:
            body = {"error": data}
    except (URLError, ValueError, OSError) as error:
        status, body = None, {"error": type(error).__name__}
    return status, body, round((time.perf_counter() - start) * 1000, 3)


def extend(limit, execute):
    cases = list(prepare(limit))
    seeds = []
    for model, folder in SMOKES.items():
        for dev, arms in cases[:5]:
            for arm, messages in arms.items():
                path = ROOT / "results/dev" / folder / f"{model}--{arm}--{dev['sample_id']}.json"
                row = load(path)
                if (row["request"] != native_payload(model, messages) or row["reference"] != dev["reference"]
                        or row["http_status"] != 200 or row["correct"] is None or row["usage"] is None):
                    raise ValueError(f"Invalid existing development result for {model}, {arm}, {dev['sample_id']}.")
                seeds.append({**{k: v for k, v in row.items() if k not in {"request", "response", "reference"}},
                              "reused": True, "source": path.relative_to(ROOT).as_posix(),
                              "served_model": row["response"].get("model")})

    def afp(usage, coeff):
        return (usage["prompt_tokens"] + usage["completion_tokens"]) * coeff / 10000

    estimate = sum(afp(row["usage"], DISCOUNTED[row["model_requested"]]) for row in seeds) * (limit - 5) / 5
    print(f"{len(seeds)} verified reused answers; {(limit - 5) * 4} new requests; "
          f"incremental no-cache estimate {estimate:.2f} AFP; guard {EXTENSION_CAP} AFP.", flush=True)
    if not execute:
        return

    key = os.environ.get("ARK_AGENT_PLAN_API_KEY") or getpass("Agent Plan API key (hidden): ")
    if not key.strip():
        raise SystemExit("Missing Agent Plan key.")
    scorer = LongContext(ROOT / ".venv/nltk_data")
    out = ROOT / "results/dev" / f"ark-backbones-v7-{utc_now():%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    lock = Lock()
    spent, reserved = Decimal(0), Decimal(0)
    rows = list(seeds)
    stops = {}

    def run_model(model, coeff):
        nonlocal spent, reserved
        for index, (dev, arms) in enumerate(cases[5:], 6):
            for arm, messages in arms.items():
                maximum = reservation(messages, coeff, 512)
                with lock:
                    if spent + reserved + maximum > EXTENSION_CAP:
                        stops[model] = "AFP guard"
                        return
                    reserved += maximum
                payload = native_payload(model, messages)
                status, response, latency = request(payload, key)
                usage = usage_of(response)
                choice = (response.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                answer = message.get("content")
                answered = status == 200 and isinstance(answer, str) and bool(answer.strip())
                row = {"sample_id": dev["sample_id"], "type": dev["type"], "model_requested": model, "arm": arm,
                       "served_model": response.get("model"), "temperature": payload["temperature"],
                       "http_status": status, "finish_reason": choice.get("finish_reason"),
                       "thinking_returned": bool(message.get("reasoning_content") or message.get("encrypted_content")),
                       "correct": scorer.score(answer, dev["reference"]) if answered else None,
                       "usage": usage, "elapsed_ms": latency, "reused": False}
                path = out / f"{model}--{arm}--{dev['sample_id']}.json"
                row["source"] = path.relative_to(ROOT).as_posix()
                dump_json(path, {**row, "request": payload, "response": response, "reference": dev["reference"],
                                 "frozen_development_case": (DEV / "cases" / f"{index:02d}.json")
                                 .relative_to(ROOT).as_posix()})
                with lock:
                    spent += afp(usage, coeff) if usage is not None else maximum
                    reserved -= maximum
                    rows.append(row)
                    if status in (400, 401, 402, 403, 404, 429):
                        stops[model] = f"HTTP {status}"
                        return

    print(out, flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_model, model, coeff) for model, coeff in DISCOUNTED.items()]
        for job in jobs:
            job.result()

    totals = []
    for model in DISCOUNTED:
        for arm in cases[0][1]:
            selected = [r for r in rows if r["model_requested"] == model and r["arm"] == arm]
            for label in ("all", "single", "multi"):
                group = [r for r in selected if label == "all" or r["type"] == label]
                totals.append({"model": model, "arm": arm, "type": label, "recorded": len(group),
                               "correct": sum(r["correct"] is True for r in group),
                               "unscored": sum(r["correct"] is None for r in group),
                               "length_limited": sum(r["finish_reason"] == "length" for r in group),
                               "reused": sum(r["reused"] for r in group),
                               "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in group if r["usage"]),
                               "completion_tokens": sum(r["usage"]["completion_tokens"] for r in group if r["usage"]),
                               "cached_tokens": sum(r["usage"]["cache_hit_tokens"] for r in group if r["usage"]),
                               "mean_api_latency_ms": round(mean(r["elapsed_ms"] for r in group), 2) if group else None})
    dump_json(out / "summary.json", {"generator": "src/smoke_ark_backbones.py", "split": "dev", "limit": limit,
              "status": "complete" if len(rows) == limit * 4 and not stops else "partial",
              "stops": stops, "new_api_calls": len(rows) - len(seeds), "reused": len(seeds),
              "new_afp_no_cache_accounted_not_invoice": str(spent),
              "afp_guard": str(EXTENSION_CAP), "totals": totals, "results": rows})


def run(limit, execute, previous, model_only):
    cases = list(prepare(limit))
    native = previous is not None
    already = Decimal(load(previous)["afp_budget_accounted_not_invoice"]) if native else Decimal(0)
    output_cap = 512 if native else OUTPUT_CAP
    models = {model: MODELS[model] for model in ([model_only] if model_only else MODELS)}
    ceiling = sum(reservation(messages, coeff, output_cap) for _, arms in cases for messages in arms.values()
                  for coeff in models.values())
    if ceiling > CAP_AFP:
        raise ValueError("The planned smoke exceeds the AFP guard.")
    print(f"{limit} development questions; at most {len(models) * 2 * limit} requests; "
          f"conservative standalone allowance {ceiling} AFP; {already} AFP previously accounted.", flush=True)
    if not execute:
        return

    scorer = LongContext(ROOT / ".venv/nltk_data")
    key = os.environ.get("ARK_AGENT_PLAN_API_KEY") or getpass("Agent Plan API key (hidden): ")
    if not key.strip():
        raise SystemExit("Missing Agent Plan key.")
    out = ROOT / "results/dev" / f"ark-backbones-smoke-{utc_now():%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True)
    accounted = already
    results = []
    for model, coeff in models.items():
        temperature = (1 if model == "kimi-k3.1-pro" else 0.6) if native else 0
        blocked = False
        for dev, arms in cases:
            if blocked:
                break
            for arm, messages in arms.items():
                held = reservation(messages, coeff, output_cap)
                if accounted + held > CAP_AFP:
                    raise SystemExit("AFP guard stopped further requests.")
                payload = {"model": model, "messages": messages, "temperature": temperature,
                           "max_tokens": output_cap}
                if not native:
                    payload["thinking"] = {"type": "disabled"}
                status, response, elapsed = request(payload, key)
                usage = usage_of(response)
                choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
                message = choice.get("message") or {}
                answer = message.get("content")
                finish = choice.get("finish_reason")
                thinking = bool(message.get("reasoning_content") or message.get("encrypted_content"))
                answered = bool(status == 200 and isinstance(answer, str) and answer.strip())
                compatible = answered and not thinking and not native
                scored = answered and (native or not thinking)
                correct = scorer.score(answer, dev["reference"]) if scored else None
                accounted += held if usage is None else (
                    (usage["prompt_tokens"] + usage["completion_tokens"]) * coeff / 10000
                )
                row = {"sample_id": dev["sample_id"], "type": dev["type"], "model_requested": model,
                       "arm": arm, "temperature": temperature, "http_status": status, "finish_reason": finish, "thinking_returned": thinking,
                       "compatible_with_original": compatible, "correct": correct, "usage": usage, "elapsed_ms": elapsed}
                dump_json(out / f"{model}--{arm}--{dev['sample_id']}.json",
                          {**row, "request": payload, "response": response, "reference": dev["reference"]})
                results.append(row)
                dump_json(out / "summary.json", {"afp_cap": str(CAP_AFP),
                          "max_output_tokens": output_cap, "thinking_requested": "default" if native else "disabled",
                          "afp_budget_accounted_not_invoice": str(accounted), "results": results})
                print(model, arm, dev["sample_id"], status, finish, answered, correct, flush=True)
                if status != 200 or (not native and not compatible):
                    blocked = True
                    break
    print(out, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--limit", required=True, type=int, choices=(5, 25))
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--continue-from", type=Path, help="Previous smoke summary for shared AFP accounting.")
    cli.add_argument("--model", choices=tuple(MODELS))
    cli.add_argument("--extend", action="store_true")
    args = cli.parse_args()
    if args.extend:
        if args.continue_from or args.model or (args.execute and args.limit != 25):
            cli.error("Use --extend --limit 25 --execute for the authorized development expansion.")
        extend(args.limit, args.execute)
    elif args.limit == 5:
        run(args.limit, args.execute, args.continue_from, args.model)
    else:
        cli.error("--limit 25 requires --extend.")
