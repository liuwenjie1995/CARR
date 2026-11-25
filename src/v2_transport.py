"""Metered HTTP transport for the frozen follow-up, with no automatic retries."""

import json
import os
import re
import time
from collections import Counter
from decimal import Decimal
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from src.evaluator import NoRedirect

FLASH = "deepseek-flash"
GPT = "gpt-4.1-mini-2025-04-14"
PROVIDER = {
    FLASH: ("https://api.deepseek.com", "DEEPSEEK_API_KEY", 1_048_576),
    GPT: ("https://api.openai.com/v1", "OPENAI_API_KEY", 1_047_576),
}
# USD per million cached input, uncached input and output. Flash uses peak rates.
RATES = {FLASH: ("0.006", "0.30", "1.20"), GPT: ("0.10", "0.40", "1.60")}
TIMEOUT = 120


def credentials(model, key_file):
    name = PROVIDER[model][1]
    if key := os.environ.get(name, "").strip():
        return key
    if not key_file:
        return None
    label = r"deepseek|\bds\b|深度求索" if model == FLASH else r"openai|\bgpt\b"
    values = [
        key for line in key_file.read_text(encoding="utf-8-sig").splitlines()
        if re.search(label, line, re.I) for key in re.findall(r"\bsk-[A-Za-z0-9_-]{12,}", line)
    ]
    return values[0] if len(values) == 1 else None


def quote(model, prompt, completion, hit=0):
    cached, full, output = map(Decimal, RATES[model])
    return (hit * cached + (prompt - hit) * full + completion * output) / 1_000_000


def request_json(model, path, key, payload=None):
    req = Request(
        PROVIDER[model][0] + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with build_opener(NoRedirect).open(req, timeout=TIMEOUT) as reply:
            value = json.loads(reply.read().decode("utf-8").replace(key, "[REDACTED]"))
            return reply.status, value, None
    except HTTPError as error:
        with error:
            message = error.read(8192).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        return error.code, {"error_body": message}, f"http_{error.code}"
    except (URLError, OSError, ValueError) as error:
        return None, None, type(error).__name__


def model_probe(model, key):
    if not key:
        return {"available": False, "reason": "credential_unavailable"}
    status, body, error = request_json(model, "/models/" + model, key)
    return {"available": status == 200 and isinstance(body, dict) and body.get("id") == model, "error": error}


def usage_of(body):
    use = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(use, dict):
        return None
    prompt, output = use.get("prompt_tokens"), use.get("completion_tokens")
    if not all(type(x) is int and x >= 0 for x in (prompt, output)):
        return None
    hit = use.get("prompt_cache_hit_tokens")
    if hit is None and isinstance(use.get("prompt_tokens_details"), dict):
        hit = use["prompt_tokens_details"].get("cached_tokens", 0)
    hit = hit if type(hit) is int and 0 <= hit <= prompt else 0
    return {"prompt_tokens": prompt, "completion_tokens": output, "cache_hit_tokens": hit}


class BudgetError(RuntimeError):
    pass


class Meter:
    def __init__(self, keys, cap):
        self.keys = keys
        self.cap = Decimal(cap)
        self.spent = Decimal(0)
        self.reserved = Decimal(0)
        self.calls = Counter()
        self.lock = Lock()

    def chat(self, model, messages, output_cap, phase, *, json_output=False):
        # UTF-8 byte bound plus framing margin, cropped to the provider hard limit.
        bound = min(PROVIDER[model][2], sum(len(m["content"].encode("utf-8")) for m in messages) + 1024)
        reserve = quote(model, bound, output_cap)
        with self.lock:
            if self.spent + self.reserved + reserve > self.cap:
                raise BudgetError(f"Budget guard stopped before {phase}. Accounted USD {self.spent}.")
            self.reserved += reserve
            self.calls[phase] += 1
            index = self.calls[phase]
        payload = {"model": model, "temperature": 0, "max_tokens": output_cap, "messages": messages}
        if model == FLASH:
            payload["thinking"] = {"type": "disabled"}
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        request_start = time.perf_counter()
        code, body, error = request_json(model, "/chat/completions", self.keys[model], payload)
        elapsed = round((time.perf_counter() - request_start) * 1000, 3)
        usage = usage_of(body)
        amount = reserve if usage is None else quote(
            model, usage["prompt_tokens"], usage["completion_tokens"], usage["cache_hit_tokens"],
        )
        with self.lock:
            self.reserved -= reserve
            self.spent += amount
        answer, finish = None, None
        if error is None:
            try:
                choice = body["choices"][0]
                answer, finish = choice["message"]["content"], choice.get("finish_reason")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError
            except (KeyError, IndexError, TypeError, ValueError):
                error = "invalid_completion"
        return {
            "id": f"{phase}-{index:06d}", "phase": phase, "model": model, "request": payload,
            "response": body, "http_status": code, "error": error, "answer": answer, "finish": finish,
            "usage": usage, "elapsed_ms": elapsed, "budget_accounted_usd": str(amount),
        }
