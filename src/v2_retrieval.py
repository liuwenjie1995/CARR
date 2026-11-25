"""Local dense retrieval and a tokenizer-matched recent-history control."""

from functools import lru_cache
from time import perf_counter

import numpy as np
import tiktoken

BGE = "BAAI/bge-small-en-v1.5"
TOP_K = 8
PASSAGE_TOKENS = 220
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def encoder():
    return tiktoken.get_encoding("o200k_base")


def proxy_tokens(text):
    return len(encoder().encode_ordinary(text))


def matched_recent(history, filtered, serialized):
    if history == filtered:
        return history
    target = proxy_tokens(serialized(filtered))
    lines = history.splitlines(keepends=True)
    low, high = 0, len(lines)
    best = ""
    while low <= high:
        start = (low + high) // 2
        candidate = "".join(lines[start:])
        if proxy_tokens(serialized(candidate)) <= target:
            best = candidate
            high = start - 1
        else:
            low = start + 1
    return best


def passages(history):
    current, count = [], 0
    for line in history.splitlines(keepends=True):
        if not line.strip():
            continue
        tokens = proxy_tokens(line)
        if current and count + tokens > PASSAGE_TOKENS:
            yield "".join(current)
            current, count = [], 0
        current.append(line)
        count += tokens
    if current:
        yield "".join(current)


def normalize_rows(rows):
    vectors = np.asarray(rows, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class DenseIndex:
    def __init__(self, model, history):
        start = perf_counter()
        self.texts = list(passages(history))
        if not self.texts:
            raise ValueError("Dense retrieval needs at least one nonempty passage.")
        self.model = model
        self.vectors = normalize_rows(list(model.embed(self.texts, batch_size=32)))
        self.build_ms = round((perf_counter() - start) * 1000, 3)

    def search(self, question):
        query = normalize_rows(list(self.model.embed([QUERY_PREFIX + question])))[0]
        scores = self.vectors @ query
        ids = sorted(sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:TOP_K])
        return "\n".join(self.texts[index].rstrip("\r\n") for index in ids), ids
