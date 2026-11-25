"""Create the locked LongMemEval-S* dev/test partition from Accurate_Retrieval.parquet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prepare_data import DATA, REVISION, json_bytes, question_key
from src.v2_data import EVENT_SHA

LME_SOURCE = "longmemeval_s*"
LME_DIR = DATA / "longmemeval_v2"


def sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def read_longmemeval_locked(split: str) -> tuple[list[dict], dict[str, str]]:
    lock = json.loads((LME_DIR / "split-lock.json").read_bytes())
    if not lock["test_locked"] or any(lock["cross_split_intersections"].values()):
        raise ValueError("Invalid split lock for longmemeval_v2.")
    buffers = {}
    for filename in ("contexts.json", "samples.jsonl"):
        relative = f"{split}/{filename}"
        data = (LME_DIR / relative).read_bytes()
        if sha(data) != lock["files"][relative]:
            raise ValueError(f"Frozen LongMemEval file changed: {relative}")
        buffers[filename] = data
    return (
        [json.loads(line) for line in buffers["samples.jsonl"].splitlines()],
        json.loads(buffers["contexts.json"]),
    )


def make_longmemeval(source: Path, limit: int | None = None) -> dict:
    import pyarrow.parquet as pq

    if sha(source.read_bytes()) != EVENT_SHA:
        raise ValueError("Accurate_Retrieval.parquet does not match its pinned SHA-256.")

    histories: dict[str, tuple[str, list[dict]]] = {}
    for row in pq.read_table(source, columns=["context", "questions", "answers", "metadata"]).to_pylist():
        meta = row["metadata"]
        if meta["source"] != LME_SOURCE:
            continue
        context = row["context"]
        key = sha(context.encode("utf-8"))
        if key in histories:
            raise ValueError("Duplicate LongMemEval context.")
        q_types = meta["question_types"]
        q_dates = meta["question_dates"]
        qa_ids = meta["qa_pair_ids"]
        q_ids = meta["question_ids"]
        samples = []
        for i, (question, aliases, q_type, q_date, qa_id, q_id) in enumerate(
            zip(row["questions"], row["answers"], q_types, q_dates, qa_ids, q_ids, strict=True)
        ):
            if not question_key(question) or not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
                raise ValueError("Invalid LongMemEval question or reference.")
            samples.append({
                "id": f"longmemeval_s-{key[:12]}-{i:03d}",
                "upstream_id": qa_id,
                "upstream_question_id": q_id,
                "source": LME_SOURCE,
                "question_type": q_type,
                "question_date": q_date,
                "context_id": key,
                "question": question,
                "answers": aliases,
            })
        histories[key] = (context, samples)

    if len(histories) != 5:
        raise ValueError(f"Expected 5 LongMemEval histories, found {len(histories)}.")

    ordered_keys = sorted(histories)
    dev_key, test_keys = ordered_keys[0], ordered_keys[1:]
    dev = histories[dev_key][1]
    dev_questions = {question_key(row["question"]) for row in dev}
    test = [
        row
        for k in test_keys
        for row in histories[k][1]
        if question_key(row["question"]) not in dev_questions
    ]
    if limit is not None and len(dev[:limit]) != limit:
        raise ValueError("Insufficient dev rows for smoke limit.")

    intersections = {
        "sample_ids": len({x["id"] for x in dev} & {x["id"] for x in test}),
        "context_hashes": len({x["context_id"] for x in dev} & {x["context_id"] for x in test}),
        "normalized_questions": len(dev_questions & {question_key(row["question"]) for row in test}),
    }
    if any(intersections.values()):
        raise ValueError("LongMemEval dev/test overlap detected.")

    outputs, partitions = {}, {}
    for name, rows in (("dev", dev), ("test", test)):
        keys = sorted({row["context_id"] for row in rows})
        outputs[f"{name}/contexts.json"] = json_bytes({k: histories[k][0] for k in keys})
        outputs[f"{name}/samples.jsonl"] = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
        partitions[name] = {
            "samples": len(rows),
            "unique_contexts": len(keys),
            "by_question_type": dict(Counter(row["question_type"] for row in rows)),
        }

    lock = {
        "revision": REVISION,
        "source_sha256": EVENT_SHA,
        "test_locked": True,
        "partitions": partitions,
        "cross_split_intersections": intersections,
        "partition_rule": "Lowest SHA-256 history is dev (1 history, 60Q). Remaining 4 histories are test, excluding repeated dev questions.",
        "files": {name: sha(content) for name, content in outputs.items()},
    }
    outputs["split-lock.json"] = json_bytes(lock)
    for name, data in outputs.items():
        path = LME_DIR / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to mutate existing lock/data: {path}")
    for name, data in outputs.items():
        path = LME_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--source", type=Path, default=DATA / "source" / "Accurate_Retrieval.parquet")
    args = parser.parse_args()
    print(json.dumps(make_longmemeval(args.source, args.limit), indent=2))
