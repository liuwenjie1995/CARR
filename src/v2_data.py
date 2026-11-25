"""Create the separate EventQA lock and read verified research splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.prepare_data import DATA, REVISION, json_bytes, question_key

EVENT_SHA = "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1"
EVENT_SOURCE = "eventqa_65536"
EVENT_DIR = DATA / "eventqa_v2"


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def read_locked(dataset, split):
    folder = DATA if dataset == "factcon" else EVENT_DIR
    lock = json.loads((folder / "split-lock.json").read_bytes())
    if not lock["test_locked"] or any(lock["cross_split_intersections"].values()):
        raise ValueError(f"Invalid split lock for {dataset}.")
    buffers = {}
    for filename in ("contexts.json", "samples.jsonl"):
        relative = f"{split}/{filename}"
        data = (folder / relative).read_bytes()
        if sha(data) != lock["files"][relative]:
            raise ValueError(f"Frozen file changed: {relative}")
        buffers[filename] = data
    return [json.loads(line) for line in buffers["samples.jsonl"].splitlines()], json.loads(buffers["contexts.json"])


def make_eventqa(source, limit):
    import pyarrow.parquet as pq

    if sha(source.read_bytes()) != EVENT_SHA:
        raise ValueError("Accurate Retrieval parquet does not match its fixed revision.")
    histories = {}
    for row in pq.read_table(source, columns=["context", "questions", "answers", "metadata"]).to_pylist():
        metadata = row["metadata"]
        if metadata["source"] != EVENT_SOURCE:
            continue
        context = row["context"]
        key = sha(context.encode("utf-8"))
        if key in histories:
            raise ValueError("Duplicate EventQA context.")
        samples = []
        for i, (question, aliases) in enumerate(zip(row["questions"], row["answers"], strict=True)):
            if not question_key(question) or not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
                raise ValueError("Invalid EventQA question or reference.")
            samples.append({
                "id": f"{EVENT_SOURCE}-{key[:16]}-{i:03d}", "upstream_id": metadata["qa_pair_ids"][i],
                "source": EVENT_SOURCE, "context_id": key, "question": question, "answers": aliases,
            })
        histories[key] = (context, samples)
    if len(histories) < 2:
        raise ValueError("EventQA must have separate histories for dev and test.")
    first, *others = sorted(histories)
    dev = histories[first][1]
    dev_questions = {question_key(row["question"]) for row in dev}
    test = [row for key in others for row in histories[key][1] if question_key(row["question"]) not in dev_questions]
    if len(dev[:limit]) != limit or not test:
        raise ValueError("EventQA preparation smoke or heldout partition is empty.")
    intersections = {
        "sample_ids": len({x["id"] for x in dev} & {x["id"] for x in test}),
        "context_hashes": len({x["context_id"] for x in dev} & {x["context_id"] for x in test}),
        "normalized_questions": len(dev_questions & {question_key(row["question"]) for row in test}),
    }
    if any(intersections.values()):
        raise ValueError("EventQA dev/test overlap.")
    outputs, partitions = {}, {}
    for name, rows in (("dev", dev), ("test", test)):
        keys = sorted({row["context_id"] for row in rows})
        outputs[f"{name}/contexts.json"] = json_bytes({key: histories[key][0] for key in keys})
        outputs[f"{name}/samples.jsonl"] = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
        partitions[name] = {
            "samples": len(rows), "unique_contexts": len(keys), "by_source": dict(Counter(row["source"] for row in rows)),
        }
    lock = {"revision": REVISION, "source_sha256": EVENT_SHA, "test_locked": True, "partitions": partitions,
            "cross_split_intersections": intersections,
            "partition_rule": "Lowest SHA-256 history is dev. Remaining histories are test; exclude repeated dev questions.",
            "files": {name: sha(content) for name, content in outputs.items()}}
    outputs["split-lock.json"] = json_bytes(lock)
    for name, data in outputs.items():
        path = EVENT_DIR / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to change existing lock or data: {path}")
    for name, data in outputs.items():
        path = EVENT_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(data)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5], required=True)
    parser.add_argument("--source", type=Path, default=DATA / "source" / "Accurate_Retrieval.parquet")
    args = parser.parse_args()
    print(json.dumps(make_eventqa(args.source, args.limit), indent=2))
