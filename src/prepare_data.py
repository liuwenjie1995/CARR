"""Build frozen local FactConsolidation manifests from the pinned public data."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
SOURCE_SHA256 = "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"
SOURCES = {
    f"factconsolidation_{hop}_{length}"
    for hop in ("sh", "mh")
    for length in ("6k", "32k", "64k", "262k")
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def question_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized))


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def split_records(source_path: Path) -> tuple[dict, dict]:
    if digest(source_path.read_bytes()) != SOURCE_SHA256:
        raise ValueError("Public dataset checksum does not match the pinned revision.")
    rows = pq.read_table(source_path).to_pylist()
    if {row["metadata"]["source"] for row in rows} != SOURCES or len(rows) != len(SOURCES):
        raise ValueError("Unexpected FactConsolidation source groups.")
    records = []
    contexts = {"dev": {}, "test": {}}
    for row in rows:
        context = row["context"]
        context_id = digest(context.encode("utf-8"))
        source = row["metadata"]["source"]
        split = "dev" if source.endswith("_6k") else "test"
        contexts[split][context_id] = context
        for index, (question, answers, sample_id) in enumerate(
            zip(row["questions"], row["answers"], row["metadata"]["qa_pair_ids"], strict=True)
        ):
            if not question_key(question) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                raise ValueError("Invalid question or answer aliases.")
            records.append((split, index, {
                "id": sample_id, "source": source, "context_id": context_id,
                "question": question, "answers": answers,
            }))

    dev_keys = {question_key(rec["question"]) for split, _, rec in records if split == "dev"}
    # The earlier exploratory work previewed the first question of each longer source.
    preview_keys = {question_key(rec["question"]) for split, i, rec in records if split == "test" and i == 0}
    samples = {"dev": [], "test": []}
    excluded = Counter()
    for split, _, record in records:
        key = question_key(record["question"])
        if split == "test" and (key in dev_keys or key in preview_keys):
            excluded["also_in_dev" if key in dev_keys else "previous_preview_or_duplicate"] += 1
            continue
        samples[split].append(record)

    dev = samples["dev"]
    test = samples["test"]
    all_ids = [rec["id"] for rec in dev + test]
    intersections = {
        "sample_ids": len({rec["id"] for rec in dev} & {rec["id"] for rec in test}),
        "context_hashes": len(contexts["dev"].keys() & contexts["test"].keys()),
        "normalized_questions": len(dev_keys & {question_key(rec["question"]) for rec in test}),
    }
    if not dev or not test or len(all_ids) != len(set(all_ids)) or any(intersections.values()):
        raise ValueError("Split isolation validation failed.")

    files = {}
    statistics = {}
    for split in ("dev", "test"):
        used_ids = {rec["context_id"] for rec in samples[split]}
        members = sorted(samples[split], key=lambda rec: rec["id"])
        files[f"{split}/samples.jsonl"] = "".join(
            json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n" for rec in members
        ).encode("utf-8")
        files[f"{split}/contexts.json"] = json_bytes({key: contexts[split][key] for key in used_ids})
        statistics[split] = {
            "samples": len(members), "unique_contexts": len(used_ids),
            "by_source": dict(Counter(rec["source"] for rec in members)),
        }

    lock = {
        "format": 1, "dataset": "ai-hyz/MemoryAgentBench", "revision": REVISION,
        "source_sha256": SOURCE_SHA256, "partitions": statistics,
        "excluded_test_samples": dict(excluded), "cross_split_intersections": intersections,
        "files": {path: digest(contents) for path, contents in files.items()},
        "test_locked": True,
    }
    files["split-lock.json"] = json_bytes(lock)
    return files, lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=[5], required=True, help="Number of dev rows to validate in the local smoke check.")
    parser.add_argument("--source", type=Path, default=DATA / "source" / "Conflict_Resolution.parquet")
    args = parser.parse_args()
    files, lock = split_records(args.source)
    dev_sample = [json.loads(line) for line in files["dev/samples.jsonl"].splitlines()[:args.limit]]
    dev_contexts = json.loads(files["dev/contexts.json"])
    if len(dev_sample) != args.limit or not all(item["context_id"] in dev_contexts for item in dev_sample):
        raise SystemExit("Could not validate five dev rows.")
    for relative, content in files.items():
        path = DATA / relative
        if path.exists() and path.read_bytes() != content:
            raise SystemExit(f"Frozen data differs: {path}")
    for relative, content in files.items():
        path = DATA / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
    print(json.dumps({**lock, "dev_rows_smoke_checked": len(dev_sample)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
