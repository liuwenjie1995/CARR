"""Analyze complete paired test trajectories and generate CSV, PDF, SVG and case studies."""

import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "results" / "test_plan" / "freeze.json"
TEST_ROOT = ROOT / "results" / "test"
ABLATIONS = ("baseline", "full", "no_dual_layer", "no_conflict_unlearning", "no_validation_gate")
OUTCOMES = ("full_only_correct", "baseline_only_correct", "both_wrong", "both_correct")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path is outside its run directory: {name}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logs_in(folder, limit=None):
    files = sorted(folder.glob("*-trajectory.json"))
    if limit is not None:
        files = files[:limit]
    rows = []
    for path in files:
        row = load(path)
        if not row.get("status") or type(row.get("correct")) is not bool or (row["status"] != "ok" and row["correct"]):
            raise ValueError(f"Unscored or inconsistent trajectory: {path}")
        usage = row.get("usage")
        if usage is not None and (not isinstance(usage, dict) or not all(
            type(usage.get(field)) is int and usage[field] >= 0
            for field in ("prompt_tokens", "completion_tokens")
        )):
            raise ValueError(f"Invalid token usage: {path}")
        if not row.get("trajectory"):
            raise ValueError(f"Empty interaction trace: {path}")
        rows.append({**row, "log_path": path.relative_to(ROOT).as_posix()})
    return rows


def check_rows(rows, sample_ids):
    found = {row["sample_id"] for row in rows}
    if found != set(sample_ids) or len(found) != len(rows):
        raise ValueError("Each cell must contain every locked sample exactly once.")


def load_formal(folder, freeze):
    manifest = load(folder / "run_manifest.json")
    if (manifest.get("split") != "test" or manifest.get("status") != "completed"
            or manifest.get("freeze_sha256") != digest(FREEZE)):
        raise ValueError("Run must be complete and reference the frozen test plan.")
    expected = {(model, variant, seed) for model, variants in freeze["schedule"].items()
                for variant in variants for seed in freeze["execution"]["seeds"]}
    sample_ids = manifest["sample_ids"]
    if len(sample_ids) != freeze["test_size"] or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest does not list the expected number of unique locked test samples.")
    locked_samples = ROOT / "data" / "test" / "samples.jsonl"
    if digest(locked_samples) != freeze["test_file_sha256"]["test/samples.jsonl"]:
        raise ValueError("Locked test sample file changed.")
    locked_ids = {json.loads(line)["id"] for line in locked_samples.read_text(encoding="utf-8").splitlines()}
    if set(sample_ids) != locked_ids:
        raise ValueError("Manifest sample IDs do not match the locked test split.")
    cells = {}
    for cell in manifest["cells"]:
        key = cell["model"], cell["variant"], cell["seed"]
        if key not in expected or key in cells:
            raise ValueError(f"Unexpected or duplicate test cell: {key}")
        rows = logs_in(inside(folder, cell["directory"]))
        check_rows(rows, sample_ids)
        for row in rows:
            for event in row["trajectory"]:
                if event.get("operation") == "chat_completion":
                    if event.get("request", {}).get("model") != key[0]:
                        raise ValueError(f"Requested model does not match manifest: {row['log_path']}")
        cells[key] = rows
    if set(cells) != expected:
        raise ValueError("Some frozen test cells are missing.")
    return cells


def load_dev_smoke(folder, limit):
    summary = load(folder / "summary.json")
    if summary.get("split") != "dev" or summary.get("complete_pairs") != limit:
        raise ValueError("Use a completed five-pair dev run to smoke test analysis.")
    rows = logs_in(folder, limit * 2)
    cells = {}
    for arm, variant in (("baseline", "baseline"), ("method", "full")):
        selected = [row for row in rows if row["arm"] == arm]
        check_rows(selected, summary["sample_ids"])
        cells[(summary["model"], variant, summary["seed"])] = selected
    return cells


def paired(base, candidate):
    base_by_id = {row["sample_id"]: row for row in base}
    pairs = []
    for row in candidate:
        other = base_by_id[row["sample_id"]]
        if any(row[key] != other[key] for key in ("context_id", "source", "reference_answers", "question")):
            raise ValueError(f"Reference or task mismatch for {row['sample_id']}")
        pairs.append((other, row))
    return pairs


def group_bootstrap(pairs, draws, seed):
    groups = defaultdict(lambda: [0, 0])
    for baseline, variant in pairs:
        group = groups[variant["context_id"]]
        group[0] += int(variant["correct"]) - int(baseline["correct"])
        group[1] += 1
    if len(groups) < 2:
        return None, None, len(groups)
    # Count weighted: the score is averaged over questions in the resampled histories.
    population = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = rng.choices(population, k=len(population))
        estimates.append(100 * sum(item[0] for item in sample) / sum(item[1] for item in sample))
    estimates.sort()
    return estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))], len(groups)


def metrics(rows):
    n = len(rows)
    accounted = [row["usage"] for row in rows if row["usage"] is not None]
    input_tokens = sum(row["prompt_tokens"] for row in accounted)
    output_tokens = sum(row["completion_tokens"] for row in accounted)
    reported = len(accounted)
    return {
        "accuracy_pct": 100 * sum(row["correct"] for row in rows) / n,
        "error_pct": 100 * sum(row["status"] != "ok" for row in rows) / n,
        "missing_usage_pct": 100 * (n - reported) / n,
        "prompt_tokens_per_case": input_tokens / reported if reported else None,
        "completion_tokens_per_case": output_tokens / reported if reported else None,
        "total_tokens_per_case": (input_tokens + output_tokens) / reported if reported else None,
        "api_calls_per_case": sum(sum(event["operation"] == "chat_completion" for event in row["trajectory"])
                                  for row in rows) / n,
        "latency_ms_per_case": mean(row["end_to_end_ms"] for row in rows),
    }


def aggregate(cells, stats):
    per_seed, by_history, paired_by_model = [], [], {}
    for (model, variant, seed), rows in sorted(cells.items()):
        baseline = cells[model, "baseline", seed]
        pairs = paired(baseline, rows)
        arm, reference = metrics(rows), metrics(baseline)
        lo, hi, groups = group_bootstrap(pairs, stats["bootstrap_draws"], stats["bootstrap_seed"])
        per_seed.append({
            "model": model, "variant": variant, "seed": seed, "samples": len(rows), "histories": groups,
            **arm, "total_token_reduction_pct": 100 * (
                1 - arm["total_tokens_per_case"] / reference["total_tokens_per_case"]
            ) if reference["total_tokens_per_case"] and arm["total_tokens_per_case"] is not None else None,
            "accuracy_gain_pp": arm["accuracy_pct"] - reference["accuracy_pct"],
            "paired_gain_ci95_low_pp": lo, "paired_gain_ci95_high_pp": hi,
            "variant_only_correct": sum(not b["correct"] and a["correct"] for b, a in pairs),
            "baseline_only_correct": sum(b["correct"] and not a["correct"] for b, a in pairs),
        })
        grouped = defaultdict(list)
        for baseline_row, other in pairs:
            grouped[other["context_id"]].append((baseline_row, other))
        for context_id, items in sorted(grouped.items()):
            by_history.append({
                "model": model, "variant": variant, "seed": seed, "context_id": context_id,
                "samples": len(items), "baseline_correct": sum(x["correct"] for x, _ in items),
                "variant_correct": sum(y["correct"] for _, y in items),
            })
        if variant == "full":
            paired_by_model[model, seed] = pairs

    metric_keys = tuple(metrics(next(iter(cells.values())))) + ("total_token_reduction_pct", "accuracy_gain_pp")
    summary = []
    group_key = lambda row: (row["model"], row["variant"])
    for (model, variant), group in itertools.groupby(sorted(per_seed, key=group_key), key=group_key):
        rows = list(group)
        item = {"model": model, "variant": variant, "seeds": len(rows),
                "samples_per_seed": rows[0]["samples"], "histories": rows[0]["histories"]}
        for metric in metric_keys:
            values = [row[metric] for row in rows if row[metric] is not None]
            item[metric + "_mean"] = mean(values) if values else None
            item[metric + "_sd"] = stdev(values) if len(values) >= 2 else None
        summary.append(item)
    return summary, per_seed, by_history, paired_by_model


def case_studies(paired_by_model, model, seed):
    pairs = paired_by_model[model, seed]
    buckets = defaultdict(list)
    for base, full in sorted(pairs, key=lambda pair: pair[1]["sample_id"]):
        outcome = (("both_correct" if full["correct"] else "baseline_only_correct") if base["correct"]
                   else ("full_only_correct" if full["correct"] else "both_wrong"))
        view = lambda row: {key: row.get(key) for key in (
            "prediction", "correct", "status", "alias_presence", "log_path",
        )}
        buckets[outcome].append({
            "outcome": outcome, "model": model, "seed": seed, "sample_id": full["sample_id"],
            "context_id": full["context_id"], "question": full["question"],
            "reference_answers": full["reference_answers"], "baseline": view(base), "full": view(full),
        })
    first_by_outcome = [items[0] for outcome in OUTCOMES if (items := buckets[outcome])]
    remaining = [item for outcome in OUTCOMES for item in buckets[outcome][1:]]
    return (first_by_outcome + remaining)[:2]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(directory, summary, smoke):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = ("#1261A0", "#9C4221", "#4D8061", "#6C5298")
    files = []

    def save(figure, name):
        for ext in ("pdf", "svg"):
            path = directory / f"{name}.{ext}"
            figure.savefig(path, bbox_inches="tight")
            files.append(path.relative_to(ROOT).as_posix())
        plt.close(figure)

    by_model = defaultdict(dict)
    for row in summary:
        by_model[row["model"]][row["variant"]] = row
    figure, axis = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
    for index, (model, rows) in enumerate(sorted(by_model.items())):
        color = colors[index % len(colors)]
        for variant, marker in (("baseline", "o"), ("full", "D")):
            row = rows[variant]
            if row["total_tokens_per_case_mean"] is not None:
                axis.scatter(row["total_tokens_per_case_mean"], row["accuracy_pct_mean"], color=color,
                             marker=marker, s=70, label=f"{model} / {variant}")
    axis.set(xlabel="Mean input and output tokens per question", ylabel="SubEM accuracy (%)",
             title="Efficiency and accuracy" + (" (dev smoke)" if smoke else ""))
    axis.grid(alpha=.25)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    else:
        axis.text(.5, .5, "No reported token usage", ha="center", transform=axis.transAxes)
    if len({row["accuracy_pct_mean"] for row in summary}) == 1:
        value = summary[0]["accuracy_pct_mean"]
        axis.set_ylim(max(0, value - 5), min(101, value + 5))
    save(figure, "efficiency_accuracy")
    if smoke:
        return files

    flash = by_model["deepseek-flash"]
    figure, axis = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    y = [flash[name]["accuracy_pct_mean"] for name in ("full", "threshold_3", "threshold_4")]
    axis.plot([2, 3, 4], y, marker="o", linewidth=2, color=colors[0])
    axis.set(xlabel="Minimum distinct values for a conflict", ylabel="SubEM accuracy (%)",
             title="Pre-specified conflict threshold sensitivity", xticks=[2, 3, 4], ylim=(0, 105))
    axis.grid(alpha=.25)
    save(figure, "threshold_sensitivity")

    figure, axis = plt.subplots(figsize=(8.1, 4.7), layout="constrained")
    labels = ("Baseline", "Full", "No dual layer", "No conflict handling", "No validation gate")
    bars = axis.bar(labels, [flash[name]["accuracy_pct_mean"] for name in ABLATIONS],
                    color=[colors[1], colors[0], "#77899A", "#77899A", "#77899A"])
    axis.bar_label(bars, fmt="%.1f", padding=4)
    axis.set(ylabel="SubEM accuracy (%)", title="DeepSeek Flash ablations", ylim=(0, 110))
    axis.tick_params(axis="x", rotation=14)
    save(figure, "ablation_accuracy")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--test-run", help="Completed run directory name under results/test.")
    inputs.add_argument("--smoke-dev", help="Five-pair smoke run directory name under results/dev.")
    parser.add_argument("--limit", type=int, choices=[5], help="Apply a limit only to dev smoke analysis.")
    args = parser.parse_args()
    if bool(args.smoke_dev) != (args.limit == 5):
        parser.error("Use --limit 5 with --smoke-dev only.")
    freeze = load(FREEZE)
    smoke = bool(args.smoke_dev)
    if smoke:
        folder = inside(ROOT / "results" / "dev", args.smoke_dev)
        cells = load_dev_smoke(folder, args.limit)
        output = folder / "analysis_smoke"
        figures = output / "figures"
    else:
        folder = inside(TEST_ROOT, args.test_run)
        cells = load_formal(folder, freeze)
        output = folder / "analysis"
        figures = ROOT / "paper" / "figures" / folder.name
    summary, seeds, histories, pairs = aggregate(cells, freeze["statistics"])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_statistics.csv", seeds)
    write_csv(output / "by_history.csv", histories)
    cases = case_studies(pairs, "deepseek-flash", 42)
    (output / "case_studies.json").write_text(json.dumps({
        "split": "dev" if smoke else "test", "cases": cases,
        "note": "Cases retain original outputs. Agreements may be shown if no disagreements exist.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assets = write_figures(figures, summary, smoke)
    (output / "provenance.json").write_text(json.dumps({
        "runner": "src/analyze_results.py", "runner_sha256": digest(Path(__file__)),
        "run": folder.relative_to(ROOT).as_posix(), "figures": assets, "freeze_sha256": digest(FREEZE),
        "variance_note": "Across-seed SD is blank for one seed. Intervals resample histories, not individual questions.",
        "interval_note": "With three histories the interval is descriptive and sensitive to individual histories.",
        "missing_data_note": "Attempted API errors count as failures. Token means use reported usage only.",
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
