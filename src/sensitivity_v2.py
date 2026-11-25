"""Zero-API-cost offline sensitivity and tombstone displacement benchmarks for CARR v2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.carr_v2.open_parser import archive_v2, parse_numbered_v2
from src.carr_v2.tombstone_index import (
    BM25Index,
    HybridLexicalDenseIndex,
    TombstoneDenseIndex,
    chunk_with_line_ids,
)
from src.evaluator import dump_json
from src.factual_v2 import norm, records_for
from src.hop_retrieval import BRIDGES, MAX_FACTS
from src.offline_diagnostics import FORMAL_RUN, has_alias
from src.run_v2 import BGE_FOLDER
from src.v2_data import read_locked
from src.v2_report import load, write_csv
from src.v2_retrieval import BGE, normalize_rows, proxy_tokens

OUT_DIR = ROOT / "results/analysis/v2_sensitivity"
CHUNK_GRID = (128, 220, 512)
TOP_K_GRID = (4, 8, 16)
HOP_GRID = (1, 2, 3, 4)


def _eval_context_recall(samples: list[dict], context_fn) -> dict:
    sh_hit = sh_n = mh_hit = mh_n = 0
    toks = []
    hits_by_id = {}
    for row in samples:
        ctx = context_fn(row)
        toks.append(proxy_tokens(ctx))
        ok = has_alias(ctx, row["answers"])
        hits_by_id[row["id"]] = ok
        if "_sh_" in row["id"]:
            sh_n += 1
            sh_hit += int(ok)
        else:
            mh_n += 1
            mh_hit += int(ok)
    tot = sh_n + mh_n
    return {
        "all_recall": sh_hit + mh_hit,
        "all_total": tot,
        "all_recall_pct": round(100 * (sh_hit + mh_hit) / tot, 2),
        "sh_recall": sh_hit,
        "sh_total": sh_n,
        "sh_recall_pct": round(100 * sh_hit / sh_n, 2),
        "mh_recall": mh_hit,
        "mh_total": mh_n,
        "mh_recall_pct": round(100 * mh_hit / mh_n, 2),
        "mean_context_tokens": round(sum(toks) / len(toks), 1),
        "hits_by_id": hits_by_id,
    }


def run_sensitivity(out_dir: Path = OUT_DIR) -> dict:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(
        BGE, cache_dir=str(ROOT / ".venv/bge_cache"), threads=2,
        local_files_only=True, specific_model_path=str(BGE_FOLDER),
    )
    samples, histories = read_locked("factcon", "test")
    by_ctx = defaultdict(list)
    for row in samples:
        by_ctx[row["context_id"]].append(row)

    # Precompute archive_v2 decisions & parser coverage per history
    archive_info = {}
    coverage_rows = []
    for cid, text in sorted(histories.items()):
        recs, clean, decisions = archive_v2(text, "numbered")
        parsed_v2 = sum(bool(parse_numbered_v2(r)) for r in recs.values())
        tombstones = {d["removed_line"] for d in decisions}
        archive_info[cid] = (clean, tombstones)
        coverage_rows.append({
            "context_id": cid,
            "numbered_facts": len(recs),
            "parsed_v2": parsed_v2,
            "parsed_v2_pct": round(100 * parsed_v2 / len(recs), 2),
            "archived_v2": len(decisions),
        })

    # Precompute question query embeddings once
    q_texts = [row["question"] for row in samples]
    # Re-use dense index rank_scores via query vectors directly? DenseIndex uses QUERY_PREFIX,
    # so calling index.rank_scores(q) directly for each chunk size is cleanest and still fast.

    grid_rows = []
    displacement_summary = []

    for chunk_size in CHUNK_GRID:
        # Build raw & repacked-clean indices per history once for this chunk_size
        raw_dense = {}
        raw_hybrid = {}
        tomb_dense = {}
        tomb_hybrid = {}
        repack_dense = {}

        for cid, text in histories.items():
            clean_text, tombstones = archive_info[cid]
            hd = HybridLexicalDenseIndex(embedder, text, tombstones=None, chunk_tokens=chunk_size)
            raw_hybrid[cid] = hd
            raw_dense[cid] = TombstoneDenseIndex(
                embedder, text, tombstones=None, chunk_tokens=chunk_size,
                _chunks=hd.chunks, _vectors=hd.vectors,
            )
            tomb_dense[cid] = raw_dense[cid].with_tombstones(tombstones)
            tomb_hybrid[cid] = hd.with_tombstones(tombstones)
            repack_dense[cid] = TombstoneDenseIndex(
                embedder, clean_text, tombstones=None, chunk_tokens=chunk_size,
            )

        # Cache ranked chunk scores per (cid, method_type, sample_id) across top_k values
        raw_scores_cache = {}
        repack_scores_cache = {}
        hybrid_order_cache = {}

        for row in samples:
            cid = row["context_id"]
            sid = row["id"]
            q = row["question"]
            ds = raw_dense[cid].rank_scores(q)
            raw_scores_cache[sid] = sorted(range(len(ds)), key=lambda i: (-float(ds[i]), i))
            rs = repack_dense[cid].rank_scores(q)
            repack_scores_cache[sid] = sorted(range(len(rs)), key=lambda i: (-float(rs[i]), i))

            ls = raw_hybrid[cid].bm25.scores(q)
            d_rank = {d: r + 1 for r, d in enumerate(raw_scores_cache[sid])}
            l_order = sorted(range(len(ls)), key=lambda i: (-float(ls[i]), i))
            l_rank = {d: r + 1 for r, d in enumerate(l_order)}
            rrf = {
                d: (1.0 / (60 + d_rank[d])) + (1.0 / (60 + l_rank[d]) if ls[d] > 0 else 0.0)
                for d in range(len(ds))
            }
            hybrid_order_cache[sid] = sorted(range(len(ds)), key=lambda i: (-rrf[i], -float(ds[i]), i))

        for top_k in TOP_K_GRID:
            res_naive = _eval_context_recall(
                samples, lambda r: raw_dense[r["context_id"]]._format_masked(raw_scores_cache[r["id"]][:top_k])[0]
            )
            res_repack = _eval_context_recall(
                samples, lambda r: repack_dense[r["context_id"]]._format_masked(repack_scores_cache[r["id"]][:top_k])[0]
            )
            res_tomb = _eval_context_recall(
                samples, lambda r: tomb_dense[r["context_id"]]._format_masked(raw_scores_cache[r["id"]][:top_k])[0]
            )
            res_hybrid = _eval_context_recall(
                samples, lambda r: tomb_hybrid[r["context_id"]]._format_masked(hybrid_order_cache[r["id"]][:top_k])[0]
            )

            # Quantify displacement: naive hit == True, archived hit == False (while gold exists in clean_text!)
            repack_displaced = sum(
                res_naive["hits_by_id"][r["id"]] and not res_repack["hits_by_id"][r["id"]]
                for r in samples
            )
            repack_gained = sum(
                (not res_naive["hits_by_id"][r["id"]]) and res_repack["hits_by_id"][r["id"]]
                for r in samples
            )
            tomb_displaced = sum(
                res_naive["hits_by_id"][r["id"]] and not res_tomb["hits_by_id"][r["id"]]
                for r in samples
            )
            tomb_gained = sum(
                (not res_naive["hits_by_id"][r["id"]]) and res_tomb["hits_by_id"][r["id"]]
                for r in samples
            )

            displacement_summary.append({
                "chunk_size": chunk_size,
                "top_k": top_k,
                "repack_displaced_gold": repack_displaced,
                "repack_gained_gold": repack_gained,
                "tombstone_displaced_gold": tomb_displaced,
                "tombstone_gained_gold": tomb_gained,
            })

            for method_name, res in [
                ("naive_bge", res_naive),
                ("repack_archive_bge", res_repack),
                ("tombstone_archive_bge", res_tomb),
                ("tombstone_bm25_bge_rrf", res_hybrid),
            ]:
                grid_rows.append({
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "method": method_name,
                    **{k: v for k, v in res.items() if k != "hits_by_id"},
                })

    # Max-hops sweep offline over the 130 frozen LLM plans using v2 exact index
    v2_idx_by_ctx = {}
    v2_lines_by_ctx = {}
    for cid, text in histories.items():
        recs, _ = records_for(text, "numbered")
        lines = text.splitlines()
        v2_lines_by_ctx[cid] = {r.order: lines[r.index] for r in recs.values()}
        idx = defaultdict(list)
        for r in recs.values():
            if f := parse_numbered_v2(r):
                idx[f.key].append(f)
        v2_idx_by_ctx[cid] = idx

    mh_samples = [r for r in samples if "_mh_" in r["id"]]
    hop_rows = []
    for hop_cap in HOP_GRID:
        completed = 0
        gold_hit = 0
        for row in mh_samples:
            trace = load(FORMAL_RUN / "traces" / f"{row['id']}.json")
            plan = trace.get("plan")
            if not plan:
                continue
            cid = row["context_id"]
            entity = plan["start"]
            queue = [(rel, True) for rel in plan["relations"]]
            proof, visited = [], set()
            ok = True
            while queue:
                rel, allow_bridge = queue.pop(0)
                cands = v2_idx_by_ctx[cid].get((norm(entity), rel), [])
                if not cands:
                    alt = norm(entity).removeprefix("the ")
                    cands = v2_idx_by_ctx[cid].get((alt, rel), []) or v2_idx_by_ctx[cid].get(("the " + alt, rel), [])
                if cands:
                    chosen = max(cands, key=lambda f: f.order)
                    if len(proof) >= hop_cap or chosen.order in visited:
                        ok = False
                        break
                    visited.add(chosen.order)
                    proof.append(chosen.order)
                    entity = chosen.value
                elif allow_bridge and rel in BRIDGES and len(proof) + len(queue) + len(BRIDGES[rel]) <= hop_cap:
                    queue = [(p, False) for p in BRIDGES[rel]] + queue
                else:
                    ok = False
                    break
            if ok and proof:
                completed += 1
                ev = "\n".join(v2_lines_by_ctx[cid][o] for o in proof)
                if has_alias(ev, row["answers"]):
                    gold_hit += 1
        hop_rows.append({
            "max_hops": hop_cap,
            "mh_total": len(mh_samples),
            "completed_chains": completed,
            "gold_in_evidence": gold_hit,
            "gold_recall_pct": round(100 * gold_hit / len(mh_samples), 2),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generator": "src/sensitivity_v2.py",
        "parser_v2_coverage": coverage_rows,
        "displacement_summary": displacement_summary,
        "chunk_topk_grid": grid_rows,
        "max_hops_sweep": hop_rows,
    }
    dump_json(out_dir / "sensitivity_summary.json", summary)
    write_csv(out_dir / "chunk_topk_grid.csv", grid_rows)
    write_csv(out_dir / "displacement_summary.csv", displacement_summary)
    write_csv(out_dir / "max_hops_sweep.csv", hop_rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    res = run_sensitivity(args.out)
    print(json.dumps({
        "parser_v2_coverage": res["parser_v2_coverage"],
        "displacement_summary": res["displacement_summary"],
        "max_hops_sweep": res["max_hops_sweep"],
    }, indent=2))
