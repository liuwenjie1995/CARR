"""Controlled external baselines sharing identical embedder and answering backbone."""

from src.baselines_v2.bm25_recency import retrieve_bm25_recency
from src.baselines_v2.hipporag_ppr import GraphFact, HippoGraphIndex
from src.baselines_v2.ircot_baseline import run_ircot
from src.baselines_v2.mem0_style import Mem0Store, consolidate_mem0_history

__all__ = [
    "GraphFact",
    "HippoGraphIndex",
    "Mem0Store",
    "consolidate_mem0_history",
    "retrieve_bm25_recency",
    "run_ircot",
]
