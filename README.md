# CARR: Conflict-Aware Recency Relay

[English](#english) | [中文说明](#中文说明)

---

<a name="english"></a>

## English

Official core experimental implementation for the paper:

> **Multi-Hop Retrieval over Evolving Agent Memory via Path Planning and Recency Lookup**  
> Wenjie Liu\*, Li Zhang, Yanru Xue, Longgang Zhao  
> Beijing, China

### Overview

**CARR** (*Conflict-Aware Recency Relay*) is a lightweight retrieval framework for long-horizon LLM agents operating over evolving histories with conflicting attribute updates. Instead of physically deleting superseded lines before chunking (which shifts chunk boundaries) or issuing per-hop LLM calls:

1. **Non-Destructive Offline Partitioning:** Parses semi-structured logs (`open_parser.py`) or multi-session dialogues into a versioned `(Norm(e), r)` fact ledger sorted by descending recency, flags superseded functional entries in an auxiliary tombstone set $\mathcal{T}$, and preserves frozen chunk boundaries (`tombstone_index.py`).
2. **Single-Pass Relation Path Planning:** One compact JSON planning call (`planner_messages_v2` in `semantic_router.py`) decomposes each query into an initial entity and up to 4 ordered relation steps (`{"mode", "start", "steps"}`).
3. **Deterministic Descending-Recency Relay:** Walks each step locally via a three-channel cascade—exact key-recency lookup (Channel 1, $O(L)$ expected lookup complexity), semantic relation matching (Channel 2), and tombstone-masked BM25 + BGE RRF fallback (Channel 3)—with exception guards and whole-query fallback on invalid/empty plans.

### Repository Structure

```text
.
├── README.md
├── requirements.txt
├── baselines/
│   ├── README.md
│   └── long_context.py
├── src/
│   ├── carr_v2/
│   │   ├── open_parser.py            # Open-relation template parser & dialogue extractor
│   │   ├── semantic_router.py        # Single-pass planner & 3-channel recency relay
│   │   └── tombstone_index.py        # Post-retrieval tombstone-masked BM25 + BGE RRF
│   ├── baselines_v2/
│   │   ├── bm25_recency.py           # Hybrid BM25 + BGE + recency baseline
│   │   ├── chain_aware_car.py        # Controlled CAR-style per-hop LLM baseline
│   │   ├── hipporag_ppr.py           # HippoRAG-style PPR & Tombstone+PPR
│   │   ├── ircot_baseline.py         # IRCoT-BGE iterative retrieval baseline
│   │   └── mem0_style.py             # Write-time ADD/UPDATE/DELETE/NOOP baseline
│   ├── run_carr_v2.py                # FactConsolidation main runner
│   ├── run_longmemeval.py            # LongMemEval-S* runner
│   ├── run_ablation_matrix_v2.py     # Layerwise A-G ablation runner
│   ├── run_car_baseline_v2.py        # CAR-style comparison runner
│   ├── run_hipporag_v2.py            # HippoRAG PPR / Tombstone+PPR runner
│   ├── run_graph_regime_sweep.py     # Graph & distractor scaling experiment
│   ├── run_paraphrase_stress_v2_1.py # Paraphrase & lexicographic valid-time stress test
│   ├── run_frozen_holdout_v2.py      # LongMemEval-S* dev & unseen-relation holdout
│   ├── audit_revision_v2.py          # TCAP / latency / offline audit script
│   └── eval_longmemeval_judge*.py    # Official-template & cross-family LLM-as-Judge
└── tests/
    ├── test_carr_v2.py
    ├── test_baselines_v2.py
    ├── test_valid_time_lexicographic.py
    └── ...
```

*(Large benchmark datasets, raw per-query API caches, and compiled PDF drafts are excluded via `.gitignore`.)*

### Headline Results

- **FactConsolidation (562Q, 32K–262K tokens):** CARR achieves **79.72%** overall accuracy (`448/562`), **63.41%** multi-hop accuracy (`175/276`), and **79.36%** TCAP, outperforming official full context (`43.42%` / `6.16%`), `HippoRAG Tombstone+PPR` (`54.63%`), and same-backbone `CAR-style` (`74.02%`) with **40.3% fewer LLM calls** and **32.8% lower median E2E API latency**. The minimal planning-and-lookup core (Group D, `78.47%`) retains **98.4%** (`441/448 = 98.44%`) of the full system's overall accuracy.
- **LongMemEval-S\* (240Q):** CARR improves objective factual accuracy (`factual_5cat`, 216Q) from `69.91%` (full context) to **77.78%** with **208.5x** answer prompt compression, while `HippoRAG Tombstone+PPR` (`76.25%` overall) and CARR (`74.58%` overall, $p=0.541$) perform comparably on sparse conversational graphs.

### Quick Start

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Configure LLM API keys via environment variables before running online experiments
# (never commit API keys to git)

# Run unit tests
pytest -q tests/test_carr_v2.py tests/test_baselines_v2.py tests/test_valid_time_lexicographic.py
```

---

<a name="中文说明"></a>

## 中文说明

论文官方核心实验代码仓库：

> **基于路径规划与新近性查表的动态智能体记忆多跳检索方法**  
> （*Multi-Hop Retrieval over Evolving Agent Memory via Path Planning and Recency Lookup*）  
> 刘文杰\*、章莉、薛燕茹、赵龙刚  
> 北京，中国

### 方法简介

**CARR**（*Conflict-Aware Recency Relay*，冲突感知新近性接力）面向持续演化的长程智能体记忆提供轻量级多跳检索框架，避免索引前物理删除导致的文本块边界偏移，以及逐跳大模型调用带来的高昂时延与费用：

1. **非破坏性离线分区：** 通过开放关系模板解析器（`open_parser.py`）与对话抽取器构建按新近性降序排列的版本化账本，将过期条目记录于辅助墓碑集合 $\mathcal{T}$，同时保持冻结分块边界不变（`tombstone_index.py`）。
2. **单次关系路径规划：** 仅用一次紧凑 JSON 规划调用（`semantic_router.py` 中的 `planner_messages_v2`）输出起始实体与至多 4 步有序关系序列（`{"mode", "start", "steps"}`）。
3. **确定性时间降序接力：** 依次通过通道 1 精确键查表（精确路径核期望复杂度 $O(L)$）、通道 2 语义关系匹配与通道 3 墓碑屏蔽 BM25+BGE 混合回退逐跳解析桥接实体，并在规划无效或空路径时安全回退至全问句混合检索。

### 核心目录

- `src/carr_v2/`：CARR 核心模块（解析器、单次规划器、三通道新近性接力与墓碑屏蔽索引）。
- `src/baselines_v2/`：同底座对照基线（`BM25+BGE+Recency`、`CAR-style`、`HippoRAG PPR / Tombstone+PPR`、`IRCoT-BGE`、`Mem0-style`）。
- `src/run_*.py`：主实验、消融矩阵、图规模缩放、压力测试与留出集运行脚本。
- `tests/`：核心逻辑与回归验证单元测试。

*(数据集、海量逐题原始日志缓存及论文编译中间产物已通过 `.gitignore` 排除，不随代码库上传；切勿在代码或仓库中提交任何 API Key。)*

### 快速运行

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 运行线上实验前请通过环境变量设置 API Key（切勿写入仓库）

# 运行核心单元测试
pytest -q tests/test_carr_v2.py tests/test_baselines_v2.py tests/test_valid_time_lexicographic.py
```
