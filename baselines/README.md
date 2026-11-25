# 第一阶段基线代码

在此目录保留 MemoryAgentBench 官方仓库的独立检出，固定提交为
`538026089d1a8a8eff05121d0db89b388f360eba`。仓库许可证为 MIT。

| 入口 | 用途 |
|---|---|
| `MemoryAgentBench/agent.py`、`initialization.py` | 官方 Long-Context Agent 的入库与问答流程 |
| `MemoryAgentBench/utils/templates.py` | FactConsolidation 的系统消息、入库模板和查询模板 |
| `MemoryAgentBench/utils/eval_other_utils.py` | 官方句子分块及归一化子串匹配评分 |
| `MemoryAgentBench/configs/data_conf/Conflict_Resolution/` | 单跳、多跳与不同历史长度的实验配置 |
| `MemoryAgentBench/methods/` | 其他 RAG 与记忆基线实现，按对应配置单独安装依赖 |
| `long_context.py` | 上游全上下文基线的 DeepSeek 适配入口，由统一评测器调用 |

适配器复用上游模板、句子分块函数、答案解析和评分函数。每条 query 都从同一份分块历史出发，上游的记忆阶段为本地拼接，不计作 API 调用。与原论文运行相比，本轮使用 DeepSeek Flash、固定模板时间标记和本项目的开发集子集。上游代码的 GPT-4o-mini 分词器只在本地切分文本，不会调用 GPT。完整官方论文表格仍需单独复现。

官方仓库 [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench/tree/538026089d1a8a8eff05121d0db89b388f360eba)。
