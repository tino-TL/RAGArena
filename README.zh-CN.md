# RAGArena

> 面向复杂 PDF 问答的 Agentic RAG 系统，覆盖结构化 chunk、多路混合检索、LangGraph 工作流与可复现实验评测。

简体中文 | [English](README.md)

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-1C3C3C?style=flat-square)
![Docling](https://img.shields.io/badge/Docling-PDF_Parsing-374151?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Qwen3.5--4B-111827?style=flat-square)
![Qwen3 Embedding](https://img.shields.io/badge/Qwen3--Embedding--4B-Embeddings-7C3AED?style=flat-square)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-Hybrid_Search-005571?style=flat-square&logo=elasticsearch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Metadata-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=flat-square&logo=redis&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-blue?style=flat-square)

RAGArena 是一个本地优先的 Agentic RAG 检索问答系统，面向论文、企业知识库、技术手册等复杂长 PDF。项目重点解决 vanilla RAG 在复杂问题、长上下文召回、图表/表格证据、查询改写、重排和引用级答案溯源上不稳定的问题。

## 核心能力

- **Agentic Chunk Pipeline**：使用 Ollama 本地 `qwen3.5:4b` 对 Docling 解析后的文档 block 做语义边界规划。
- **body / visual / fused 三层 chunk**：正文、表格 Markdown、图表 caption、邻近正文和检索增强 fused chunk 独立存储与索引。
- **混合检索链路**：Elasticsearch BM25 + Qwen3-Embedding-4B 向量检索 + 可选 HyDE + RRF 融合。
- **精排能力**：可选 `BAAI/bge-reranker-v2-m3`，并限制候选数和输入长度以控制本地延迟。
- **LangGraph 工作流**：包含 guardrail、router、retrieve、rerank、grade、rewrite、generate 和 give-up 路径。
- **评测框架**：支持固定切片 vs agentic 切片 ablation、图表/表格子集评测、检索链路对比、rerank 抽样评测、延迟、Recall@k、MRR@10、NDCG@10。

## 系统架构

### 总体架构

![RAGArena 系统架构](docs/images/ragarena-system-architecture.png)

### Agent 工作流

![RAGArena LangGraph 工作流](docs/images/ragarena-agent-workflow.png)

在线检索问答主流程：

```text
query
 -> guardrail
 -> router
 -> hybrid_retrieve
 -> rerank
 -> grade_documents
 -> generate_answer | rewrite_query | give_up
```

## 当前评测结果

![RAGArena 评测总览](docs/images/ragarena-evaluation-summary.png)

当前评测主要验证检索和引用命中能力，即系统是否能召回 gold evidence chunks/pages。这里不把端到端答案正确率作为结论指标。

### 评测数据规模

| 项目 | 数值 |
| --- | ---: |
| 论文数量 | 10 |
| Docling 解析 block | 998 |
| 已索引 chunk | 882 |
| QA 数量 | 100 |
| 图表/表格 QA 数量 | 73 |
| Embedding 模型 | `Qwen3-Embedding-4B` |
| 决策/切片模型 | Ollama `qwen3.5:4b` |

Chunk 分布：

| Strategy / Type | 数量 |
| --- | ---: |
| `fixed:fixed` | 587 |
| `agentic:retrieval_unit` | 199 |
| `agentic:figure_caption` | 39 |
| `agentic:table` | 21 |
| `agentic_fusion:fused` | 36 |

### 固定切片 vs Agentic 切片

数据集：`data/eval/qa_ablation_100.json`  
报告：`reports/evaluation/ablation_10paper_hybrid.json`

| 方案 | Cases | P50 ms | P95 ms | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定切片 + 混合检索 | 100 | 69.179 | 84.223 | 0.0683 | 0.1682 | 0.2337 | 0.3784 | 0.6154 | 0.3720 |
| Agentic 切片 + 混合检索 | 100 | 58.715 | 79.781 | 0.0963 | 0.2327 | 0.3381 | 0.5301 | 0.7755 | 0.5181 |
| 相对提升 | - | - | - | +41.0% | +38.4% | +44.7% | +40.1% | +26.0% | +39.3% |

### 图表/表格类问题 Ablation

该子集只统计 figure/table 问题。

| 方案 | Cases | P50 ms | P95 ms | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定切片 + 混合检索 | 73 | 59.296 | 65.624 | 0.0730 | 0.1583 | 0.2045 | 0.3413 | 0.6432 | 0.3535 |
| Agentic visual/fused chunk + 混合检索 | 73 | 58.479 | 80.008 | 0.0886 | 0.1814 | 0.2686 | 0.4711 | 0.7697 | 0.4699 |
| 相对提升 | - | - | - | +21.4% | +14.6% | +31.3% | +38.0% | +19.7% | +32.9% |

### 检索链路对比样本

数据集：`data/eval/qa_ablation_sample30.json`  
报告：`reports/evaluation/retrieval_stack_sample30.json`

| 方案 | Cases | P50 ms | P95 ms | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 30 | 5.528 | 6.332 | 0.5256 | 0.7583 | 0.5170 |
| Dense | 30 | 47.132 | 87.914 | 0.5395 | 0.7585 | 0.5245 |
| Hybrid | 30 | 55.310 | 62.539 | 0.5145 | 0.8014 | 0.5223 |
| Hybrid + HyDE | 30 | 939.924 | 1277.299 | 0.5284 | 0.7900 | 0.5270 |

### Rerank 抽样结果

数据集：`data/eval/qa_ablation_rerank_sample10.json`  
报告：`reports/evaluation/rerank_sample10.json`

| 方案 | Cases | P50 ms | P95 ms | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 10 | 59.122 | 7590.011 | 0.4548 | 0.7958 | 0.4740 |
| Hybrid + HyDE + Rerank | 10 | 27991.271 | 32062.291 | 0.5321 | 0.9000 | 0.5661 |

Rerank 在小样本上提升明显，但本地延迟较高。因此当前 README 不把完整 100 QA rerank 作为主结果，只把它作为精排收益样本展示。

## 快速开始

### 1. 安装依赖

```powershell
uv sync
```

### 2. 启动基础设施

```powershell
docker compose up -d postgres elasticsearch redis
```

当前 `docker-compose.yml` 已将 PostgreSQL 数据目录绑定到 E 盘，避免长期实验把系统盘占满。

### 3. 准备本地模型

拉取本地决策/切片模型：

```powershell
ollama pull qwen3.5:4b
```

下载 Qwen3-Embedding-4B，并在 `.env` 中配置本地路径：

```powershell
uv run hf download Qwen/Qwen3-Embedding-4B --local-dir E:\models\Qwen3-Embedding-4B
```

```env
EMBEDDING_MODEL=E:\models\Qwen3-Embedding-4B
EMBEDDING_DIMENSIONS=2560
AGENT_DECISION_MODEL=qwen3.5:4b
AGENTIC_CHUNK_MODEL=qwen3.5:4b
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 4. 启动 API

```powershell
uv run uvicorn app.main:app --reload
```

常用接口：

| Method | Path | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 服务健康检查 |
| `POST` | `/api/v1/search` | 检索接口 |
| `POST` | `/api/v1/ask` | 普通 RAG 问答 |
| `POST` | `/api/v1/agent` | LangGraph Agentic RAG |
| `POST` | `/api/v1/stream` | 流式回答 |

## 复现实验

清空数据：

```powershell
uv run ragarena-reset-data --yes
```

准备 10 篇论文 ablation 语料，包括固定切片、agentic 切片、fused chunk、embedding、Elasticsearch 索引和 100 条 QA：

```powershell
uv run ragarena-prepare-ablation-corpus `
  --papers-dir E:\ragarena-data\papers `
  --limit 10 `
  --qa-per-paper 10 `
  --output data/eval/qa_ablation_100.json `
  --plan-output data/eval/ablation_10paper_plan.json `
  --planner-provider ollama `
  --planner-model qwen3.5:4b
```

运行固定切片 vs agentic 切片、图表/表格子集 ablation：

```powershell
uv run ragarena-eval benchmark `
  --dataset data/eval/qa_ablation_100.json `
  --plan data/eval/ablation_10paper_hybrid_plan.json `
  --output reports/evaluation/ablation_10paper_hybrid.json `
  --markdown reports/evaluation/ablation_10paper_hybrid.md
```

运行检索链路对比样本：

```powershell
uv run ragarena-eval benchmark `
  --dataset data/eval/qa_ablation_sample30.json `
  --plan data/eval/retrieval_stack_sample30_plan.json `
  --output reports/evaluation/retrieval_stack_sample30.json `
  --markdown reports/evaluation/retrieval_stack_sample30.md
```

运行 rerank 抽样评测：

```powershell
uv run ragarena-eval benchmark `
  --dataset data/eval/qa_ablation_rerank_sample10.json `
  --plan data/eval/rerank_sample10_plan.json `
  --output reports/evaluation/rerank_sample10.json `
  --markdown reports/evaluation/rerank_sample10.md
```

## 项目结构

```text
app/                         FastAPI 应用与 API 路由
src/ragarena/agent/           LangGraph 工作流与 agent 策略
src/ragarena/chunking/        固定切片、block 切片、agentic 切片
src/ragarena/cli/             解析、索引、检索、评测命令行工具
src/ragarena/evaluation/      评测框架与检索指标
src/ragarena/embedding/       embedding 编码与存储
src/ragarena/papers/          arXiv 下载与 PDF 解析
src/ragarena/retrieval/       Elasticsearch BM25/向量检索与 RRF
src/ragarena/reranking/       BGE reranker 封装
reports/evaluation/           评测输出
docs/images/                  架构图与评测图
tests/                        pytest 测试
```

## 测试

```powershell
uv run pytest
```

只跑检索相关测试：

```powershell
uv run pytest tests/test_retrieval.py
```

## 说明

- README 中的评测结果来自当前 10 篇论文实验语料。
- 当前评测重点是检索命中和引用证据命中，不把 `Answer Acc` 作为主结论。
- Rerank 有收益，但本地延迟较高；当前代码已经限制 rerank 候选数和输入长度。
- 当前仓库尚未包含 license 文件。
