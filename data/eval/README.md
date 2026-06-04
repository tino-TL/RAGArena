# RAGArena Evaluation Framework

This workspace is the source of truth for resume-grade RAGArena evaluation. Keep every number reproducible from the files in this directory and reports under `reports/evaluation`.

## Evaluation Goals

The framework is designed for the claims in the resume:

- Visual/fused chunk enhancement improves figure/table recall and answer correctness.
- Agentic chunking beats fixed-length chunking on long PDF QA.
- Local `qwen3.5:4b` decision nodes keep answer quality stable while reducing main-LLM usage and latency.
- `BM25 -> dense -> HyDE -> RRF -> BGE-Reranker` improves retrieval layer by layer.

## Files

- `qa_gold.json`: gold QA set. Fill this first.
- `retrieval.jsonl`: chunk/document-id-only retrieval benchmark. Use it after ingestion when exact ids are known.
- `answer_judge.json`: human or judge-model answer labels.
- `ablation_plan.json`: experiments and variants.

## Gold QA Schema

Use 30-50 high-quality questions minimum. For strong results, include at least 99 QA items and keep a stable train/dev split outside the report if you tune prompts.

```json
{
  "id": "qa-0001",
  "query": "According to Figure 2, what trend is shown by the proposed method?",
  "expected_answer": "The proposed method improves recall as context length increases.",
  "category": "figure_table",
  "tags": ["visual", "numeric", "citation"],
  "difficulty": "medium",
  "arxiv_id": "2606.xxxxx",
  "paper_id": null,
  "gold_sections": ["Experiments"],
  "gold_keywords": ["Figure 2", "recall", "context length"],
  "gold_chunk_ids": [],
  "gold_document_ids": [],
  "gold_page_numbers": [6],
  "gold_visual_refs": ["Figure 2"],
  "answer_must_include": ["improves recall", "context length"],
  "answer_must_not_include": ["decreases"],
  "notes": ""
}
```

Recommended categories:

- `figure_table`: requires figure/table evidence.
- `numeric_table`: asks for numbers, comparisons, or table values.
- `long_context`: requires evidence across long sections.
- `cross_section`: requires multiple sections.
- `method`: asks about algorithm/pipeline design.
- `definition`: asks about a term or component.
- `citation`: answer must cite the right source.

## Gold Quality Rules

- `gold_chunk_ids` gives the most accurate `recall@k`, `MRR`, and `nDCG`.
- `gold_document_ids` is acceptable before chunk ids are finalized, but weaker.
- `gold_sections`, `gold_keywords`, `gold_page_numbers`, and `gold_visual_refs` measure citation/source grounding.
- `answer_must_include` and `answer_must_not_include` are deterministic sanity checks, not a replacement for judge labels.
- Put `visual` or `numeric` in `tags` for questions that should count toward visual/table ablations.

## Answer Judge Schema

After running answers, label each `(qa_id, variant)` pair:

```json
{
  "qa_id": "qa-0001",
  "variant": "D_retrieval_stack::hybrid_hyde_rerank",
  "answer_correct": true,
  "citation_correct": true,
  "unsupported_claim": false,
  "score": 1.0,
  "notes": "Answer cites Figure 2 and states the correct trend."
}
```

Use `answer_correct` for answer accuracy and `citation_correct` for source-grounded correctness. Use the same rubric for all variants.

## Running Evaluation

Validate the retrieval-only JSONL set:

```cmd
uv run ragarena-eval run --dataset data/eval/retrieval.jsonl --strategies bm25,dense,hybrid,hybrid_hyde,hybrid_hyde_rerank --top-k 1,3,5,10
```

Run the full benchmark plan:

```cmd
uv run ragarena-eval benchmark --dataset data/eval/qa_gold.json --plan data/eval/ablation_plan.json --judge data/eval/answer_judge.json --top-k 1,3,5,10 --output reports/evaluation/benchmark.json
```

Outputs:

- `reports/evaluation/benchmark.json`: machine-readable details for every case.
- `reports/evaluation/benchmark.md`: comparison table for reporting.

## Ablation Requirements

A. Visual/fused enhancement:

- Build one baseline index without visual/fused chunks: `ragarena_body_only_chunks`.
- Build one enhanced index with body/visual/fused chunks: `ragarena_chunks`.
- Compare only `category in ["figure_table", "numeric_table"]`.

B. Agentic chunking:

- Ingest the same documents with `chunk_strategy=fixed`.
- Ingest the same documents with `chunk_strategy=agentic`.
- Compare on the same QA set and same retrieval strategy.

C. Local 4B decision pipeline:

- Run one variant where route/rewrite/grade/chunk decisions use the main LLM.
- Run one variant where those decisions use `qwen3.5:4b`.
- Report answer correctness, citation correctness, p50/p95 latency, and main-LLM token usage.

D. Retrieval stack:

- Run `bm25`, `dense`, `hybrid`, `hybrid_hyde`, `hybrid_hyde_rerank` on the same fixed index.
- Report `recall@1/3/5/10`, `MRR@10`, `nDCG@10`, hit rate, and p95 latency.

Only write a percentage in the resume when the two compared variants have `error_cases=0`, the same `case_count`, and the QA subset is fixed.
