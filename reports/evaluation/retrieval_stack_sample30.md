# RAGArena Benchmark Report

Dataset: `data\eval\qa_ablation_sample30.json`
Plan: `data\eval\retrieval_stack_sample30_plan.json`
Cases: 30

| Variant | Cases | Errors | P50 ms | P95 ms | Citation Hit | Answer Acc | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_retrieval_stack_sample30::bm25_agentic | 30 | 0 | 5.528 | 6.332 | 1.0 | 0.0 | 0.0847 | 0.2157 | 0.3119 | 0.5256 |
| C_retrieval_stack_sample30::dense_agentic | 30 | 0 | 47.132 | 87.914 | 1.0 | 0.0 | 0.0942 | 0.2026 | 0.3038 | 0.5395 |
| C_retrieval_stack_sample30::hybrid_agentic | 30 | 0 | 55.31 | 62.539 | 1.0 | 0.0 | 0.0972 | 0.2185 | 0.3385 | 0.5145 |
| C_retrieval_stack_sample30::hybrid_hyde_agentic | 30 | 0 | 939.924 | 1277.299 | 1.0 | 0.0 | 0.0937 | 0.2262 | 0.3254 | 0.5284 |
