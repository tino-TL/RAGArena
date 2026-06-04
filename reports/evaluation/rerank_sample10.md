# RAGArena Benchmark Report

Dataset: `data\eval\qa_ablation_rerank_sample10.json`
Plan: `data\eval\rerank_sample10_plan.json`
Cases: 10

| Variant | Cases | Errors | P50 ms | P95 ms | Citation Hit | Answer Acc | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D_rerank_sample10::hybrid_agentic | 10 | 0 | 59.122 | 7590.011 | 1.0 | 0.0 | 0.0875 | 0.1768 | 0.2911 | 0.4548 |
| D_rerank_sample10::hybrid_hyde_rerank_agentic | 10 | 0 | 27991.271 | 32062.291 | 1.0 | 0.0 | 0.1018 | 0.2637 | 0.347 | 0.5321 |
