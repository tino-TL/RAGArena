# RAGArena Benchmark Report

Dataset: `data\eval\qa_ablation_100.json`
Plan: `data\eval\ablation_10paper_hybrid_plan.json`
Cases: 100

| Variant | Cases | Errors | P50 ms | P95 ms | Citation Hit | Answer Acc | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_chunk_strategy_hybrid::fixed_hybrid | 100 | 0 | 69.179 | 84.223 | 1.0 | 0.0 | 0.0683 | 0.1682 | 0.2337 | 0.3784 |
| A_chunk_strategy_hybrid::agentic_hybrid | 100 | 0 | 58.715 | 79.781 | 1.0 | 0.0 | 0.0963 | 0.2327 | 0.3381 | 0.5301 |
| B_visual_fused_hybrid::fixed_visual_hybrid | 73 | 0 | 59.296 | 65.624 | 1.0 | 0.0 | 0.073 | 0.1583 | 0.2045 | 0.3413 |
| B_visual_fused_hybrid::agentic_visual_hybrid | 73 | 0 | 58.479 | 80.008 | 1.0 | 0.0 | 0.0886 | 0.1814 | 0.2686 | 0.4711 |
