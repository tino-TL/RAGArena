from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    recall: float
    mrr: float
    ndcg: float
    hit_rate: float


def evaluate_ranked_results(
    ranked_ids: list[int],
    relevant_ids: set[int],
    *,
    top_k: int,
) -> RetrievalMetrics:
    if not relevant_ids:
        return RetrievalMetrics(recall=0.0, mrr=0.0, ndcg=0.0, hit_rate=0.0)

    ranked_at_k = ranked_ids[:top_k]
    hits = [1 if item_id in relevant_ids else 0 for item_id in ranked_at_k]
    hit_count = sum(hits)
    recall = hit_count / len(relevant_ids)
    hit_rate = 1.0 if hit_count else 0.0
    mrr = reciprocal_rank(ranked_at_k, relevant_ids)
    ndcg = ndcg_at_k(hits, min(len(relevant_ids), top_k))
    return RetrievalMetrics(recall=recall, mrr=mrr, ndcg=ndcg, hit_rate=hit_rate)


def reciprocal_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    for index, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / log2(index + 2) for index, hit in enumerate(hits))
    ideal_dcg = sum(1.0 / log2(index + 2) for index in range(ideal_hits))
    return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg
