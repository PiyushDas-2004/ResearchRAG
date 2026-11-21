# Ablation sweep
**Components:** offline mock components, illustrative (fake embedder + mock extractive LLM)  
**Questions:** 88; dataset hash `feae585ee401`

Each configuration changes one factor from the base (one-factor-at-a-time unless noted).
▲/▼ = significantly better/worse than base (paired bootstrap 95% CI of the per-question difference excludes 0).

| config | recall@5 | mrr | ndcg@5 | context_recall | citation_precision | faithfulness | hallucination_rate | answer_correctness | abstention_accuracy | over_abstention | p50 ms | USD/query |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.803 | 0.618 | 0.646 | 0.922 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 86.8 | 0.000000 |
| retrieval.mode=vector | 0.817 | 0.620 | 0.652 | 0.915 | 0.537 | 0.917 | 0.302 | 0.467 | 0.693 | 0.310 | 88.4 | 0.000000 |
| retrieval.mode=lexical | 0.782 | 0.623 | 0.639 | 0.915 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 39.7 | 0.000000 |
| chunking.chunk_tokens=96 | 0.662 ▼ | 0.559 | 0.550 ▼ | 0.901 | 0.557 | 0.915 | 0.297 | 0.486 ▲ | 0.705 | 0.296 | 94.0 | 0.000000 |
| chunking.chunk_tokens=256 | 0.824 | 0.655 | 0.675 | 0.880 ▼ | 0.454 ▼ | 0.915 | 0.286 | 0.482 | 0.727 | 0.282 | 92.5 | 0.000000 |
| chunking.overlap_tokens=0 | 0.768 | 0.618 | 0.629 | 0.908 | 0.517 | 0.926 | 0.274 | 0.488 | 0.716 | 0.296 | 92.2 | 0.000000 |
| retrieval.top_k=4 | 0.768 ▼ | 0.601 ▼ | 0.630 ▼ | 0.845 ▼ | 0.557 | 0.923 | 0.274 | 0.484 | 0.716 | 0.296 | 91.0 | 0.000000 |
| retrieval.top_k=12 | 0.803 | 0.623 | 0.646 | 0.922 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 82.7 | 0.000000 |
| rerank.provider=none | 0.782 | 0.623 | 0.630 | 0.887 | 0.583 | 0.923 | 0.295 | 0.473 | 0.693 | 0.324 | 87.6 | 0.000000 |
| rewrite.enabled=true | 0.803 | 0.618 | 0.646 | 0.922 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 283.5 | 0.000000 |
| retrieval.ef_search=16 | 0.803 | 0.618 | 0.646 | 0.922 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 86.2 | 0.000000 |
| retrieval.ef_search=200 | 0.803 | 0.618 | 0.646 | 0.922 | 0.527 | 0.918 | 0.286 | 0.476 | 0.705 | 0.296 | 83.9 | 0.000000 |
| retrieval.mmr_enabled=false | 0.761 ▼ | 0.632 | 0.632 | 0.908 | 0.517 | 0.918 | 0.286 | 0.482 | 0.705 | 0.296 | 85.7 | 0.000000 |

## Which factor moves which metric
| factor | significant changes vs base (largest per metric) |
|---|---|
| retrieval.mode | no significant change |
| chunking.chunk_tokens | recall@5 -0.141 (chunking.chunk_tokens=96); ndcg@5 -0.096 (chunking.chunk_tokens=96); context_recall -0.042 (chunking.chunk_tokens=256); citation_precision -0.063 (chunking.chunk_tokens=256); answer_correctness +0.010 (chunking.chunk_tokens=96) |
| chunking.overlap_tokens | no significant change |
| retrieval.top_k | recall@5 -0.035 (retrieval.top_k=4); mrr -0.018 (retrieval.top_k=4); ndcg@5 -0.015 (retrieval.top_k=4); context_recall -0.077 (retrieval.top_k=4) |
| rerank.provider | no significant change |
| rewrite.enabled | no significant change |
| retrieval.ef_search | no significant change |
| retrieval.mmr_enabled | recall@5 -0.042 (retrieval.mmr_enabled=false) |

**Best quality per dollar:** `chunking.overlap_tokens=0`. All configurations had zero model cost (offline components), so the pick maximises answer_correctness and breaks ties by p50 latency.
