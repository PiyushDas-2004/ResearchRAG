# Ablation sweep
**Components:** offline mock components, illustrative (fake embedder + mock extractive LLM)  
**Questions:** 88; dataset hash `feae585ee401`

Each configuration changes one factor from the base (one-factor-at-a-time unless noted).
▲/▼ = significantly better/worse than base (paired bootstrap 95% CI of the per-question difference excludes 0).

| config | recall@5 | mrr | ndcg@5 | context_recall | citation_precision | faithfulness | hallucination_rate | answer_correctness | abstention_accuracy | over_abstention | p50 ms | USD/query |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 79.4 | 0.000000 |
| retrieval.ef_search=4 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 83.5 | 0.000000 |
| retrieval.ef_search=8 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 83.3 | 0.000000 |
| retrieval.ef_search=16 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 82.4 | 0.000000 |
| retrieval.ef_search=200 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 82.4 | 0.000000 |
| retrieval.mode=vector | 0.817 | 0.620 | 0.652 | 0.915 | 0.537 | 0.917 | 0.302 | 0.467 | 0.693 | 0.310 | 86.1 | 0.000000 |

## Which factor moves which metric
| factor | significant changes vs base (largest per metric) |
|---|---|
| retrieval.ef_search | no significant change |
| retrieval.mode | no significant change |

**Best quality per dollar:** `retrieval.mode=vector`. All configurations had zero model cost (offline components), so the pick maximises recall@5 and breaks ties by p50 latency.
