# Evaluation

ResearchRAG ships an evaluation harness (`researchrag eval ...`) that scores retrieval, the
context sent to the model, citations, faithfulness, correctness and abstention, with bootstrap
confidence intervals, ablation sweeps and a CI regression gate.

> **Read this first.** Every number in this repository was produced by the offline components
> (hashing embedder, mock extractive LLM, lexical judge) on a synthetic corpus written for the
> project. Synthetic corpora overstate real-world performance: the questions were written by the
> same process as the papers, the layouts are clean, and the mock LLM copies sentences, so its
> faithfulness is high by construction. These numbers demonstrate that the pipeline and the
> measurement work end to end; they are not a claim about language-model quality.

## Dataset

| | |
|---|---|
| Corpus | 16 fictional mini-papers in 4 topics (information retrieval, climate/ecology, energy materials, computational health), 2-3 pages each, rendered to two-column PDFs with running headers, footers, page numbers, tables with captions and reference lists (`data/corpus/papers/*.yaml`, `researchrag corpus build`) |
| Questions | 88 (`data/eval/questions_*.jsonl`): 32 single-hop, 16 multi-hop (evidence from two papers), 16 numeric/table, 12 unanswerable, 12 adversarial near-misses (6 answerable with a misleading premise, 6 unanswerable); 17 unanswerable overall (19%) |
| Gold labels | each answerable question has a short gold answer and gold evidence: the paper key plus a verbatim passage (one or two sentences, or one table row). `scripts/validate_corpus.py` checks every passage is a verbatim substring of the paper, and a test checks every gold passage survives PDF extraction |
| Calibration set | 40 claim/evidence pairs (10 per topic) labelled supported / partially supported / unsupported / contradicted (`data/eval/calibration_*.jsonl`) |
| Versioning | the dataset is identified by a SHA-256 over the question files; reports and baselines record it |

How the labels were made: the papers, questions, gold answers, gold evidence and calibration
labels were written with AI assistance for this project (fictional authors, places, datasets and
numbers), then validated programmatically (verbatim evidence, question-type mix). The calibration
labels have **not** been reviewed by independent human annotators; treat them as a careful single
annotation, and relabel them before relying on judge-agreement numbers.

## Metrics

Relevance is defined against gold evidence, independently of chunking: a retrieved text is
relevant to a gold passage when it comes from the gold paper and contains at least 60% of the
passage's content terms (stemmed, stop-words removed). This keeps metrics comparable across chunk
sizes.

| Metric | Definition | Questions |
|---|---|---|
| hit@k | 1 if any of the top-k retrieved chunks is relevant | answerable |
| recall@k | share of gold passages covered by the top-k | answerable |
| precision@k | share of the top-k that is relevant | answerable |
| MRR | 1 / rank of the first relevant chunk | answerable |
| nDCG@k | binary-gain nDCG; each gold passage earns gain once (at its first match), so nDCG <= 1 | answerable |
| context precision | share of packed context passages that are relevant | answerable |
| context recall | share of gold passages present in the packed context | answerable |
| citation precision | share of cited passages that match some gold passage | answered, answerable |
| citation recall | share of gold passages that were cited | answered, answerable |
| faithfulness | (supported + 0.5 x partially supported) / claims, judged by the configured verifier on the returned answer | answered |
| hallucination rate | 1 if the first generated answer had any unsupported or contradicted claim (before the remove/regenerate policy) | answers with claims |
| removed-claim rate | share of claims removed by the policy | answers with claims |
| answer correctness | rubric judge vs the gold answer (1 = all key facts match, 0.5 = partial, 0 = wrong); unanswerable questions score 1 for an abstention and 0 otherwise | all |
| answer relevance | offline proxy: share of the question's content terms addressed by the answer | answered |
| abstention accuracy | 1 if the system abstained exactly when the question is unanswerable | all |
| over-abstention | abstained although answerable | answerable |
| under-abstention | answered although unanswerable | unanswerable |

Statistics: means with 95% percentile bootstrap intervals (2,000 resamples of questions). Sweeps
compare each configuration with the base using a paired bootstrap of per-question differences; a
change is marked significant when the 95% CI of the difference excludes zero.

Answer correctness uses the configured LLM as the judge (`judge` task). With the offline mock this
is a lexical rubric (gold-term recall, F1 and number agreement), which is crude; with a real LLM
the rubric in `generation/prompts.py` is applied.

## Results: offline components (illustrative)

Command: `researchrag eval run` (config `configs/eval-offline.yaml`: fake embedder, mock LLM that
injects a numeric hallucination into 20% of answers so the verifier has something to catch,
lexical reranker and judge, parent-child chunks 160/480 tokens, hybrid retrieval, in-memory exact
search). Full report: [example-report/eval-eval-offline.md](example-report/eval-eval-offline.md)
([HTML](example-report/eval-eval-offline.html), [JSON](example-report/eval-eval-offline.json)).

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| recall@5 | 0.803 | [0.711, 0.880] | 71 |
| MRR | 0.618 | [0.535, 0.708] | 71 |
| nDCG@5 | 0.646 | [0.560, 0.729] | 71 |
| context recall | 0.922 | [0.866, 0.965] | 71 |
| context precision | 0.302 | [0.268, 0.338] | 71 |
| citation precision | 0.527 | [0.403, 0.643] | 50 |
| citation recall | 0.630 | [0.490, 0.760] | 50 |
| faithfulness | 0.918 | [0.867, 0.964] | 55 |
| hallucination rate (before verification) | 0.286 | [0.175, 0.397] | 63 |
| removed-claim rate | 0.198 | [0.119, 0.291] | 63 |
| answer correctness | 0.476 | [0.393, 0.560] | 88 |
| abstention accuracy | 0.705 | [0.602, 0.796] | 88 |
| over-abstention | 0.296 | [0.197, 0.408] | 71 |
| under-abstention | 0.294 | [0.118, 0.529] | 17 |

By question type (answer correctness / abstention accuracy / recall@5): single-hop 0.477 / 0.781 /
0.844; multi-hop 0.262 / 0.500 / 0.594; numeric/table 0.468 / 0.812 / 0.938; adversarial 0.493 /
0.583 / 0.786; unanswerable 0.750 / 0.750 / n/a. Multi-hop questions are clearly the weakest:
the extractive mock rarely combines evidence from two papers.

Reading the numbers: retrieval puts gold evidence in the packed context for most answerable
questions (context recall 0.92), but the extractive mock often picks a sentence other than the gold
one (citation precision 0.53) and abstains on 30% of answerable questions. The verifier flags a
failing claim in 29% of first-draft answers (20% injected hallucinations plus lexical-judge false
positives); after removal and regeneration, returned answers have faithfulness 0.92 *as judged by
the same lexical judge*, which is not independent evidence (see calibration below).

## Ablation sweep (offline components, illustrative)

Command: `researchrag eval sweep configs/sweep.yaml` (one factor at a time from
`eval-offline.yaml`, all 88 questions, memory backend). ▲/▼ = significantly better/worse than the
base by a paired bootstrap. Full report: [example-report/sweep.md](example-report/sweep.md)
([HTML](example-report/sweep.html)).

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

Which factor moves which metric (largest significant change versus base):

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

Takeaways (for this corpus and these components):

- **Chunk size matters most.** 96-token children lose 14 points of recall@5 (more chunks compete
  for the top 5); 256-token children lose context recall and citation precision (fewer, broader
  passages fit the context budget). 160 tokens is the best of the three on retrieval.
- **top_k = 4 is too small** (context recall -0.077); 12 adds latency without gains.
- **MMR helps recall@5** (+0.042 versus disabling it) by removing near-duplicate children of the
  same parent.
- **Hybrid vs vector vs lexical**: no significant difference here, because the offline embedder is
  itself lexical (feature hashing). Expect hybrid to matter with a real semantic embedder.
- **Query rewriting** (mock) adds no quality and triples latency; the lexical reranker's effects
  are not significant on this set.
- **ef_search** cannot matter on the exact in-memory backend; see the Postgres HNSW sweep below.
- **Quality per dollar**: every configuration costs $0 offline, so the "best" pick
  (`chunking.overlap_tokens=0`) is decided by an answer-correctness difference that is *not*
  significant. With paid models the same report ranks configurations by quality per USD.

## HNSW sweep on PostgreSQL (offline components, illustrative)

Command: `researchrag eval sweep configs/sweep-hnsw.yaml --backend postgres` against the Compose
database (PostgreSQL 16, pgvector 0.8.7, iterative index scans on). Full report:
[example-report/sweep-hnsw.md](example-report/sweep-hnsw.md).

| config | recall@5 | mrr | ndcg@5 | context_recall | citation_precision | faithfulness | hallucination_rate | answer_correctness | abstention_accuracy | over_abstention | p50 ms | USD/query |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 79.4 | 0.000000 |
| retrieval.ef_search=4 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 83.5 | 0.000000 |
| retrieval.ef_search=8 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 83.3 | 0.000000 |
| retrieval.ef_search=16 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 82.4 | 0.000000 |
| retrieval.ef_search=200 | 0.803 | 0.619 | 0.646 | 0.922 | 0.527 | 0.918 | 0.297 | 0.476 | 0.705 | 0.296 | 82.4 | 0.000000 |
| retrieval.mode=vector | 0.817 | 0.620 | 0.652 | 0.915 | 0.537 | 0.917 | 0.302 | 0.467 | 0.693 | 0.310 | 86.1 | 0.000000 |

Result: **no measurable effect of `ef_search` at this scale**, and the PostgreSQL/HNSW results match
the exact in-memory search (recall@5 0.803 in both). Two reasons, both expected: the pipeline never
sets `ef_search` below the number of candidates requested (`candidate_k = 40`, because HNSW cannot
return more rows than `ef_search`), and the index holds only ~500 child chunks, where HNSW with
`ef_search >= 40` is effectively exact. The recall/latency trade-off of `ef_search` (ADR 0002)
appears on corpora with tens of thousands of chunks; this sweep is the harness for measuring it
there, not evidence of it.

## Judge calibration

`researchrag eval calibrate` runs the configured verifier judge over the 40 labelled pairs.

| Judge | Exact agreement | Supported-vs-not agreement | Cohen's kappa |
|---|---|---|---|
| lexical (offline) | 0.45 | 0.57 | 0.24 |

Confusion (rows = label, columns = lexical judge):

| label \ judge | supported | partially | unsupported | contradicted |
|---|---|---|---|---|
| supported | 0 | 1 | 10 | 1 |
| partially supported | 0 | 3 | 5 | 0 |
| unsupported | 0 | 0 | 12 | 0 |
| contradicted | 0 | 1 | 4 | 3 |

The lexical judge never recognises paraphrased support (it needs shared terms) and so is
conservative: it rarely passes an unsupported claim but rejects many supported paraphrases. It
also has a known false negative in the other direction: a claim that repeats the evidence and adds
an unsupported causal clause can still pass on word overlap (`tests/unit/test_generation_verification.py`
documents this case). Tuning its thresholds on these 40 items moved agreement by at most 5 points,
within noise, so the defaults were kept rather than overfitting the set. Use the LLM judge
(`verification.judge: llm`) or a local NLI cross-encoder (`nli`) for real verification, and
calibrate them with the same command.

## Regression gate (CI)

`researchrag eval check --baseline main` reruns the offline evaluation and compares it with
`eval/baselines/main.json`, paired by question, on recall@5, MRR, context recall, faithfulness,
answer correctness and abstention accuracy. A metric fails when its mean drops by more than the
tolerance (default 0.03 absolute) **and** the paired bootstrap 95% CI of the change lies below
zero. Requiring both keeps the gate quiet on noise and loud on real drops.

Evidence that it works (real runs):

- Re-running the unchanged config: every diff is exactly 0.000 (the offline stack is
  deterministic) and the gate passes (`example-report/gate.log`).
- A seeded regression ([seeded-regression-config.yaml](example-report/seeded-regression-config.yaml):
  `retrieval.top_k = 1`, `candidate_k = 3`, no reranker, no MMR) fails with five significant
  regressions and exit code 1 ([gate-seeded-regression.log](example-report/gate-seeded-regression.log)):
  recall@5 0.803 -> 0.422 (diff -0.380, CI [-0.479, -0.282]), context recall 0.922 -> 0.465,
  answer correctness 0.476 -> 0.340, abstention accuracy 0.705 -> 0.489. Faithfulness does not move
  (the few remaining answers are still extractive), which is why the gate watches several metrics.
- Unit tests check the decision rule on synthetic per-question vectors: random flips of 4% of
  outcomes pass; losing 25% of previously correct questions fails.

The baseline is updated deliberately with `researchrag eval check --update` in the same PR as the
change that moves it.

## Live evaluation

`configs/openai.yaml` and `configs/gemini.yaml` run the same harness with real models and an LLM
judge (`researchrag eval run --config configs/openai.yaml`), and the manual **Live evaluation**
workflow runs it in GitHub Actions from repository secrets. **It has not been run for this
repository**: no API keys were used, and the owner requires a zero-cost setup. Live runs cost money
(see `configs/pricing.yaml`).

## Latency and load

**Offline components, illustrative; measured 2026-11-21.**

- In-process latency per query in the offline evaluation (memory backend): p50 86 ms, p95 122 ms.
- HTTP load test: Locust (`loadtest/locustfile.py`), 10 users, 60 s, against the Compose stack
  (single uvicorn worker, PostgreSQL 16 + pgvector 0.8.7, 16 papers / ~500 retrievable chunks;
  load generator and server on the same 4-vCPU container host, so this is a lower bound):

```text
Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
GET      /health                                                                           57     0(0.00%) |      9       2      40      4 |    0.95        0.00
GET      /v1/documents                                                                     58     0(0.00%) |     30       9     135     18 |    0.97        0.00
POST     /v1/query                                                                        437     0(0.00%) |    122      54     547    100 |    7.29        0.00
         Aggregated                                                                       552     0(0.00%) |    101       2     547     89 |    9.20        0.00
Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
GET      /health                                                                                 4      6      7     10     29     35     38     40     40     40     40     57
GET      /v1/documents                                                                          18     26     46     55     66    100    110    140    140    140    140     58
POST     /v1/query                                                                             100    120    140    160    200    230    260    530    550    550    550    437
         Aggregated                                                                             89    110    130    140    190    230    250    330    550    550    550    552
```

`POST /v1/query`: 437 requests, 0 failures, p50 100 ms, p95 230 ms, p99 530 ms, ~7.3 queries/s.
With real LLM providers, latency is dominated by the model call (seconds), not by retrieval.
