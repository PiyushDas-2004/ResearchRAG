"""Evaluation harness: metrics, statistics, regression gate, sweeps, calibration and reports."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchrag.config.pipeline import PipelineConfig
from researchrag.config.pricing import PricingTable
from researchrag.evaluation.calibration import calibrate, cohen_kappa
from researchrag.evaluation.dataset import EvalQuestion, GoldEvidence, load_calibration, load_dataset
from researchrag.evaluation.export import export_cases, feedback_to_candidates, llmbench_available
from researchrag.evaluation.gate import (
    baseline_from_run,
    check_run,
    compare,
    load_baseline,
    save_baseline,
)
from researchrag.evaluation.metrics import (
    RetrievedText,
    abstention_metrics,
    answer_relevance,
    citation_metrics,
    context_metrics,
    retrieval_metrics,
)
from researchrag.evaluation.report import (
    calibration_markdown,
    run_html,
    run_markdown,
    sweep_html,
    sweep_markdown,
    write_reports,
)
from researchrag.evaluation.runner import EvalEnvironment, EvalRunner, is_offline, run_label
from researchrag.evaluation.stats import bootstrap_ci, paired_bootstrap, percentile
from researchrag.evaluation.sweep import SweepSpec, best_quality_per_dollar, expand, load_sweep, run_sweep
from researchrag.verification.judges import LexicalJudge
from tests.conftest import EVAL_DIR, PAPERS_DIR

GOLD = "Low-marsh cores buried 212 g C per square metre per year across fourteen cores."
Q = EvalQuestion(
    id="q1",
    question="What burial rate?",
    type="single_hop",
    answerable=True,
    gold_answer="212",
    evidence=[GoldEvidence(doc_key="a", passage=GOLD)],
)
U = EvalQuestion(id="u1", question="Unknown?", type="unanswerable", answerable=False, gold_answer="none")


def rt(key: str | None, text: str) -> RetrievedText:
    return RetrievedText(key, text)


def test_retrieval_metrics() -> None:
    hit = rt("a", "Intro text. " + GOLD)
    miss = rt("b", GOLD)  # right text, wrong document
    noise = rt("a", "Something unrelated about bees.")
    m = retrieval_metrics(Q, [noise, hit, hit, miss])
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert 0 < m["ndcg@3"] <= 1.0  # duplicate matches earn no extra gain
    assert retrieval_metrics(U, [hit]) == {}
    assert retrieval_metrics(Q, [])["mrr"] == 0.0


@given(st.lists(st.booleans(), max_size=10))
def test_ndcg_never_exceeds_one(pattern: list[bool]) -> None:
    ranked = [rt("a", GOLD) if p else rt("a", "noise text") for p in pattern]
    for k in (1, 3, 5):
        assert 0.0 <= retrieval_metrics(Q, ranked)[f"ndcg@{k}"] <= 1.0  # type: ignore[operator]


def test_context_citation_abstention_metrics() -> None:
    assert context_metrics(Q, []) == {"context_precision": 0.0, "context_recall": 0.0}
    ctx = context_metrics(Q, [rt("a", GOLD), rt("a", "noise")])
    assert ctx == {"context_precision": 0.5, "context_recall": 1.0}
    assert citation_metrics(Q, [rt("a", GOLD)]) == {"citation_precision": 1.0, "citation_recall": 1.0}
    assert citation_metrics(Q, []) == {}
    assert context_metrics(U, []) == {}
    assert abstention_metrics(Q, True) == {
        "abstention_accuracy": 0.0,
        "over_abstention": 1.0,
        "under_abstention": None,
    }
    assert abstention_metrics(U, False) == {
        "abstention_accuracy": 0.0,
        "over_abstention": None,
        "under_abstention": 1.0,
    }
    assert answer_relevance("What burial rate in marsh?", "The burial rate in the marsh was 212.") == 1.0


def test_bootstrap_and_paired() -> None:
    est = bootstrap_ci([1.0, 0.0] * 50)
    assert est.mean == 0.5
    assert est.low < 0.5 < est.high
    assert bootstrap_ci([]).n == 0
    same = paired_bootstrap([0.5] * 20, [0.5] * 20)
    assert same.diff == 0.0
    assert same.p_value == 1.0
    assert not same.significant
    better = paired_bootstrap([0.0] * 30, [1.0] * 30)
    assert better.significant
    assert better.diff == 1.0
    assert paired_bootstrap([None], [1.0]).n == 0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def _values(n: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {f"q{i}": float(rng.random() < 0.7) for i in range(n)}


def test_gate_fails_on_seeded_regression_and_passes_on_noise() -> None:
    base = {"recall@5": _values(80, 1), "faithfulness": {f"q{i}": 0.9 for i in range(80)}}
    rng = random.Random(2)
    # Noise: flip ~4% of binary outcomes in both directions; faithfulness jitters +/- 0.02.
    noisy = {
        "recall@5": {k: (1.0 - v if rng.random() < 0.04 else v) for k, v in base["recall@5"].items()},
        "faithfulness": {k: v + rng.uniform(-0.02, 0.02) for k, v in base["faithfulness"].items()},
    }
    passed = compare(base, noisy, metrics=("recall@5", "faithfulness"))
    assert passed.passed, passed
    # Seeded regression: recall lost on 25% of questions that previously succeeded.
    regressed = {
        "recall@5": {
            k: (0.0 if v == 1.0 and int(k[1:]) % 4 == 0 else v) for k, v in base["recall@5"].items()
        },
        "faithfulness": base["faithfulness"],
    }
    failed = compare(base, regressed, metrics=("recall@5", "faithfulness"))
    assert not failed.passed
    assert [c.metric for c in failed.checks if c.regressed] == ["recall@5"]
    lower_better = compare(
        {"hallucination_rate": {"a": 0.0, "b": 0.0, "c": 0.0}},
        {"hallucination_rate": {"a": 1.0, "b": 1.0, "c": 1.0}},
        metrics=("hallucination_rate",),
    )
    assert not lower_better.passed
    empty = compare({"mrr": {}}, {"mrr": {"x": 1.0}}, metrics=("mrr",))
    assert empty.passed
    assert empty.warnings


def test_dataset_and_calibration_loading() -> None:
    data = load_dataset(EVAL_DIR)
    assert len(data.questions) == 88
    assert len(data.content_hash) == 64
    assert data.subset(5).questions == data.questions[:5]
    assert all(q.type == "multi_hop" for q in data.subset(types={"multi_hop"}).questions)
    assert sum(not q.answerable for q in data.questions) >= 13
    assert len(load_calibration(EVAL_DIR)) == 40


async def test_calibration_report() -> None:
    report = await calibrate(LexicalJudge(), load_calibration(EVAL_DIR))
    assert report.n == 40
    assert sum(sum(row.values()) for row in report.confusion.values()) == 40
    assert -1.0 <= report.cohen_kappa <= 1.0
    assert "Judge calibration" in calibration_markdown(report)
    assert cohen_kappa([]) == 0.0
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0


@pytest.fixture(scope="module")
async def small_run():  # type: ignore[no-untyped-def]
    env = EvalEnvironment.memory(PAPERS_DIR)
    runner = EvalRunner(env, load_dataset(EVAL_DIR).subset(12), pricing=PricingTable())
    config = PipelineConfig().with_overrides({"generation.mock.hallucination_rate": 0.3}, name="t")
    return runner, await runner.run(config)


async def test_runner_produces_complete_results(small_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _runner, run = small_run
    assert len(run.questions) == 12
    assert is_offline(run.components)
    assert "illustrative" in run_label(run)
    assert {"recall@5", "answer_correctness", "abstention_accuracy"} <= set(run.summary)
    assert run.latency_ms["p95"] >= run.latency_ms["p50"]
    assert run.cost["total_usd"] == 0.0
    md = run_markdown(run)
    assert "Worst-scoring questions" in md
    html_text = run_html(run)
    assert html_text.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html_text
    paths = write_reports(tmp_path, "r", md, html_text, run.model_dump_json())
    assert all(p.exists() for p in paths)
    baseline = baseline_from_run(run)
    save_baseline(baseline, tmp_path / "b.json")
    loaded = load_baseline(tmp_path / "b.json")
    report = check_run(run, loaded)
    assert report.passed
    loaded.dataset_hash = "other"
    assert check_run(run, loaded).warnings


async def test_runner_detects_degraded_config(small_run) -> None:  # type: ignore[no-untyped-def]
    runner, run = small_run
    degraded = PipelineConfig().with_overrides(
        {
            "retrieval.top_k": 1,
            "retrieval.candidate_k": 2,
            "rerank.provider": "none",
            "generation.mock.hallucination_rate": 0.3,
        },
        name="degraded",
    )
    bad = await runner.run(degraded)
    report = check_run(bad, baseline_from_run(run), tolerance=0.03)
    assert not report.passed


def test_sweep_expansion_and_best(tmp_path: Path) -> None:
    base = PipelineConfig()
    spec = SweepSpec(factors={"retrieval.mode": ["hybrid", "vector"], "retrieval.top_k": [8, 4]})
    names = [n for n, _, _ in expand(spec, base)]
    assert names == ["base", "retrieval.mode=vector", "retrieval.top_k=4"]
    grid = SweepSpec(
        mode="grid", factors={"retrieval.mode": ["hybrid", "vector"], "rerank.provider": ["lexical", "none"]}
    )
    assert len(expand(grid, base)) == 4
    with pytest.raises(Exception, match="Unknown sweep factor"):
        expand(SweepSpec(factors={"nope.x": [1]}), base)
    with pytest.raises(Exception, match="max_configs"):
        expand(SweepSpec(factors={"retrieval.top_k": list(range(1, 50))}, max_configs=5), base)
    file = tmp_path / "s.yaml"
    file.write_text("base: missing.yaml\nfactors: {retrieval.top_k: [3]}\n")
    with pytest.raises(Exception, match="not found"):
        load_sweep(file)
    (tmp_path / "b.yaml").write_text("retrieval: {top_k: 5}\n")
    file.write_text("base: b.yaml\nfactors: {retrieval.top_k: [3, 5]}\n")
    spec2, base2 = load_sweep(file)
    assert base2.retrieval.top_k == 5
    assert [n for n, _, _ in expand(spec2, base2)] == ["base", "retrieval.top_k=3"]


async def test_run_sweep_end_to_end(small_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner, _ = small_run
    spec = SweepSpec(factors={"retrieval.mode": ["hybrid", "lexical"]})
    result = await run_sweep(runner, spec, PipelineConfig().with_overrides({}, name="base"))
    assert [r.name for r in result.runs] == ["base", "retrieval.mode=lexical"]
    assert "retrieval.mode=lexical" in result.comparisons
    assert "zero model cost" in result.best_note
    md = sweep_markdown(result)
    assert "Which factor moves which metric" in md
    assert sweep_html(result).startswith("<!doctype html>")
    costly = [
        r.model_copy(update={"cost": {"total_usd": 1.0, "per_query_usd": c}})
        for r, c in zip(result.runs, (0.01, 0.001), strict=True)
    ]
    name, note = best_quality_per_dollar(costly, "answer_correctness")
    assert name in {"base", "retrieval.mode=lexical"}
    assert "per USD" in note


def test_export_helpers(tmp_path: Path) -> None:
    data = load_dataset(EVAL_DIR).subset(2)
    path = export_cases(
        data,
        {data.questions[0].id: {"generate": {"prompt": [{"role": "system", "content": "x"}]}}},
        "cfg",
        tmp_path / "c.jsonl",
    )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["messages"][0]["role"] == "system"
    assert lines[1]["messages"] is None
    assert isinstance(llmbench_available(), bool)
    cands = feedback_to_candidates(
        [{"trace_id": "abcdef123456", "question": "q?", "answer": "a", "comment": "wrong"}]
    )
    assert cands[0]["needs_review"] is True
    assert cands[0]["id"] == "fb-abcdef12"
