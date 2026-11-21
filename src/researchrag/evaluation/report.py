"""Markdown and self-contained HTML evaluation reports."""

from __future__ import annotations

import html
import math
from pathlib import Path

from researchrag.evaluation.calibration import CalibrationReport
from researchrag.evaluation.runner import HEADLINE, LOWER_IS_BETTER, RunResult, run_label
from researchrag.evaluation.sweep import SweepResult, factor_effects

CSS = """
:root{--bg:#fff;--fg:#1d2330;--muted:#5b6577;--line:#d9dee7;--good:#17803d;--bad:#b42318;--head:#f3f5f9}
@media (prefers-color-scheme: dark){:root{--bg:#12151c;--fg:#e6e9ef;--muted:#9aa4b5;--line:#2c3340;--good:#4ac77a;--bad:#ff7b72;--head:#1b2029}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:1.6rem}h2{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--head)}
.label{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:12px;color:var(--muted);font-size:.85rem}
.good{color:var(--good)}.bad{color:var(--bad)}.muted{color:var(--muted)}details{margin:8px 0}code{font-size:.85em}
"""


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _metric_rows(run: RunResult, metrics: list[str]) -> list[list[str]]:
    rows = []
    for metric in metrics:
        est = run.summary.get(metric)
        if est is None:
            continue
        arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        rows.append(
            [metric, _fmt(est["mean"]), f"[{_fmt(est['low'])}, {_fmt(est['high'])}]", str(est["n"]), arrow]
        )
    return rows


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _html_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def worst_questions(run: RunResult, n: int = 8) -> list[dict[str, object]]:
    """Questions with the lowest answer correctness (ties: lowest recall@5)."""
    scored = sorted(
        run.questions,
        key=lambda q: (q.metrics.get("answer_correctness") or 0.0, q.metrics.get("recall@5") or 0.0, q.id),
    )
    return [q.model_dump() for q in scored[:n]]


def run_markdown(run: RunResult) -> str:
    """Markdown report for one run."""
    metrics = HEADLINE + sorted(set(run.summary) - set(HEADLINE))
    parts = [
        f"# Evaluation report: {run.name}",
        f"**Components:** {run_label(run)}  ",
        f"**Dataset:** `{run.dataset}` ({len(run.questions)} questions, hash `{run.dataset_hash[:12]}`)  ",
        f"**Config hash:** `{run.config_hash}`  **Backend:** {run.backend}  **Created:** {run.created_at}",
        "",
        "Means with 95% bootstrap confidence intervals (questions resampled, 2,000 resamples).",
        "",
        _md_table(["metric", "mean", "95% CI", "n", "direction"], _metric_rows(run, metrics)),
        "",
        f"Latency per query (ms): p50 {run.latency_ms['p50']}, p95 {run.latency_ms['p95']}. "
        f"Cost: {run.cost['total_usd']:.6f} USD total, {run.cost['per_query_usd']:.8f} USD/query.",
        "",
        "## By question type",
        _md_table(
            ["type", "n", "answer_correctness", "abstention_accuracy", "recall@5"],
            [
                [
                    t,
                    str(int(v.get("n", 0))),
                    _fmt(v.get("answer_correctness")),
                    _fmt(v.get("abstention_accuracy")),
                    _fmt(v.get("recall@5")),
                ]
                for t, v in run.by_type.items()
            ],
        ),
        "",
        "## Worst-scoring questions",
    ]
    for q in worst_questions(run):
        parts.append(f"- **{q['id']}** ({q['type']}): {q['question']}")
        parts.append(f"  - gold: {q['gold_answer']}")
        parts.append(f"  - answer: {q['answer'] or '(abstained) ' + str(q['abstain_reason'])}")
        for r in q["retrieved"][:3]:  # type: ignore[index]
            parts.append(f"  - retrieved `{r['doc_key']}` p.{r['page']}: {str(r['text'])[:160]}...")
    return "\n".join(parts) + "\n"


def run_html(run: RunResult) -> str:
    """Self-contained HTML report for one run."""
    metrics = HEADLINE + sorted(set(run.summary) - set(HEADLINE))
    worst = []
    for q in worst_questions(run):
        retrieved = "".join(
            f"<li><code>{html.escape(str(r['doc_key']))}</code> p.{r['page']}: {html.escape(str(r['text'])[:300])}</li>"
            for r in q["retrieved"]  # type: ignore[attr-defined]
        )
        answer = (
            html.escape(str(q["answer"]))
            or f"<span class='muted'>(abstained) {html.escape(str(q['abstain_reason']))}</span>"
        )
        worst.append(
            f"<details><summary><b>{html.escape(str(q['id']))}</b> ({q['type']}): {html.escape(str(q['question']))}</summary>"
            f"<p><b>Gold:</b> {html.escape(str(q['gold_answer']))}</p><p><b>Answer:</b> {answer}</p>"
            f"<p><b>Retrieved passages:</b></p><ol>{retrieved}</ol></details>"
        )
    types = _html_table(
        ["type", "n", "answer_correctness", "abstention_accuracy", "recall@5"],
        [
            [
                t,
                str(int(v.get("n", 0))),
                _fmt(v.get("answer_correctness")),
                _fmt(v.get("abstention_accuracy")),
                _fmt(v.get("recall@5")),
            ]
            for t, v in run.by_type.items()
        ],
    )
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ResearchRAG evaluation</title><style>{CSS}</style></head><body>"
        f"<h1>Evaluation report: {html.escape(run.name)}</h1><p><span class='label'>{html.escape(run_label(run))}</span></p>"
        f"<p>Dataset <code>{html.escape(run.dataset)}</code>, {len(run.questions)} questions, hash <code>{run.dataset_hash[:12]}</code>; "
        f"config <code>{run.config_hash}</code>; backend {run.backend}; created {run.created_at}.</p>"
        f"<h2>Metrics (95% bootstrap CI)</h2>{_html_table(['metric', 'mean', '95% CI', 'n', 'direction'], _metric_rows(run, metrics))}"
        f"<p>Latency p50 {run.latency_ms['p50']} ms, p95 {run.latency_ms['p95']} ms. Cost {run.cost['total_usd']:.6f} USD.</p>"
        f"<h2>By question type</h2>{types}<h2>Worst-scoring questions</h2>{''.join(worst)}</body></html>"
    )


def _sweep_rows(result: SweepResult) -> list[list[str]]:
    rows = []
    for run in result.runs:
        cells = [run.name]
        for metric in HEADLINE:
            est = run.summary.get(metric)
            cmp = result.comparisons.get(run.name, {}).get(metric)
            text = _fmt(est["mean"]) if est else "n/a"
            if cmp and cmp["significant"]:
                worse = (float(cmp["diff"]) > 0) == (metric in LOWER_IS_BETTER)
                text += " ▼" if worse else " ▲"
            cells.append(text)
        cells += [_fmt(run.latency_ms["p50"], 1), f"{run.cost['per_query_usd']:.6f}"]
        rows.append(cells)
    return rows


def sweep_markdown(result: SweepResult) -> str:
    """Markdown report for a sweep."""
    label = run_label(result.runs[0])
    effects = factor_effects(result)
    effect_rows = []
    for factor, metrics in effects.items():
        moved = [
            f"{m} {float(e['diff']):+.3f} ({e['config']})" for m, e in metrics.items() if e["significant"]
        ]
        effect_rows.append([factor, "; ".join(moved) or "no significant change"])
    return "\n".join(
        [
            "# Ablation sweep",
            f"**Components:** {label}  ",
            f"**Questions:** {len(result.runs[0].questions)}; dataset hash `{result.runs[0].dataset_hash[:12]}`",
            "",
            "Each configuration changes one factor from the base (one-factor-at-a-time unless noted).",
            "▲/▼ = significantly better/worse than base (paired bootstrap 95% CI of the per-question "
            "difference excludes 0).",
            "",
            _md_table(["config", *HEADLINE, "p50 ms", "USD/query"], _sweep_rows(result)),
            "",
            "## Which factor moves which metric",
            _md_table(["factor", "significant changes vs base (largest per metric)"], effect_rows),
            "",
            f"**Best quality per dollar:** `{result.best_quality_per_dollar}`. {result.best_note}",
            "",
        ]
    )


def sweep_html(result: SweepResult) -> str:
    """Self-contained HTML report for a sweep."""
    md_effects = factor_effects(result)
    rows = [
        [
            html.escape(f),
            "; ".join(f"{m} {float(e['diff']):+.3f}" for m, e in ms.items() if e["significant"]) or "none",
        ]
        for f, ms in md_effects.items()
    ]
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ResearchRAG sweep</title><style>{CSS}</style></head><body><h1>Ablation sweep</h1>"
        f"<p><span class='label'>{html.escape(run_label(result.runs[0]))}</span></p>"
        f"<p>▲/▼ = significantly better/worse than base (paired bootstrap 95% CI excludes 0).</p>"
        f"{_html_table(['config', *HEADLINE, 'p50 ms', 'USD/query'], [[html.escape(c) for c in r] for r in _sweep_rows(result)])}"
        f"<h2>Which factor moves which metric</h2>{_html_table(['factor', 'significant changes'], rows)}"
        f"<p><b>Best quality per dollar:</b> <code>{html.escape(result.best_quality_per_dollar)}</code>. "
        f"{html.escape(result.best_note)}</p></body></html>"
    )


def calibration_markdown(report: CalibrationReport) -> str:
    """Markdown summary of a judge calibration."""
    labels = list(report.confusion)
    rows = [[gold, *[str(report.confusion[gold][p]) for p in labels]] for gold in labels]
    return "\n".join(
        [
            f"## Judge calibration: `{report.judge}` (n={report.n})",
            f"Exact agreement {report.accuracy:.2f}, supported-vs-not agreement {report.binary_accuracy:.2f}, "
            f"Cohen's kappa {report.cohen_kappa:.2f}.",
            "",
            _md_table(["label \\ judge", *labels], rows),
            "",
        ]
    )


def write_reports(
    out_dir: str | Path, stem: str, markdown: str, html_text: str, json_text: str
) -> list[Path]:
    """Write ``<stem>.md``, ``<stem>.html`` and ``<stem>.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.md", out / f"{stem}.html", out / f"{stem}.json"]
    for path, text in zip(paths, (markdown, html_text, json_text), strict=True):
        path.write_text(text, encoding="utf-8")
    return paths
