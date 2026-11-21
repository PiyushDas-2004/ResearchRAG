"""``researchrag`` command-line interface (Typer)."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from researchrag.config.pipeline import PipelineConfig, load_pipeline_config
from researchrag.config.pricing import load_pricing
from researchrag.config.settings import Settings, get_settings
from researchrag.errors import ResearchRAGError

T = TypeVar("T")
app = typer.Typer(help="ResearchRAG: citation-grounded QA over academic papers.", no_args_is_help=True)


@app.callback()
def _setup(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every query (JSON)")] = False,
) -> None:
    """Configure logging for CLI commands (quiet by default)."""
    from researchrag.observability.logging import configure_logging  # noqa: PLC0415

    settings = get_settings()
    configure_logging(
        "INFO" if verbose else "WARNING", json_logs=settings.log_json, redact=settings.redact_pii
    )


db_app = typer.Typer(help="Database management.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness.", no_args_is_help=True)
corpus_app = typer.Typer(help="Synthetic corpus.", no_args_is_help=True)
feedback_app = typer.Typer(help="User feedback.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(eval_app, name="eval")
app.add_typer(corpus_app, name="corpus")
app.add_typer(feedback_app, name="feedback")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Pipeline config YAML")]
BackendOpt = Annotated[str, typer.Option("--backend", help="memory (no database) or postgres")]


def _run(coro_fn: Callable[[], Awaitable[T]]) -> T:
    try:
        return asyncio.run(coro_fn())  # type: ignore[arg-type]
    except ResearchRAGError as exc:
        typer.secho(f"error ({exc.code}): {exc.message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc


def _config(path: Path | None, settings: Settings) -> PipelineConfig:
    return load_pipeline_config(
        path or (settings.pipeline_config if Path(settings.pipeline_config).exists() else None)
    )


@dataclass
class _Db:
    settings: Settings
    store: Any
    backend: Any
    traces: Any
    conversations: Any
    feedback: Any


@asynccontextmanager
async def _database(settings: Settings) -> AsyncIterator[_Db]:
    from researchrag.storage.db import create_engine, session_factory  # noqa: PLC0415
    from researchrag.storage.repositories import (  # noqa: PLC0415
        PgConversationStore,
        PgFeedbackRepository,
        PgTraceRepository,
    )
    from researchrag.storage.search import PgSearchBackend  # noqa: PLC0415
    from researchrag.storage.store import PgDocumentStore  # noqa: PLC0415

    engine = create_engine(settings)
    factory = session_factory(engine)
    try:
        yield _Db(
            settings,
            PgDocumentStore(factory),
            PgSearchBackend(factory),
            PgTraceRepository(factory),
            PgConversationStore(factory),
            PgFeedbackRepository(factory),
        )
    finally:
        await engine.dispose()


# --------------------------------------------------------------------------- serve
@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "0.0.0.0",  # noqa: S104 - container default
    port: Annotated[int, typer.Option(envvar="PORT", help="Port (Cloud Run sets $PORT)")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload (development)")] = False,
    workers: Annotated[int, typer.Option(help="Worker processes")] = 1,
) -> None:
    """Run the API server."""
    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        "researchrag.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        proxy_headers=True,
        timeout_graceful_shutdown=20,
        access_log=False,
    )


# --------------------------------------------------------------------------- corpus
@corpus_app.command("build")
def corpus_build(
    papers: Annotated[Path, typer.Option(help="Paper YAML directory")] = Path("data/corpus/papers"),
    out: Annotated[Path, typer.Option(help="Output directory for PDFs")] = Path("data/corpus/pdfs"),
) -> None:
    """Render the synthetic papers to PDFs (deterministic)."""
    from researchrag.evaluation.synthetic import build_corpus  # noqa: PLC0415

    hashes = build_corpus(papers, out)
    for key, digest in hashes.items():
        typer.echo(f"{key}.pdf  sha256:{digest[:16]}")
    typer.echo(f"{len(hashes)} PDFs written to {out}")


# --------------------------------------------------------------------------- ingest / query
@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="PDF files or directories")],
    config: ConfigOpt = None,
    collection: Annotated[str | None, typer.Option(help="Collection label")] = None,
) -> None:
    """Ingest PDFs into PostgreSQL."""
    settings = get_settings()
    files = sorted(p for path in paths for p in ([path] if path.is_file() else path.glob("*.pdf")))
    if not files:
        typer.secho("no PDF files found", fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    async def go() -> None:
        from researchrag.factory import build_stack  # noqa: PLC0415
        from researchrag.ingestion.pdf_parser import ParseLimits  # noqa: PLC0415
        from researchrag.ingestion.service import ParseOptions  # noqa: PLC0415
        from researchrag.security.upload import UploadPolicy  # noqa: PLC0415

        async with _database(settings) as db:
            parse = ParseOptions(
                ParseLimits(settings.max_pages, settings.max_extracted_chars, settings.ocr_enabled),
                settings.parse_isolation,
                settings.parse_timeout_s,
                settings.parse_memory_limit_mb,
                UploadPolicy(settings.max_upload_bytes),
            )
            stack = await build_stack(
                _config(config, settings),
                store=db.store,
                backend=db.backend,
                traces=db.traces,
                conversations=db.conversations,
                pricing=load_pricing(settings.pricing_file),
                parse=parse,
            )
            for file in files:
                try:
                    result = await stack.ingestion.ingest_bytes(
                        file.read_bytes(), file.name, collection=collection
                    )
                except ResearchRAGError as exc:
                    typer.secho(f"REJECTED {file.name}: {exc.message}", fg=typer.colors.RED)
                    continue
                status = "ingested" if result.created else "unchanged"
                typer.echo(f"{status:9} {file.name} -> {result.document_id} ({result.chunk_count} chunks)")

    _run(go)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    config: ConfigOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full JSON result")] = False,
) -> None:
    """Answer a question against the indexed papers (PostgreSQL)."""
    settings = get_settings()

    async def go() -> None:
        from researchrag.factory import build_stack  # noqa: PLC0415
        from researchrag.pipeline import QueryRequest  # noqa: PLC0415

        async with _database(settings) as db:
            stack = await build_stack(
                _config(config, settings),
                store=db.store,
                backend=db.backend,
                traces=db.traces,
                conversations=db.conversations,
                pricing=load_pricing(settings.pricing_file),
            )
            result = await stack.pipeline.answer(QueryRequest(question))
        if as_json:
            typer.echo(result.model_dump_json(indent=2))
            return
        if result.abstained:
            typer.secho(result.abstain_reason or "Not enough evidence.", fg=typer.colors.YELLOW)
        else:
            typer.echo(result.answer)
            for c in result.citations:
                mark = "verified" if c.quote_verified else "UNVERIFIED"
                typer.echo(f"  [{c.marker}] {c.title}, p.{c.page} ({mark}): {c.quote or ''}")
        typer.echo(f"trace {result.trace_id}  faithfulness={result.faithfulness}  {result.latency_ms:.0f} ms")

    _run(go)


# --------------------------------------------------------------------------- db
@db_app.command("migrate")
def db_migrate() -> None:
    """Apply migrations (guarded by a Postgres advisory lock)."""
    from researchrag.storage.migrate import migrate  # noqa: PLC0415

    revision = _run(lambda: migrate(get_settings().database_url))
    typer.echo(f"database at revision {revision}")


@db_app.command("reset")
def db_reset(yes: Annotated[bool, typer.Option("--yes", help="Confirm destructive reset")] = False) -> None:
    """Drop and recreate the schema (development only)."""
    settings = get_settings()
    if settings.environment == "prod":
        typer.secho("refusing to reset a prod database", fg=typer.colors.RED)
        raise typer.Exit(2)
    if not yes:
        typer.secho("add --yes to drop all data", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    from researchrag.storage.migrate import reset  # noqa: PLC0415

    _run(lambda: reset(settings.database_url))
    typer.echo("database reset and migrated")


@db_app.command("stats")
def db_stats() -> None:
    """Row counts, embedding versions and index sizes."""
    settings = get_settings()

    async def go() -> dict[str, Any]:
        from sqlalchemy import text  # noqa: PLC0415

        from researchrag.storage.db import create_engine, pgvector_version, session_factory  # noqa: PLC0415

        engine = create_engine(settings)
        try:
            async with session_factory(engine)() as session:
                counts = {}
                for table in (
                    "documents",
                    "chunks",
                    "chunk_embeddings",
                    "query_traces",
                    "feedback",
                    "ingestion_jobs",
                ):
                    counts[table] = (await session.execute(text(f"SELECT count(*) FROM {table}"))).scalar()  # noqa: S608
                versions = [
                    dict(r._mapping)
                    for r in (
                        await session.execute(
                            text(
                                "SELECT v.id, v.model_id, v.dimension, count(e.chunk_id) AS vectors "
                                "FROM embedding_versions v "
                                "LEFT JOIN chunk_embeddings e ON e.version_id = v.id "
                                "GROUP BY v.id ORDER BY v.id"
                            )
                        )
                    ).all()
                ]
                indexes = [
                    dict(r._mapping)
                    for r in (
                        await session.execute(
                            text(
                                "SELECT indexrelname AS index, "
                                "pg_size_pretty(pg_relation_size(indexrelid)) AS size "
                                "FROM pg_stat_user_indexes "
                                "WHERE relname IN ('chunks', 'chunk_embeddings') ORDER BY 1"
                            )
                        )
                    ).all()
                ]
                pgv = ".".join(map(str, await pgvector_version(session)))
            return {"pgvector": pgv, "counts": counts, "embedding_versions": versions, "indexes": indexes}
        finally:
            await engine.dispose()

    typer.echo(json.dumps(_run(go), indent=2, default=str))


# --------------------------------------------------------------------------- eval
@asynccontextmanager
async def _eval_env(backend: str, papers: Path) -> AsyncIterator[Any]:
    from researchrag.evaluation.runner import EvalEnvironment  # noqa: PLC0415

    if backend == "memory":
        yield EvalEnvironment.memory(papers)
        return
    if backend != "postgres":
        raise typer.BadParameter("backend must be memory or postgres")
    async with _database(get_settings()) as db:
        yield EvalEnvironment(db.store, db.backend, "postgres", papers)


def _runner(env: Any, dataset_dir: Path, limit: int | None) -> Any:
    from researchrag.evaluation.dataset import load_dataset  # noqa: PLC0415
    from researchrag.evaluation.runner import EvalRunner  # noqa: PLC0415

    dataset = load_dataset(dataset_dir).subset(limit)
    return EvalRunner(env, dataset, pricing=load_pricing(get_settings().pricing_file))


DatasetOpt = Annotated[Path, typer.Option("--dataset", help="Directory with questions_*.jsonl")]
PapersOpt = Annotated[Path, typer.Option("--papers", help="Directory with paper YAML specs")]
OutOpt = Annotated[Path, typer.Option("--out", help="Report output directory")]
LimitOpt = Annotated[int | None, typer.Option("--limit", help="Only the first N questions")]


@eval_app.command("run")
def eval_run(
    config: ConfigOpt = Path("configs/eval-offline.yaml"),
    backend: BackendOpt = "memory",
    dataset: DatasetOpt = Path("data/eval"),
    papers: PapersOpt = Path("data/corpus/papers"),
    out: OutOpt = Path("reports"),
    limit: LimitOpt = None,
) -> None:
    """Evaluate one configuration; writes Markdown, HTML and JSON reports."""
    from researchrag.evaluation.report import run_html, run_markdown, write_reports  # noqa: PLC0415

    async def go() -> Any:
        async with _eval_env(backend, papers) as env:
            return await _runner(env, dataset, limit).run(_config(config, get_settings()))

    run = _run(go)
    paths = write_reports(
        out, f"eval-{run.name}", run_markdown(run), run_html(run), run.model_dump_json(indent=2)
    )
    for metric in ("recall@5", "faithfulness", "answer_correctness", "abstention_accuracy"):
        est = run.summary.get(metric)
        if est:
            typer.echo(f"{metric:22} {est['mean']:.3f}  [{est['low']:.3f}, {est['high']:.3f}]  n={est['n']}")
    typer.echo("reports: " + ", ".join(str(p) for p in paths))


@eval_app.command("sweep")
def eval_sweep(
    spec: Annotated[Path, typer.Argument(help="Sweep YAML")] = Path("configs/sweep.yaml"),
    backend: BackendOpt = "memory",
    dataset: DatasetOpt = Path("data/eval"),
    papers: PapersOpt = Path("data/corpus/papers"),
    out: OutOpt = Path("reports"),
    limit: LimitOpt = None,
) -> None:
    """Run an ablation sweep with paired bootstrap comparisons against the base config."""
    from researchrag.evaluation.report import sweep_html, sweep_markdown, write_reports  # noqa: PLC0415
    from researchrag.evaluation.sweep import load_sweep, run_sweep  # noqa: PLC0415

    sweep_spec, base = load_sweep(spec)

    async def go() -> Any:
        async with _eval_env(backend, papers) as env:
            return await run_sweep(_runner(env, dataset, limit), sweep_spec, base)

    result = _run(go)
    paths = write_reports(
        out, "sweep", sweep_markdown(result), sweep_html(result), result.model_dump_json(indent=2)
    )
    typer.echo(
        f"{len(result.runs)} configurations; best quality per dollar: {result.best_quality_per_dollar}"
    )
    typer.echo("reports: " + ", ".join(str(p) for p in paths))


@eval_app.command("check")
def eval_check(
    baseline: Annotated[
        str, typer.Option("--baseline", help="Baseline name (eval/baselines/<name>.json) or path")
    ] = "main",
    config: ConfigOpt = Path("configs/eval-offline.yaml"),
    backend: BackendOpt = "memory",
    dataset: DatasetOpt = Path("data/eval"),
    papers: PapersOpt = Path("data/corpus/papers"),
    tolerance: Annotated[float, typer.Option(help="Allowed absolute drop before failing")] = 0.03,
    update: Annotated[
        bool, typer.Option("--update", help="Write the current run as the new baseline")
    ] = False,
    limit: LimitOpt = None,
) -> None:
    """Regression gate: exit 1 if retrieval/faithfulness/correctness drop significantly vs the baseline."""
    from researchrag.evaluation.gate import baseline_from_run, check_run, load_baseline, save_baseline  # noqa: PLC0415

    path = Path(baseline) if baseline.endswith(".json") else Path("eval/baselines") / f"{baseline}.json"

    async def go() -> Any:
        async with _eval_env(backend, papers) as env:
            return await _runner(env, dataset, limit).run(_config(config, get_settings()))

    run = _run(go)
    if update:
        save_baseline(baseline_from_run(run), path)
        typer.echo(f"baseline written to {path}")
        return
    if not path.exists():
        typer.secho(f"baseline {path} not found (run with --update to create it)", fg=typer.colors.RED)
        raise typer.Exit(2)
    report = check_run(run, load_baseline(path), tolerance=tolerance)
    for c in report.checks:
        flag = "REGRESSED" if c.regressed else "ok"
        typer.echo(
            f"{c.metric:20} base {c.baseline_mean:.3f} -> {c.current_mean:.3f}  "
            f"diff {c.diff:+.3f} [{c.low:+.3f}, {c.high:+.3f}]  {flag}"
        )
    for warning in report.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW)
    if not report.passed:
        typer.secho("evaluation gate FAILED", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("evaluation gate passed", fg=typer.colors.GREEN)


@eval_app.command("calibrate")
def eval_calibrate(
    config: ConfigOpt = None,
    dataset: DatasetOpt = Path("data/eval"),
    out: OutOpt = Path("reports"),
) -> None:
    """Measure verifier-judge agreement with the hand-labelled calibration set."""
    from researchrag.evaluation.calibration import calibrate  # noqa: PLC0415
    from researchrag.evaluation.dataset import load_calibration  # noqa: PLC0415
    from researchrag.evaluation.report import calibration_markdown  # noqa: PLC0415
    from researchrag.llm.factory import build_llm  # noqa: PLC0415
    from researchrag.verification.judges import build_judge  # noqa: PLC0415

    cfg = _config(config, get_settings())
    judge = build_judge(cfg.verification, build_llm(cfg.generation, cfg.verification.judge_model))
    report = _run(lambda: calibrate(judge, load_calibration(dataset)))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"calibration-{report.judge}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(calibration_markdown(report))


@eval_app.command("export")
def eval_export(
    config: ConfigOpt = Path("configs/eval-offline.yaml"),
    dataset: DatasetOpt = Path("data/eval"),
    papers: PapersOpt = Path("data/corpus/papers"),
    out: Annotated[Path, typer.Option("--out")] = Path("reports/cases.jsonl"),
    limit: LimitOpt = None,
) -> None:
    """Export prompts + references as neutral JSONL (e.g. for LLMBench)."""
    from researchrag.evaluation.dataset import load_dataset  # noqa: PLC0415
    from researchrag.evaluation.export import export_cases, llmbench_available  # noqa: PLC0415
    from researchrag.evaluation.runner import EvalEnvironment  # noqa: PLC0415
    from researchrag.factory import build_stack  # noqa: PLC0415
    from researchrag.ingestion.service import ParseOptions  # noqa: PLC0415
    from researchrag.pipeline import QueryRequest  # noqa: PLC0415
    from researchrag.storage.repositories import MemoryConversationStore, MemoryTraceSink  # noqa: PLC0415

    cfg = _config(config, get_settings())
    data = load_dataset(dataset).subset(limit)

    async def go() -> Path:
        env = EvalEnvironment.memory(papers)
        traces = MemoryTraceSink()
        stack = await build_stack(
            cfg,
            store=env.store,
            backend=env.backend,
            traces=traces,
            conversations=MemoryConversationStore(),
            pricing=load_pricing(get_settings().pricing_file),
            parse=ParseOptions(isolation="inline"),
        )
        for key, pdf in env.pdfs().items():
            await stack.ingestion.ingest_bytes(pdf, f"{key}.pdf")
        by_question: dict[str, dict[str, Any]] = {}
        for q in data.questions:
            result = await stack.pipeline.answer(QueryRequest(q.question))
            by_question[q.id] = (await traces.get(result.trace_id) or {}).get("stages", {})
        return export_cases(data, by_question, cfg.name, out)

    path = _run(go)
    typer.echo(f"wrote {path}")
    if llmbench_available():
        typer.echo(
            "llmbench is installed: point it at this file to benchmark other models on the same cases."
        )


# --------------------------------------------------------------------------- feedback
@feedback_app.command("export")
def feedback_export(
    out: Annotated[Path, typer.Option("--out")] = Path("data/eval/candidates_from_feedback.jsonl"),
    limit: Annotated[int, typer.Option(help="Maximum traces")] = 200,
) -> None:
    """Export thumbs-down traces as candidate eval questions (gold labels need human review)."""
    from researchrag.evaluation.export import feedback_to_candidates  # noqa: PLC0415

    async def go() -> list[dict[str, Any]]:
        async with _database(get_settings()) as db:
            rows: list[dict[str, Any]] = await db.feedback.low_rated(limit)
            return rows

    candidates = feedback_to_candidates(_run(go))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c) + "\n" for c in candidates), encoding="utf-8")
    typer.echo(f"{len(candidates)} candidate question(s) written to {out}")


def main() -> None:  # pragma: no cover - console entry point
    """Entry point."""
    app(sys.argv[1:])
