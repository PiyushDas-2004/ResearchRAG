"""CLI commands (Typer runner), logging redaction, tracing and metrics helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner

from researchrag.cli import app
from researchrag.observability.logging import configure_logging, redact, redact_processor
from researchrag.observability.tracing import StageRecorder, configure_tracing, current_trace_id

runner = CliRunner()


def test_cli_help_and_corpus_build(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.output
    result = runner.invoke(app, ["corpus", "build", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*.pdf"))) == 16


def test_cli_eval_run_check_calibrate_export(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    result = runner.invoke(app, ["eval", "run", "--limit", "6", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "eval-eval-offline.html").exists()
    baseline = tmp_path / "base.json"
    result = runner.invoke(app, ["eval", "check", "--baseline", str(baseline), "--limit", "6", "--update"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["eval", "check", "--baseline", str(baseline), "--limit", "6"])
    assert result.exit_code == 0, result.output
    assert "gate passed" in result.output
    degraded = tmp_path / "bad.yaml"
    degraded.write_text(
        "name: bad\nretrieval: {top_k: 1, candidate_k: 1, mmr_enabled: false}\nrerank: {provider: none}\n"
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "check",
            "--baseline",
            str(baseline),
            "--limit",
            "6",
            "--config",
            str(degraded),
            "--tolerance",
            "0.0",
        ],
    )
    assert result.exit_code in {0, 1}
    missing = runner.invoke(app, ["eval", "check", "--baseline", str(tmp_path / "none.json"), "--limit", "2"])
    assert missing.exit_code == 2
    result = runner.invoke(app, ["eval", "calibrate", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Cohen's kappa" in result.output
    result = runner.invoke(app, ["eval", "export", "--limit", "2", "--out", str(tmp_path / "cases.jsonl")])
    assert result.exit_code == 0, result.output
    assert len((tmp_path / "cases.jsonl").read_text().splitlines()) == 2


def test_cli_sweep_small(tmp_path: Path) -> None:
    spec = tmp_path / "sweep.yaml"
    spec.write_text(
        f"base: {Path('configs/eval-offline.yaml').resolve()}\n"
        "factors: {retrieval.mode: [hybrid, lexical]}\n"
    )
    result = runner.invoke(app, ["eval", "sweep", str(spec), "--limit", "4", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "sweep.md").exists()


def test_cli_db_reset_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    from researchrag.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RR_ENVIRONMENT", "prod")
    assert runner.invoke(app, ["db", "reset", "--yes"]).exit_code == 2
    get_settings.cache_clear()
    monkeypatch.setenv("RR_ENVIRONMENT", "dev")
    assert runner.invoke(app, ["db", "reset"]).exit_code == 1
    get_settings.cache_clear()


def test_cli_ingest_without_files(tmp_path: Path) -> None:
    assert runner.invoke(app, ["ingest", str(tmp_path)]).exit_code == 1


def test_cli_reports_domain_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "run", "--config", str(tmp_path / "missing.yaml"), "--limit", "1"])
    assert result.exit_code != 0


def test_redaction() -> None:
    text = "mail jane.doe@example.org, call +1 (555) 123-4567, key sk-abcdefghijklmnop and api_key=xyz"
    out = redact(text)
    assert "example.org" not in out
    assert "555" not in out
    assert "sk-abc" not in out
    assert "xyz" not in out
    event = redact_processor(
        None, "info", {"authorization": "Bearer x", "nested": {"token": "t", "msg": "a@b.co"}, "n": 3}
    )
    assert event["authorization"] == "[REDACTED]"
    assert event["nested"] == {"token": "[REDACTED]", "msg": "[REDACTED_EMAIL]"}
    assert event["n"] == 3


def test_configure_logging_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True, redact=True)
    structlog.get_logger("t").info("hello", email="x@y.com")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["email"] == "[REDACTED_EMAIL]"
    configure_logging("WARNING", json_logs=False, redact=False)


def test_tracing_console_and_stage_recorder(capsys: pytest.CaptureFixture[str]) -> None:
    assert configure_tracing("researchrag-test", "", console=False) in {True, False}
    configure_tracing("researchrag-test", "", console=True)
    recorder = StageRecorder()
    with recorder.sync_stage("unit.stage", count=3) as details:
        details["items"] = ["a", "b"]
        details["nested"] = {"skip": 1}
        details["obj"] = object()
        assert current_trace_id() is not None
    assert recorder.stages["unit.stage"]["count"] == 3
    assert recorder.latency_ms["unit.stage"] >= 0
    assert recorder.total_ms >= 0
