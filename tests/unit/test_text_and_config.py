"""Text utilities, cleaning, section detection, configuration, pricing and upload validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from researchrag.config.pipeline import PipelineConfig, load_pipeline_config
from researchrag.config.pricing import CostLedger, PricingTable, load_pricing
from researchrag.config.settings import Settings
from researchrag.errors import UploadRejectedError, ValidationFailedError
from researchrag.evaluation.synthetic import build_corpus, load_papers, render_paper
from researchrag.ingestion.cleaning import fingerprint, is_page_number, join_lines, repeating_margin_texts
from researchrag.ingestion.isolation import parse_document
from researchrag.ingestion.metadata import split_authors
from researchrag.ingestion.pdf_parser import ParseLimits
from researchrag.ingestion.sections import heading_label, heading_title
from researchrag.security.upload import UploadPolicy, sanitize_filename, sniff_pdf, validate_upload
from researchrag.textutils import (
    collapse_whitespace,
    content_terms,
    coverage,
    jaccard,
    normalize_unicode,
    numbers,
    split_sentences,
    stem,
)
from tests.conftest import PAPERS_DIR, paper_pdf

ROOT = Path(__file__).resolve().parents[2]


def test_normalize_unicode_ligatures_and_dashes() -> None:
    assert normalize_unicode("ﬁne\u2013tuned \u201cquote\u201d") == 'fine-tuned "quote"'


def test_stem_and_terms() -> None:
    assert stem("retrievals") == stem("retrieval")
    assert stem("studies") == "study"
    assert stem("class") == "class"
    assert "the" not in content_terms("The model retrieves the passages")
    assert coverage(["a", "b"], {"a"}) == 0.5
    assert coverage([], {"a"}) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, {"a", "b"}) == 0.5


def test_numbers_normalised() -> None:
    assert numbers("rose from 1,200 to 0.50 and -3.10") == {"1200", "0.5", "-3.1"}
    assert numbers("nDCG@10") == {"10"}


def test_split_sentences_with_abbreviations() -> None:
    text = "We used e.g. BM25 et al. methods. Recall was 0.61! Next, 3 models ran."
    spans = split_sentences(text, offset=10)
    assert [s.text for s in spans] == [
        "We used e.g. BM25 et al. methods.",
        "Recall was 0.61!",
        "Next, 3 models ran.",
    ]
    assert all(text[s.start - 10 : s.end - 10] == s.text for s in spans)


@given(st.text(alphabet="abc .!?ABC\n", max_size=200))
def test_split_sentences_spans_map_back(text: str) -> None:
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text
        assert span.text.strip() == span.text


def test_cleaning_helpers() -> None:
    assert join_lines(["The trans-", "former works"]) == "The transformer works"
    assert join_lines(["ﬁne"]) == "fine"
    assert is_page_number("Page 3 of 12")
    assert is_page_number("17")
    assert not is_page_number("Table 3")
    assert fingerprint("Journal 2023, p. 4") == fingerprint("Journal 2023, p. 5")
    pages = [["Running head", "1"], ["Running head", "2"], ["Running head", "3"]]
    assert fingerprint("Running head") in repeating_margin_texts(pages)
    assert repeating_margin_texts([["only one page"]]) == set()


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("1 Introduction", "introduction"),
        ("2.1 Methods", "methods"),
        ("References", "references"),
        ("IV. Results", "results"),
        ("3 Site Description and Data", "other"),
        ("5 cores were taken", None),
        ("1.71 V for NF-ref", "other"),
        ("This is a long sentence that is definitely not a heading at all, ok", None),
    ],
)
def test_heading_label(text: str, label: str | None) -> None:
    assert heading_label(text) == label


def test_heading_title_and_authors() -> None:
    assert heading_title("2.1 Methods:") == "Methods"
    assert split_authors("Ana Lima, Bo Chen and Cy Dube") == ["Ana Lima", "Bo Chen", "Cy Dube"]
    assert split_authors("University of Somewhere 2021") == []


def test_pipeline_config_overrides_and_hash(tmp_path: Path) -> None:
    base = PipelineConfig()
    changed = base.with_overrides({"retrieval.top_k": 3, "rerank.provider": "none"}, name="x")
    assert changed.retrieval.top_k == 3 and changed.name == "x"
    assert changed.config_hash() != base.config_hash()
    assert base.with_overrides({}, name="renamed").config_hash() == base.config_hash()
    with pytest.raises(ValidationFailedError):
        base.with_overrides({"retrieval.nope": 1})
    with pytest.raises(ValidationFailedError):
        base.with_overrides({"nosection.x": 1})
    with pytest.raises(ValidationFailedError):
        base.with_overrides({"retrieval.top_k": -1})
    path = tmp_path / "cfg.yaml"
    path.write_text("retrieval:\n  mode: vector\n")
    loaded = load_pipeline_config(path)
    assert loaded.retrieval.mode == "vector" and loaded.name == "cfg"
    assert load_pipeline_config(None) == PipelineConfig()
    with pytest.raises(ValidationFailedError):
        load_pipeline_config(tmp_path / "missing.yaml")
    path.write_text("- a list\n")
    with pytest.raises(ValidationFailedError):
        load_pipeline_config(path)
    path.write_text("retrieval:\n  bogus: 1\n")
    with pytest.raises(ValidationFailedError):
        load_pipeline_config(path)


def test_shipped_configs_are_valid() -> None:
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if path.name in {"pricing.yaml", "sweep.yaml"} or path.name.startswith("sweep"):
            continue
        load_pipeline_config(path)
    pricing = load_pricing(ROOT / "configs" / "pricing.yaml")
    assert pricing.as_of and "erify" in pricing.note


def test_pricing_and_ledger(tmp_path: Path) -> None:
    table = PricingTable.model_validate(
        {"models": {"openai/x": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}}
    )
    assert table.cost("openai/x", 1_000_000, 500_000) == (2.0, True)
    assert table.cost("unknown/y", 10) == (0.0, False)
    ledger = CostLedger(table)
    ledger.record("generate", "openai/x", 1000, 1000)
    ledger.record("embed", "unknown/y", 50)
    assert ledger.input_tokens == 1000 and ledger.embedding_tokens == 50
    assert ledger.unpriced_models == {"unknown/y"}
    assert ledger.cost_usd == pytest.approx(0.003)
    assert load_pricing(tmp_path / "missing.yaml").models == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: {a: {input_per_mtok: -1}}")
    with pytest.raises(ValidationFailedError):
        load_pricing(bad)


def test_settings_parse_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RR_API_KEYS", "k1, k2")
    monkeypatch.setenv("RR_CORS_ORIGINS", "http://a,http://b")
    settings = Settings()
    assert [k.get_secret_value() for k in settings.api_keys] == ["k1", "k2"]
    assert settings.cors_origins == ["http://a", "http://b"]
    assert settings.auth_enabled
    assert "+psycopg" in settings.sync_database_url()
    assert not Settings(api_keys=[], environment="dev").auth_enabled
    assert Settings(api_keys=[], environment="prod").auth_enabled


def test_upload_validation(sample_pdf: bytes) -> None:
    policy = UploadPolicy(max_bytes=len(sample_pdf) + 10)
    assert validate_upload(sample_pdf, "../../etc/paper.pdf", policy) == "paper.pdf"
    assert sniff_pdf(b"junk" + sample_pdf)
    with pytest.raises(UploadRejectedError) as exc:
        validate_upload(sample_pdf + b"x" * 100, "a.pdf", policy)
    assert exc.value.detail["reason"] == "too_large"
    with pytest.raises(UploadRejectedError) as exc:
        validate_upload(b"PK\x03\x04" + b"0" * 200, "paper.pdf", policy)
    assert exc.value.detail["reason"] == "not_pdf"
    with pytest.raises(UploadRejectedError) as exc:
        validate_upload(b"%PDF", "x.pdf", policy)
    assert exc.value.detail["reason"] == "too_small"
    assert sanitize_filename(None) == "upload.pdf"
    assert sanitize_filename("C:\\docs\\my<paper>.pdf") == "my_paper_.pdf"
    assert sanitize_filename("...") == "upload.pdf"


@pytest.mark.slow
def test_isolated_parse_success_and_failure(sample_pdf: bytes) -> None:
    doc = parse_document(sample_pdf, ParseLimits(), isolation="subprocess", timeout_s=60, memory_mb=1024)
    assert doc.metadata.year == 2023
    with pytest.raises(UploadRejectedError) as exc:
        parse_document(
            b"%PDF-1.4\n" + b"\x00" * 300, ParseLimits(), isolation="subprocess", timeout_s=60, memory_mb=1024
        )
    assert exc.value.detail["reason"] == "malformed"
    with pytest.raises(UploadRejectedError) as exc:
        parse_document(sample_pdf, ParseLimits(), isolation="subprocess", timeout_s=0.001, memory_mb=1024)
    assert exc.value.detail["reason"] == "parse_timeout"
    inline = parse_document(sample_pdf, ParseLimits(), isolation="inline", timeout_s=1, memory_mb=0)
    assert inline.text == doc.text


def test_corpus_generation_is_deterministic(tmp_path: Path) -> None:
    first = build_corpus(PAPERS_DIR, tmp_path / "a")
    second = build_corpus(PAPERS_DIR, tmp_path / "b")
    assert first == second and len(first) == len(load_papers(PAPERS_DIR))
    assert hashlib.sha256(paper_pdf("ir-01")).hexdigest() == first["ir-01"]


def test_render_with_extra_paragraphs() -> None:
    paper = load_papers(PAPERS_DIR)[0]
    data = render_paper(
        paper, extra_paragraphs=[("Appendix Notes", "An appended note."), ("Introduction", "Extra.")]
    )
    assert data.startswith(b"%PDF")
    assert collapse_whitespace("An appended note.")
