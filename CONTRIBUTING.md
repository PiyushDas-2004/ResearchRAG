# Contributing

Thanks for helping improve ResearchRAG.

## Setup

```bash
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"
pre-commit install
docker compose up -d db             # PostgreSQL 16 + pgvector for db tests
export RR_TEST_DATABASE_URL=postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag
```

## Before opening a pull request

```bash
make lint type test          # ruff, mypy --strict, pytest with the 85% coverage gate
researchrag eval check --baseline main   # offline evaluation regression gate
```

- Keep changes focused; use [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`).
- New behaviour needs tests; retrieval or generation changes should show their effect with
  `researchrag eval run` (and update the baseline with `--update` only when the change is an
  intended improvement, explained in the PR).
- Core pipeline modules must not import provider SDKs (enforced by ruff `TID251`).
- Never commit secrets, `.env` files, database dumps or third-party PDFs. Test documents are
  generated from the synthetic YAML specs in `data/corpus/papers`.
- Numbers in documentation must come from a real run and say which components produced them.

## Adding a provider

Implement the `Embedder` or `LLMClient` protocol in `src/researchrag/embeddings/` or
`src/researchrag/llm/`, map SDK errors onto `researchrag.errors`, add unit tests with a fake
client, and a `@pytest.mark.live` test that only runs when the key is present.
