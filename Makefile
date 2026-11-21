# Developer entry points. Everything works offline (fake embedder + mock LLM).
PY ?= python
export RR_TEST_DATABASE_URL ?= postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag_test

.PHONY: help install lint format type test test-unit test-fast cov security audit corpus eval-gate up down migrate serve docker loadtest tf-validate check

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package with dev extras (editable)
	$(PY) -m pip install -e ".[dev]"

lint:  ## Ruff lint + format check
	ruff check src tests scripts loadtest
	ruff format --check src tests scripts loadtest

format:  ## Auto-format and fix lint
	ruff format src tests scripts loadtest
	ruff check --fix src tests scripts loadtest

type:  ## mypy --strict on src/
	mypy --strict src/researchrag

test:  ## Full test suite with coverage gate (needs Postgres for db tests)
	$(PY) -m pytest --cov=researchrag --cov-report=term-missing --cov-report=xml

test-unit:  ## Unit tests only (no database)
	$(PY) -m pytest tests/unit

security:  ## Bandit static analysis
	bandit -q -r src -c pyproject.toml

audit:  ## Dependency vulnerability audit
	pip-audit --skip-editable

corpus:  ## Render the synthetic corpus PDFs into data/corpus/pdfs
	researchrag corpus build

eval-gate:  ## Offline evaluation regression gate against the stored baseline
	researchrag eval check --baseline main

up:  ## docker compose up (API + Postgres/pgvector)
	docker compose up --build -d

down:  ## Stop the compose stack
	docker compose down

migrate:  ## Apply database migrations
	researchrag db migrate

serve:  ## Run the API locally with reload
	researchrag serve --reload

docker:  ## Build the production image
	docker build -t researchrag:local .

loadtest:  ## Small locust load test against a running server (offline stack)
	locust -f loadtest/locustfile.py --headless -u 10 -r 5 -t 60s --host http://localhost:8000

tf-validate:  ## terraform fmt + validate
	cd infra/terraform && terraform fmt -check -recursive && terraform init -backend=false -input=false >/dev/null && terraform validate

check: lint type security test  ## Everything CI runs (except docker/terraform)
