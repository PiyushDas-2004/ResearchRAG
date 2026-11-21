"""Process-level settings, read from environment variables (prefix ``RR_``) and ``.env``.

Settings hold deployment concerns (database URL, auth, limits, telemetry). Retrieval and
generation behaviour lives in :class:`researchrag.config.pipeline.PipelineConfig`, which is
loaded from YAML so it can be swept by the evaluation harness.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API, CLI and background workers."""

    model_config = SettingsConfigDict(env_prefix="RR_", env_file=".env", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "postgresql+asyncpg://researchrag:researchrag@localhost:5432/researchrag"
    db_pool_size: int = 5
    db_max_overflow: int = 5
    pipeline_config: str = "configs/default.yaml"
    pricing_file: str = "configs/pricing.yaml"

    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    allow_anonymous: bool = False
    rate_limit_per_minute: int = 60
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_request_bytes: int = 30 * 1024 * 1024

    max_upload_bytes: int = 25 * 1024 * 1024
    max_pages: int = 200
    max_extracted_chars: int = 2_000_000
    parse_timeout_s: float = 60.0
    parse_memory_limit_mb: int = 1024
    parse_isolation: Literal["subprocess", "inline"] = "subprocess"
    ocr_enabled: bool = False

    run_worker: bool = True
    worker_poll_interval_s: float = 1.0
    job_max_retries: int = 3
    job_stale_after_s: float = 300.0
    run_migrations_on_startup: bool = False

    log_level: str = "INFO"
    log_json: bool = True
    log_prompts: bool = False
    redact_pii: bool = True
    store_prompts_in_traces: bool = True
    otel_exporter_endpoint: str = ""
    otel_service_name: str = "researchrag"
    otel_console: bool = False

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def auth_enabled(self) -> bool:
        """Auth is enforced whenever keys are configured, and always outside dev."""
        return bool(self.api_keys) or (self.environment == "prod" and not self.allow_anonymous)

    def sync_database_url(self) -> str:
        """Return the database URL with the psycopg driver, for tools that need sync access."""
        return self.database_url.replace("+asyncpg", "+psycopg")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings."""
    return Settings()
