"""User-editable price table for cost accounting.

Prices change often and differ by region and contract, so they are never hardcoded: they
live in ``configs/pricing.yaml`` together with an ``as_of`` date. Unknown models cost zero
and are reported as ``unpriced`` so a missing entry is visible rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from researchrag.errors import ValidationFailedError


class ModelPrice(BaseModel):
    """USD price per one million tokens."""

    input_per_mtok: float = Field(0.0, ge=0)
    output_per_mtok: float = Field(0.0, ge=0)


class PricingTable(BaseModel):
    """Prices keyed by ``provider/model``."""

    as_of: str = "unknown"
    note: str = "Verify prices with the provider before relying on these numbers."
    currency: str = "USD"
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    def cost(self, model_id: str, input_tokens: int, output_tokens: int = 0) -> tuple[float, bool]:
        """Return ``(cost_usd, priced)``; ``priced`` is False when the model is not in the table."""
        price = self.models.get(model_id)
        if price is None:
            return 0.0, False
        usd = input_tokens / 1e6 * price.input_per_mtok + output_tokens / 1e6 * price.output_per_mtok
        return usd, True


def load_pricing(path: str | Path | None) -> PricingTable:
    """Load the pricing table; a missing file yields an empty table."""
    if path is None or not Path(path).exists():
        return PricingTable()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return PricingTable.model_validate(raw)
    except ValueError as exc:
        raise ValidationFailedError(f"Invalid pricing file {path}: {exc}") from exc


@dataclass
class CostLedger:
    """Accumulates token usage and cost for one query or evaluation run."""

    pricing: PricingTable
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_models: set[str] = field(default_factory=set)
    by_stage: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, model_id: str, input_tokens: int, output_tokens: int = 0) -> float:
        """Record one model call and return its cost."""
        usd, priced = self.pricing.cost(model_id, input_tokens, output_tokens)
        if not priced:
            self.unpriced_models.add(model_id)
        if stage == "embed":
            self.embedding_tokens += input_tokens
        else:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
        self.cost_usd += usd
        self.by_stage[stage] = self.by_stage.get(stage, 0.0) + usd
        return usd
