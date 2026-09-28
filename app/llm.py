"""The only place that talks to an LLM. Structured output, token/cost logging, retries, timeout."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import Settings, settings as default_settings

log = logging.getLogger("llm")

# USD per 1M tokens (input, output). Anthropic first-party rates.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
}

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.model = other.model or self.model
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.latency_ms += other.latency_ms
        self.calls += other.calls


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICING.get(model, PRICING["claude-haiku-4-5"])
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class LLMClient:
    cfg: Settings = field(default_factory=lambda: default_settings)
    client: Any = None  # injectable (tests pass a fake with .messages.parse)
    totals: Usage = field(default_factory=Usage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.client is None and self.cfg.pipeline_mode != "rules" and self.has_credentials():
            import anthropic

            self.client = anthropic.Anthropic(timeout=self.cfg.llm_timeout_s, max_retries=self.cfg.llm_max_retries)

    @staticmethod
    def has_credentials() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))

    @property
    def available(self) -> bool:
        return self.client is not None and self.cfg.pipeline_mode != "rules"

    @property
    def model(self) -> str:
        return self.cfg.llm_model

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4096) -> tuple[T, Usage]:
        if not self.available:
            raise LLMError("LLM not available (no credentials or PIPELINE_MODE=rules)")
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except Exception as exc:  # SDK already retried transient errors
            raise LLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000
        if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
            raise LLMError(f"LLM stopped with {resp.stop_reason}")
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise LLMError("LLM returned no parsable structured output")
        u = getattr(resp, "usage", None)
        it = int(getattr(u, "input_tokens", 0) or 0)
        ot = int(getattr(u, "output_tokens", 0) or 0)
        usage = Usage(self.model, it, ot, cost_for(self.model, it, ot), latency, 1)
        with self._lock:
            self.totals.add(usage)
        log.info(
            "llm schema=%s model=%s in=%d out=%d cost=$%.5f latency=%.0fms",
            schema.__name__, self.model, it, ot, usage.cost_usd, latency,
        )
        return parsed, usage
