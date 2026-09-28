"""The only place that talks to an LLM. Structured output, token/cost logging, retries, timeout.

Provider: Meta Model API (Muse Spark) over its Anthropic-Messages-compatible
endpoint. The `anthropic` SDK is pointed at the Meta base host and authenticates
with a Bearer token (MODEL_API_KEY from the dev.meta.ai dashboard).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import Settings, settings as default_settings

log = logging.getLogger("llm")

# USD per 1M tokens (input, output). Meta Model API standard tier.
PRICING: dict[str, tuple[float, float]] = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
    "muse-spark-1.1": (1.25, 4.25),
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
    pin, pout = PRICING.get(model, PRICING["muse-spark-1.3"])
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

            # Meta Model API exposes an Anthropic-Messages-compatible endpoint:
            # point the SDK at the Meta base host (it appends /v1/messages) and
            # authenticate with a Bearer token. NOTE: use auth_token, not
            # api_key — the latter sends an x-api-key header, which the Meta
            # endpoint does not accept.
            self.client = anthropic.Anthropic(
                auth_token=os.getenv("MODEL_API_KEY"),
                base_url=self.cfg.llm_base_url,
                timeout=self.cfg.llm_timeout_s,
                max_retries=self.cfg.llm_max_retries,
            )

    @staticmethod
    def has_credentials() -> bool:
        return bool(os.getenv("MODEL_API_KEY"))

    @property
    def available(self) -> bool:
        return self.client is not None and self.cfg.pipeline_mode != "rules"

    @property
    def model(self) -> str:
        return self.cfg.llm_model

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4096) -> tuple[T, Usage]:
        """Structured output with automatic fallback.

        Prefers the SDK's ``messages.parse`` (structured-output beta). If the
        provider endpoint rejects the beta (e.g. HTTP 400 on ``output_format``),
        retries in JSON mode: the model is instructed to return a bare JSON
        object, which is then validated against the schema here.
        """
        if not self.available:
            raise LLMError("LLM not available (no MODEL_API_KEY or PIPELINE_MODE=rules)")
        try:
            return self._structured_parse(system, user, schema, max_tokens)
        except LLMError as exc:
            msg = str(exc).lower()
            if "output_format" in msg or "not supported" in msg or "400" in msg or "bad request" in msg:
                log.warning("structured-output beta unavailable (%s); falling back to JSON mode", exc)
                return self._structured_json(system, user, schema, max_tokens)
            raise

    def _structured_parse(self, system: str, user: str, schema: type[T], max_tokens: int) -> tuple[T, Usage]:
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

    def _structured_json(self, system: str, user: str, schema: type[T], max_tokens: int) -> tuple[T, Usage]:
        """JSON-mode fallback: ask for a bare JSON object, validate locally."""
        prompt = (
            user
            + "\n\nRespond with a single JSON object that matches the schema described above. "
            + "Return ONLY the JSON object: no markdown fences, no commentary, no other text."
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK already retried transient errors
            raise LLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000
        if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
            raise LLMError(f"LLM stopped with {resp.stop_reason}")
        chunks: list[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                chunks.append(getattr(block, "text", ""))
        text = "".join(chunks).strip()
        if text.startswith("```"):  # tolerate fences even though we asked for none
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except Exception as exc:
            raise LLMError(f"LLM did not return valid JSON: {exc}") from exc
        try:
            parsed = schema.model_validate(data)
        except Exception as exc:
            raise LLMError(f"LLM JSON failed schema validation: {exc}") from exc
        u = getattr(resp, "usage", None)
        it = int(getattr(u, "input_tokens", 0) or 0)
        ot = int(getattr(u, "output_tokens", 0) or 0)
        usage = Usage(self.model, it, ot, cost_for(self.model, it, ot), latency, 1)
        with self._lock:
            self.totals.add(usage)
        log.info(
            "llm schema=%s model=%s in=%d out=%d cost=$%.5f latency=%.0fms (json-mode)",
            schema.__name__, self.model, it, ot, usage.cost_usd, latency,
        )
        return parsed, usage
