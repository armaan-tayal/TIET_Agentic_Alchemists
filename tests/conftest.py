from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.embed import get_models  # noqa: E402


@pytest.fixture(scope="session")
def models():
    return get_models().load()


@pytest.fixture(scope="session")
def catalog(models):
    from app.pipeline.deeplinks import Catalog

    return Catalog(settings.deeplinks_path, models)


@pytest.fixture(scope="session")
def siis_rows() -> list[dict[str, Any]]:
    return json.loads((ROOT / "data" / "siis_responses.json").read_text(encoding="utf-8"))["responses"]


class FakeMessages:
    """Stands in for anthropic.Anthropic().messages; routes on the requested output schema."""

    def __init__(self, handlers: dict[str, Callable[[str, str], Any]]) -> None:
        self.handlers = handlers
        self.calls: list[str] = []

    def parse(self, *, model: str, max_tokens: int, system: str, messages: list, output_format: type) -> Any:
        name = output_format.__name__
        self.calls.append(name)
        parsed = self.handlers[name](system, messages[0]["content"])
        return SimpleNamespace(
            parsed_output=output_format.model_validate(parsed) if isinstance(parsed, dict) else parsed,
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
            stop_reason="end_turn",
        )


@pytest.fixture
def make_engine(models, tmp_path):
    from app.engine import Engine
    from app.llm import LLMClient

    def _make(handlers: dict | None = None, **overrides):
        cfg = replace(settings, cache_db=tmp_path / "cache.sqlite3", pipeline_mode=overrides.pop("mode", "auto"), **overrides)
        fake = FakeMessages(handlers) if handlers is not None else None
        llm = LLMClient(cfg, client=SimpleNamespace(messages=fake) if fake else None)
        if fake is None:
            llm.client = None  # force rules mode even if the developer has a key in .env
        eng = Engine(cfg, llm=llm, models=models).load()
        return eng, fake

    return _make
