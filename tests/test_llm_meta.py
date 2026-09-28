"""LLM client: Meta Model API wiring, Muse Spark pricing, structured-output fallback."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.config import settings
from app.llm import LLMClient, LLMError, cost_for
from app.pipeline.enrich import LLMEnrichment

ENRICH_PAYLOAD = {
    "canonical_query": "Screen flickers on the Galaxy phone",
    "domain": "display",
    "symptom": "flicker",
    "trigger": "",
    "topic": "display flicker",
    "title": "Fix screen flicker",
    "kind": "Troubleshooting",
    "paraphrases": ["my display keeps flickering", "screen flashes on and off"],
}


def _usage_ns(it=100, ot=50):
    return SimpleNamespace(input_tokens=it, output_tokens=ot)


class _ParseFailsCreateJson:
    """Stands in for anthropic.Anthropic().messages.

    parse() raises like an endpoint that rejects the structured-output beta;
    create() returns a bare JSON object (the JSON-mode fallback path).
    """

    def __init__(self, payload):
        self.payload = payload
        self.parse_calls = 0
        self.create_calls = 0
        self.create_kwargs = None

    def parse(self, **kwargs):
        self.parse_calls += 1
        raise Exception("400 {'type': 'error', 'error': {'message': 'output_format is not supported'}}")

    def create(self, **kwargs):
        self.create_calls += 1
        self.create_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=_usage_ns(),
            stop_reason="end_turn",
        )


def _client_with(messages, monkeypatch, mode="auto"):
    monkeypatch.setenv("MODEL_API_KEY", "LLM|test|secret")
    cfg = replace(settings, pipeline_mode=mode)
    return LLMClient(cfg, client=SimpleNamespace(messages=messages))


def test_has_credentials_reads_model_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert LLMClient.has_credentials() is False
    monkeypatch.setenv("MODEL_API_KEY", "LLM|x|y")
    assert LLMClient.has_credentials() is True


def test_cost_for_muse_spark_pricing():
    # $1.25 / 1M input, $4.25 / 1M output (Meta Model API standard tier)
    assert cost_for("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.50)
    assert cost_for("muse-spark-1.2", 2_000_000, 0) == pytest.approx(2.50)


def test_default_model_is_muse_spark():
    assert settings.llm_model == "muse-spark-1.3"
    assert settings.llm_base_url == "https://api.meta.ai"


def test_structured_falls_back_to_json_mode(monkeypatch):
    messages = _ParseFailsCreateJson(ENRICH_PAYLOAD)
    client = _client_with(messages, monkeypatch)
    out, usage = client.structured("system", "user", LLMEnrichment, max_tokens=500)

    assert messages.parse_calls == 1
    assert messages.create_calls == 1
    assert isinstance(out, LLMEnrichment)
    assert out.domain == "display" and out.symptom == "flicker"
    # usage still logged from the JSON-mode response
    assert usage.input_tokens == 100 and usage.output_tokens == 50
    assert usage.cost_usd == pytest.approx(cost_for("muse-spark-1.3", 100, 50))
    # totals accumulate
    assert client.totals.cost_usd == pytest.approx(usage.cost_usd)


def test_structured_fallback_tolerates_fences_and_validates_schema(monkeypatch):
    payload = dict(ENRICH_PAYLOAD)
    messages = _ParseFailsCreateJson(payload)
    client = _client_with(messages, monkeypatch)
    orig_create = messages.create

    def fenced_create(**kwargs):
        resp = orig_create(**kwargs)
        resp.content[0].text = "```json\n" + resp.content[0].text + "\n```"
        return resp

    messages.create = fenced_create
    out, _ = client.structured("system", "user", LLMEnrichment, max_tokens=500)
    assert out.canonical_query == "Screen flickers on the Galaxy phone"


def test_structured_fallback_rejects_bad_json(monkeypatch):
    messages = _ParseFailsCreateJson(ENRICH_PAYLOAD)

    def bad_create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="not json at all")],
            usage=_usage_ns(),
            stop_reason="end_turn",
        )

    messages.create = bad_create
    client = _client_with(messages, monkeypatch)
    with pytest.raises(LLMError, match="valid JSON"):
        client.structured("system", "user", LLMEnrichment, max_tokens=500)


def test_non_beta_errors_do_not_fallback(monkeypatch):
    class AlwaysFails:
        def parse(self, **kwargs):
            raise Exception("503 overloaded")
        def create(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("fallback must not trigger on 503")

    client = _client_with(AlwaysFails(), monkeypatch)
    with pytest.raises(LLMError, match="LLM call failed"):
        client.structured("system", "user", LLMEnrichment, max_tokens=500)
