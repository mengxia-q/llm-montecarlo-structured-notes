"""DeepSeek provider against mocked API responses: both modes, empty and invalid output."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.extract.llm_schema import RawNoteTerms
from src.extract.providers import TOOL_NAME, DeepSeekProvider, strict_json_schema

VALID = {
    "issuer": "X",
    "notional_per_unit": 1000,
    "pricing_date": "2026-07-01",
    "observation_dates": ["2026-10-01"],
    "coupon": {"rate_per_period_pct": 2.0, "barrier_pct": 70, "memory": False},
}


def _response(*, content=None, arguments=None, reasoning=100, completion=600):
    tool_calls = None
    if arguments is not None:
        tool_calls = [
            SimpleNamespace(function=SimpleNamespace(name=TOOL_NAME, arguments=arguments))
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=20_000,
        completion_tokens=completion,
        total_tokens=20_000 + completion,
        prompt_cache_hit_tokens=19_000,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        model_dump_json=lambda: json.dumps({"mock": True}),
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _provider(responses, **kwargs):
    client = FakeClient(responses)
    return DeepSeekProvider(client=client, **kwargs), client


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_strict_schema_is_closed_and_fully_required():
    schema = strict_json_schema(RawNoteTerms)
    assert "$defs" not in schema
    for node in _walk(schema):
        assert "$ref" not in node and "default" not in node and "format" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
    # Nested objects are inlined, nullable scalars stay nullable.
    coupon = schema["properties"]["coupon"]
    assert coupon["type"] == "object"
    rate_types = {alt.get("type") for alt in coupon["properties"]["rate_per_period_pct"]["anyOf"]}
    assert rate_types == {"number", "null"}


def test_strict_mode_parses_tool_arguments_and_accounts_tokens():
    provider, client = _provider([_response(arguments=json.dumps(VALID))])
    call = provider.extract("doc", seed_index=0)
    assert call.ok and call.raw.issuer == "X"
    assert call.mode == "strict_tool" and call.endpoint.endswith("/beta")
    assert call.thinking_tokens == 100 and call.output_tokens == 500
    assert call.billed_output_tokens == 600
    assert call.cached_input_tokens == 19_000
    assert call.raw_response and call.api_retries == 0
    sent = client.requests[0]
    assert sent["tools"][0]["function"]["strict"] is True
    # Thinking mode rejects a forced tool_choice, so it must be "auto".
    assert sent["tool_choice"] == "auto"


def test_strict_mode_without_thinking_forces_the_tool():
    provider, client = _provider([_response(arguments=json.dumps(VALID))], thinking=False)
    provider.extract("doc", seed_index=0)
    sent = client.requests[0]
    assert sent["tool_choice"]["function"]["name"] == TOOL_NAME
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


def test_json_object_mode_validates_locally():
    provider, client = _provider([_response(content=json.dumps(VALID))], mode="json_object")
    call = provider.extract("doc", seed_index=2)
    assert call.ok and call.mode == "json_object" and call.seed_index == 2
    sent = client.requests[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in sent["messages"][0]["content"]


def test_empty_reply_is_retried_as_api_failure():
    provider, _ = _provider(
        [_response(content=""), _response(content=json.dumps(VALID))], mode="json_object"
    )
    call = provider.extract("doc", seed_index=0)
    assert call.ok and call.api_retries == 1


def test_missing_tool_call_exhausts_retries_as_api_failure():
    provider, _ = _provider([_response(content="thinking aloud")] * 3)
    call = provider.extract("doc", seed_index=0)
    assert not call.ok and call.error_kind == "api" and call.api_retries == 3


def test_transport_error_then_success(monkeypatch):
    monkeypatch.setattr("src.extract.providers.time.sleep", lambda s: None)
    provider, _ = _provider([RuntimeError("boom"), _response(arguments=json.dumps(VALID))])
    call = provider.extract("doc", seed_index=0)
    assert call.ok and call.api_retries == 1


def test_invalid_output_is_a_schema_error_not_retried():
    bad = json.dumps({**VALID, "pricing_date": "not a date"})
    provider, client = _provider([_response(content=bad)], mode="json_object")
    call = provider.extract("doc", seed_index=0)
    assert not call.ok and call.error_kind == "schema"
    assert len(client.requests) == 1


def test_unknown_mode_rejected():
    with pytest.raises(Exception):
        DeepSeekProvider(client=FakeClient([]), mode="json_schema")
