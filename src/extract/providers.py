"""Pluggable LLM providers.

``deepseek`` is the paper's main evaluation model and ``qwen`` the second; ``gemini`` is kept because the
early pilot (``experiments/02_vertical_slice.py``) ran on it.  Adding another vendor means adding one
subclass and one entry to ``_PROVIDERS``; nothing else in the pipeline changes.

Every call records the model id, the sampling temperature, token usage and
latency, which is what the paper's cost table is built from, and
the structured-output mode and raw response, which the paper publishes.
"""

from __future__ import annotations

import copy
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .llm_schema import RawNoteTerms


class ProviderError(RuntimeError):
    """Raised when a provider call fails in a way retrying will not fix."""


@dataclass(frozen=True)
class ExtractionCall:
    """One sampled extraction, with the accounting the paper reports.

    Reasoning tokens are recorded separately because they are *not* included in
    the visible-output count but are billed as output, and on these models they
    are not small: a nine-token prompt returning eighteen output tokens consumed
    214 reasoning tokens.  A cost table built on ``output_tokens`` alone would
    understate the bill by a large factor, so ``billed_output_tokens`` is what
    the paper's efficiency table must use.
    """

    raw: RawNoteTerms | None
    model: str
    temperature: float
    seed_index: int
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    error: str | None = None
    finish_reason: str | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    mode: str | None = None
    """Structured-output mode actually used: ``strict_tool``, ``json_object`` ..."""

    endpoint: str | None = None
    called_at: str | None = None
    """UTC ISO timestamp of the first attempt."""

    raw_response: str | None = None
    """The provider's full response body for the successful (or last) attempt."""

    api_retries: int = 0
    """Attempts that failed at the API level (errors, empty content) before this one."""

    error_kind: str | None = None
    """``api`` for transport/empty-response failures, ``schema`` for output that did
    not parse as ``RawNoteTerms``.  Only the latter is an extraction error."""

    @property
    def ok(self) -> bool:
        return self.raw is not None

    @property
    def billed_output_tokens(self) -> int | None:
        """Visible output plus reasoning tokens, which bill at the output rate."""
        if self.output_tokens is None:
            return None
        return self.output_tokens + (self.thinking_tokens or 0)


class LLMProvider(ABC):
    """Extracts ``RawNoteTerms`` from document text, with schema-constrained output."""

    name: str

    def __init__(self, model: str, *, temperature: float = 0.7) -> None:
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def extract(self, prompt: str, *, seed_index: int, temperature: float | None = None) -> ExtractionCall:
        ...

    def sample(self, prompt: str, k: int, *, temperature: float | None = None) -> list[ExtractionCall]:
        """Draw ``k`` independent extractions for the self-consistency vote."""
        return [
            self.extract(prompt, seed_index=i, temperature=temperature)
            for i in range(k)
        ]


class GeminiProvider(LLMProvider):
    """Google Gemini with a JSON response schema derived from ``RawNoteTerms``."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        *,
        temperature: float = 0.7,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model, temperature=temperature)
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.max_retries = max_retries

    def extract(
        self, prompt: str, *, seed_index: int, temperature: float | None = None
    ) -> ExtractionCall:
        temp = self.temperature if temperature is None else temperature
        config = {
            "response_mime_type": "application/json",
            "response_schema": RawNoteTerms,
            "temperature": temp,
        }

        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                usage = getattr(response, "usage_metadata", None)
                parsed = getattr(response, "parsed", None)
                if parsed is None and response.text:
                    parsed = RawNoteTerms.model_validate_json(response.text)
                finish = None
                if getattr(response, "candidates", None):
                    finish = str(getattr(response.candidates[0], "finish_reason", None))
                return ExtractionCall(
                    raw=parsed,
                    model=self.model,
                    temperature=temp,
                    seed_index=seed_index,
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                    thinking_tokens=getattr(usage, "thoughts_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                    cached_input_tokens=getattr(
                        usage, "cached_content_token_count", None
                    ),
                    latency_s=time.monotonic() - started,
                    finish_reason=finish,
                )
            except Exception as exc:  # noqa: BLE001 - providers raise many types
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))

        return ExtractionCall(
            raw=None,
            model=self.model,
            temperature=temp,
            seed_index=seed_index,
            input_tokens=None,
            output_tokens=None,
            latency_s=time.monotonic() - started,
            error=f"{type(last_error).__name__}: {last_error}",
        )


def strict_json_schema(model: type) -> dict:
    """Pydantic JSON Schema rewritten into the subset strict function calling accepts.

    Strict mode constrains decoding to the schema, but only for a restricted
    dialect: every property must be listed in ``required``, objects must set
    ``additionalProperties: false``, and nullability is expressed as a union with
    ``null`` rather than by omission.  ``$ref`` is inlined and ``title``,
    ``default`` and ``format`` are dropped (the date format lives in the field
    descriptions, which the model still sees).
    """
    root = model.model_json_schema()
    defs = root.pop("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            target.update(extra)
            return walk(target)
        out = {}
        for key, value in node.items():
            if key in ("title", "default", "format"):
                continue
            if key == "properties":
                out[key] = {name: walk(sub) for name, sub in value.items()}
            else:
                out[key] = walk(value)
        if out.get("type") == "object" or "properties" in out:
            out["type"] = "object"
            out["required"] = list(out.get("properties", {}))
            out["additionalProperties"] = False
        return out

    return walk(root)


#: Name of the function the model is asked to call in strict mode.
TOOL_NAME = "record_note_terms"

_STRICT_SYSTEM = (
    "You are a careful financial document analyst. Call the "
    f"`{TOOL_NAME}` function exactly once with the extracted terms."
)
_JSON_SYSTEM = (
    "You are a careful financial document analyst. Reply with a single JSON "
    "object, and nothing else, that conforms to this JSON Schema:\n{schema}"
)


class DeepSeekProvider(LLMProvider):
    """DeepSeek through its OpenAI-compatible API.

    Two structured-output modes, recorded on every call:

    ``strict_tool``
        ``RawNoteTerms`` is passed as a strict function schema on the beta
        endpoint, so the arguments are decoded under the schema's constraints.
        Thinking mode rejects a forced ``tool_choice`` (HTTP 400, measured
        2026-10-06), so with thinking on the choice is ``auto`` and a reply
        without a tool call counts as an API failure and is retried.
    ``json_object``
        The schema is written into the system prompt and the reply is only
        guaranteed to be valid JSON; it is checked locally with Pydantic.

    An empty reply - which DeepSeek documents can happen in ``json_object``
    mode - is an API failure, retried and counted in ``api_retries``, never an
    extraction error.
    """

    name = "deepseek"
    BASE_URL = "https://api.deepseek.com"
    BETA_URL = "https://api.deepseek.com/beta"

    def __init__(
        self,
        model: str = "deepseek-flash",
        *,
        temperature: float = 0.7,
        mode: str = "strict_tool",
        thinking: bool = True,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout_s: float = 600.0,
        client=None,
    ) -> None:
        super().__init__(model, temperature=temperature)
        if mode not in ("strict_tool", "json_object"):
            raise ProviderError(f"unknown DeepSeek mode {mode!r}")
        self.mode = mode
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.endpoint = self.BETA_URL if mode == "strict_tool" else self.BASE_URL
        self.schema = strict_json_schema(RawNoteTerms)
        if client is None:
            key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise ProviderError("DEEPSEEK_API_KEY is not set")
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=self.endpoint, timeout=timeout_s)
        self._client = client

    def request_kwargs(self, prompt: str, temperature: float) -> dict:
        """The exact request body sent, minus the model id (also logged)."""
        kwargs: dict = {"temperature": temperature}
        if self.mode == "strict_tool":
            kwargs["messages"] = [
                {"role": "system", "content": _STRICT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": "Record the economic terms of the note.",
                        "strict": True,
                        "parameters": self.schema,
                    },
                }
            ]
            kwargs["tool_choice"] = (
                "auto"
                if self.thinking
                else {"type": "function", "function": {"name": TOOL_NAME}}
            )
        else:
            kwargs["messages"] = [
                {
                    "role": "system",
                    "content": _JSON_SYSTEM.format(schema=json.dumps(self.schema)),
                },
                {"role": "user", "content": prompt},
            ]
            kwargs["response_format"] = {"type": "json_object"}
        extra: dict = {}
        if not self.thinking:
            extra["thinking"] = {"type": "disabled"}
        if extra:
            kwargs["extra_body"] = extra
        if self.thinking and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _cached_tokens(self, usage) -> int | None:
        return getattr(usage, "prompt_cache_hit_tokens", None)

    def _payload_text(self, response) -> str | None:
        message = response.choices[0].message
        if self.mode == "strict_tool":
            for call in message.tool_calls or []:
                if call.function.name == TOOL_NAME and call.function.arguments:
                    return call.function.arguments
            return None
        return message.content or None

    def extract(
        self, prompt: str, *, seed_index: int, temperature: float | None = None
    ) -> ExtractionCall:
        temp = self.temperature if temperature is None else temperature
        kwargs = self.request_kwargs(prompt, temp)
        called_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = time.monotonic()
        failures = 0
        last_error = "no attempt made"
        last_raw: str | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - transport errors vary
                failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                continue

            last_raw = response.model_dump_json()
            text = self._payload_text(response)
            finish = response.choices[0].finish_reason
            usage = response.usage
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None) or 0
            completion = getattr(usage, "completion_tokens", None)
            accounting = dict(
                model=self.model,
                temperature=temp,
                seed_index=seed_index,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=None if completion is None else completion - reasoning,
                thinking_tokens=reasoning,
                total_tokens=getattr(usage, "total_tokens", None),
                cached_input_tokens=self._cached_tokens(usage),
                finish_reason=finish,
                mode=self.mode,
                endpoint=self.endpoint,
                called_at=called_at,
                raw_response=last_raw,
                api_retries=failures,
            )

            if not text:
                failures += 1
                last_error = f"empty reply (finish_reason={finish})"
                if attempt < self.max_retries - 1:
                    continue
                return ExtractionCall(
                    raw=None,
                    latency_s=time.monotonic() - started,
                    error=last_error,
                    error_kind="api",
                    **{**accounting, "api_retries": failures},
                )

            try:
                parsed = RawNoteTerms.model_validate_json(text)
            except Exception as exc:  # noqa: BLE001 - pydantic / json errors
                return ExtractionCall(
                    raw=None,
                    latency_s=time.monotonic() - started,
                    error=f"{type(exc).__name__}: {str(exc)[:500]}",
                    error_kind="schema",
                    **accounting,
                )
            return ExtractionCall(
                raw=parsed, latency_s=time.monotonic() - started, **accounting
            )

        return ExtractionCall(
            raw=None,
            model=self.model,
            temperature=temp,
            seed_index=seed_index,
            input_tokens=None,
            output_tokens=None,
            latency_s=time.monotonic() - started,
            error=last_error,
            error_kind="api",
            mode=self.mode,
            endpoint=self.endpoint,
            called_at=called_at,
            raw_response=last_raw,
            api_retries=failures,
        )


class QwenProvider(DeepSeekProvider):
    """Qwen through Alibaba's OpenAI-compatible endpoint (second evaluation model).

    The request differs from DeepSeek's in three ways, all measured 2026-10-13 on
    ``qwen3.8-flash``:

    ``strict_schema``
        Constrained decoding is requested with ``response_format`` of type
        ``json_schema`` and ``strict: true``.  Qwen accepts ``strict`` on a tool
        definition but did not enforce it (an integer field came back as 7.5),
        whereas the ``json_schema`` response format did, with reasoning on and off.
    ``json_object``
        As for DeepSeek: schema in the system prompt, only valid JSON guaranteed.
    Reasoning is switched with ``enable_thinking``, and cache hits are reported in
    ``prompt_tokens_details.cached_tokens``.

    The key and endpoint are read from ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``.
    """

    name = "qwen"

    def __init__(
        self,
        model: str = "qwen3.8-flash",
        *,
        temperature: float = 0.7,
        mode: str = "strict_schema",
        thinking: bool = True,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout_s: float = 600.0,
        client=None,
    ) -> None:
        LLMProvider.__init__(self, model, temperature=temperature)
        if mode not in ("strict_schema", "json_object"):
            raise ProviderError(f"unknown Qwen mode {mode!r}")
        self.mode = mode
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.endpoint = os.environ.get("OPENAI_BASE_URL", "")
        self.schema = strict_json_schema(RawNoteTerms)
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key or not self.endpoint:
                raise ProviderError("OPENAI_API_KEY / OPENAI_BASE_URL are not set")
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=self.endpoint, timeout=timeout_s)
        self._client = client

    def request_kwargs(self, prompt: str, temperature: float) -> dict:
        kwargs: dict = {"temperature": temperature}
        if self.mode == "strict_schema":
            kwargs["messages"] = [
                {"role": "system", "content": "You are a careful financial document analyst. "
                 "Reply with the extracted terms as a JSON object."},
                {"role": "user", "content": prompt},
            ]
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": TOOL_NAME, "strict": True, "schema": self.schema},
            }
        else:
            kwargs["messages"] = [
                {"role": "system", "content": _JSON_SYSTEM.format(schema=json.dumps(self.schema))},
                {"role": "user", "content": prompt},
            ]
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"enable_thinking": self.thinking}
        return kwargs

    def _payload_text(self, response) -> str | None:
        return response.choices[0].message.content or None

    def _cached_tokens(self, usage) -> int | None:
        details = getattr(usage, "prompt_tokens_details", None)
        return getattr(details, "cached_tokens", None)


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
}


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate a provider by name.

    Registered: ``deepseek``, ``gemini``, ``qwen``.  The rest of the pipeline is
    provider-agnostic.
    """
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise ProviderError(
            f"unknown provider {name!r}; registered: {sorted(_PROVIDERS)}"
        ) from None
    return cls(**kwargs)
