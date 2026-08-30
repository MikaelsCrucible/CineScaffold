from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage


TraceEventCallback = Callable[[str, dict[str, Any]], None]


class TraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_event_bytes: int = Field(default=32_768, ge=2_048)
    max_string_chars: int = Field(default=4_096, ge=256)
    max_list_items: int = Field(default=100, ge=10)


class CostRates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    currency: str = "USD"
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    cache_read_per_million: Decimal | None = Field(default=None, ge=0)
    cache_write_per_million: Decimal | None = Field(default=None, ge=0)
    source: str = "user_supplied"


class TraceRecorder:
    def __init__(
        self,
        path: Path,
        run_id: str,
        config: TraceConfig | None = None,
        event_callback: TraceEventCallback | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.config = config or TraceConfig()
        self.event_callback = event_callback
        self._sequence = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def record(self, event_type: str, **payload: Any) -> None:
        self._sequence += 1
        safe_payload = _sanitize(payload, self.config)
        event = {
            "trace_version": "0.1",
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": safe_payload,
        }
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self.config.max_event_bytes:
            event["payload"] = {
                "truncated": True,
                "original_size_bytes": len(encoded.encode("utf-8")),
                "summary": _payload_summary(payload),
            }
            encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")
        if self.event_callback is not None:
            # 展示层失败不得中断实验状态机或 Trace 写入。
            try:
                self.event_callback(event_type, safe_payload)
            except Exception:
                pass


class TracingModel(WrapperModel):
    def __init__(self, wrapped: Model, trace: TraceRecorder) -> None:
        super().__init__(wrapped)
        self.trace = trace
        self._request_index = 0
        self.request_metrics: list[dict[str, int]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._request_index += 1
        started = time.monotonic()
        self.trace.record(
            "model_request_started",
            request_index=self._request_index,
            provider=self.system,
            model=self.model_name,
            message_count=len(messages),
            function_tools=[item.name for item in model_request_parameters.function_tools],
            output_tools=[item.name for item in model_request_parameters.output_tools],
        )
        try:
            response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except Exception as error:
            self.trace.record(
                "model_request_failed",
                request_index=self._request_index,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        usage_values = _usage_dict(response.usage)
        input_tokens = int(usage_values.get("input_tokens", 0))
        cache_read_tokens = int(usage_values.get("cache_read_tokens", 0))
        cache_write_tokens = int(usage_values.get("cache_write_tokens", 0))
        self.request_metrics.append(
            {
                "request_index": self._request_index,
                "message_count": len(messages),
                "input_tokens": input_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "uncached_input_tokens": max(
                    0, input_tokens - cache_read_tokens - cache_write_tokens
                ),
                "output_tokens": int(usage_values.get("output_tokens", 0)),
            }
        )
        self.trace.record(
            "model_request_completed",
            request_index=self._request_index,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            usage=usage_values,
            response_parts=_response_part_summary(response),
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
        )
        return response


def usage_summary(
    usage: RunUsage,
    rates: CostRates | None,
    request_metrics: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    values = _usage_dict(usage)
    metrics = request_metrics or []
    result: dict[str, Any] = {
        "tokens": values,
        "context": {
            "semantics": (
                "aggregate_input_tokens sums every request and includes cache reads; "
                "max_request_input_tokens is the largest single model context"
            ),
            "max_request_input_tokens": max(
                (item["input_tokens"] for item in metrics), default=0
            ),
            "max_request_message_count": max(
                (item["message_count"] for item in metrics), default=0
            ),
            "aggregate_uncached_input_tokens": sum(
                item["uncached_input_tokens"] for item in metrics
            ),
            "cache_hit_ratio": (
                values.get("cache_read_tokens", 0) / values.get("input_tokens", 1)
                if values.get("input_tokens", 0)
                else 0.0
            ),
        },
        "pricing_snapshot": None,
        "estimated_cost": None,
    }
    if rates is None:
        return result
    input_tokens = Decimal(values.get("input_tokens", 0))
    output_tokens = Decimal(values.get("output_tokens", 0))
    cache_read = Decimal(values.get("cache_read_tokens", 0))
    cache_write = Decimal(values.get("cache_write_tokens", 0))
    uncached = max(Decimal(0), input_tokens - cache_read - cache_write)
    cache_read_rate = rates.cache_read_per_million or rates.input_per_million
    cache_write_rate = rates.cache_write_per_million or rates.input_per_million
    total = (
        uncached * rates.input_per_million
        + cache_read * cache_read_rate
        + cache_write * cache_write_rate
        + output_tokens * rates.output_per_million
    ) / Decimal(1_000_000)
    result["pricing_snapshot"] = {
        **rates.model_dump(mode="json"),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "formula": "uncached_input + cache_read + cache_write + output",
    }
    result["estimated_cost"] = {
        "amount": f"{total.quantize(Decimal('0.00000001')):.8f}",
        "currency": rates.currency,
    }
    return result


def _usage_dict(usage: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(usage):
        values = dataclasses.asdict(usage)
    elif hasattr(usage, "__dict__"):
        values = dict(usage.__dict__)
    else:
        return {}
    return {key: value for key, value in values.items() if value not in (0, {}, None)}


def _response_part_summary(response: ModelResponse) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            summaries.append({"kind": "tool_call", "tool_name": part.tool_name})
        elif isinstance(part, ThinkingPart):
            # 只记录存在性，不保存模型隐藏思维。
            summaries.append({"kind": "thinking", "content_recorded": False})
        else:
            content = getattr(part, "content", None)
            summaries.append(
                {
                    "kind": getattr(part, "part_kind", type(part).__name__),
                    "content_length": len(content) if isinstance(content, str) else None,
                }
            )
    return summaries


def _sanitize(value: Any, config: TraceConfig, key: str | None = None) -> Any:
    if key and any(marker in key.lower() for marker in ("api_key", "authorization", "password", "secret")):
        return "<redacted>"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize(item_value, config, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        selected = list(value[: config.max_list_items])
        result = [_sanitize(item, config) for item in selected]
        if len(value) > config.max_list_items:
            result.append({"truncated_items": len(value) - config.max_list_items})
        return result
    if isinstance(value, str) and len(value) > config.max_string_chars:
        return value[: config.max_string_chars] + f"…<truncated {len(value) - config.max_string_chars} chars>"
    if isinstance(value, Decimal):
        return str(value)
    return value


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(payload),
        "value_types": {key: type(value).__name__ for key, value in payload.items()},
    }
