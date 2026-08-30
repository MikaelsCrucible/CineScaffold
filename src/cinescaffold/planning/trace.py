from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
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
        self.active_stream_request_index: int | None = None
        self.active_stream_started: float | None = None

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
        self._record_completed_response(
            request_index=self._request_index,
            message_count=len(messages),
            started=started,
            response=response,
        )
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        """记录流式请求边界，具体内容只由遥测处理器计数。"""

        self._request_index += 1
        request_index = self._request_index
        started = time.monotonic()
        self.active_stream_request_index = request_index
        self.active_stream_started = started
        self.trace.record(
            "model_request_started",
            request_index=request_index,
            provider=self.system,
            model=self.model_name,
            message_count=len(messages),
            function_tools=[item.name for item in model_request_parameters.function_tools],
            output_tools=[item.name for item in model_request_parameters.output_tools],
            streaming=True,
        )
        try:
            async with self.wrapped.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response_stream:
                self.trace.record(
                    "model_stream_connected",
                    request_index=request_index,
                    connection_latency_ms=round((time.monotonic() - started) * 1000, 3),
                )
                yield response_stream
                self._record_completed_response(
                    request_index=request_index,
                    message_count=len(messages),
                    started=started,
                    response=response_stream.get(),
                )
        except Exception as error:
            self.trace.record(
                "model_request_failed",
                request_index=request_index,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            self.active_stream_request_index = None
            self.active_stream_started = None

    def _record_completed_response(
        self,
        *,
        request_index: int,
        message_count: int,
        started: float,
        response: ModelResponse,
    ) -> None:
        usage_values = _usage_dict(response.usage)
        input_tokens = int(usage_values.get("input_tokens", 0))
        cache_read_tokens = int(usage_values.get("cache_read_tokens", 0))
        cache_write_tokens = int(usage_values.get("cache_write_tokens", 0))
        self.request_metrics.append(
            {
                "request_index": request_index,
                "message_count": message_count,
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
            request_index=request_index,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            usage=usage_values,
            response_parts=_response_part_summary(response),
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
        )


class StreamTelemetryHandler:
    """消费 Agent 流事件，只记录推理与输出的时序和规模。"""

    def __init__(
        self,
        trace: TraceRecorder,
        model: TracingModel,
        *,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        self.trace = trace
        self.model = model
        self.heartbeat_seconds = heartbeat_seconds

    async def __call__(
        self,
        _run_context: RunContext[Any],
        events: AsyncIterable[Any],
    ) -> None:
        request_index = self.model.active_stream_request_index
        started = self.model.active_stream_started
        if request_index is None or started is None:
            async for _ in events:
                pass
            return

        metrics = _new_stream_metrics(request_index)
        iterator = events.__aiter__()
        pending = asyncio.create_task(anext(iterator))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {pending},
                    timeout=self.heartbeat_seconds,
                )
                if not done:
                    self.trace.record(
                        "model_stream_stalled",
                        request_index=request_index,
                        elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                        seconds_since_last_event=round(
                            time.monotonic() - metrics["last_event_at"],
                            3,
                        ),
                        reasoning_chunks=metrics["reasoning_chunks"],
                        reasoning_chars=metrics["reasoning_chars"],
                        text_chunks=metrics["text_chunks"],
                        text_chars=metrics["text_chars"],
                    )
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                _accumulate_stream_event(metrics, event, started)
                pending = asyncio.create_task(anext(iterator))
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        self.trace.record(
            "model_stream_telemetry_completed",
            **_stream_metrics_payload(metrics, started),
        )


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
            # 只记录规模，不保存模型推理原文。
            summaries.append(
                {
                    "kind": "thinking",
                    "content_length": len(part.content),
                    "content_recorded": False,
                }
            )
        else:
            content = getattr(part, "content", None)
            summaries.append(
                {
                    "kind": getattr(part, "part_kind", type(part).__name__),
                    "content_length": len(content) if isinstance(content, str) else None,
                }
            )
    return summaries


def _new_stream_metrics(request_index: int) -> dict[str, Any]:
    now = time.monotonic()
    return {
        "request_index": request_index,
        "event_count": 0,
        "reasoning_chunks": 0,
        "reasoning_chars": 0,
        "text_chunks": 0,
        "text_chars": 0,
        "tool_call_chunks": 0,
        "first_event_at": None,
        "first_reasoning_at": None,
        "first_text_at": None,
        "first_tool_call_at": None,
        "last_event_at": now,
        "max_inter_event_gap_seconds": 0.0,
    }


def _accumulate_stream_event(
    metrics: dict[str, Any],
    event: Any,
    started: float,
) -> None:
    """在此处读取流事件；Trace 只接收计数和时间，不接收内容。"""

    now = time.monotonic()
    metrics["event_count"] += 1
    if metrics["first_event_at"] is None:
        metrics["first_event_at"] = now
    gap = now - metrics["last_event_at"]
    metrics["max_inter_event_gap_seconds"] = max(
        metrics["max_inter_event_gap_seconds"],
        gap,
    )
    metrics["last_event_at"] = now

    if isinstance(event, PartStartEvent):
        part = event.part
        if isinstance(part, ThinkingPart):
            _count_stream_content(metrics, "reasoning", part.content, now)
        elif isinstance(part, TextPart):
            _count_stream_content(metrics, "text", part.content, now)
        elif isinstance(part, ToolCallPart):
            metrics["tool_call_chunks"] += 1
            metrics["first_tool_call_at"] = metrics["first_tool_call_at"] or now
    elif isinstance(event, PartDeltaEvent):
        delta = event.delta
        if isinstance(delta, ThinkingPartDelta):
            _count_stream_content(metrics, "reasoning", delta.content_delta, now)
        elif isinstance(delta, TextPartDelta):
            _count_stream_content(metrics, "text", delta.content_delta, now)
        elif isinstance(delta, ToolCallPartDelta):
            metrics["tool_call_chunks"] += 1
            metrics["first_tool_call_at"] = metrics["first_tool_call_at"] or now


def _count_stream_content(
    metrics: dict[str, Any],
    kind: str,
    content: str | None,
    now: float,
) -> None:
    if not content:
        return
    metrics[f"{kind}_chunks"] += 1
    metrics[f"{kind}_chars"] += len(content)
    metrics[f"first_{kind}_at"] = metrics[f"first_{kind}_at"] or now


def _stream_metrics_payload(metrics: dict[str, Any], started: float) -> dict[str, Any]:
    ended = time.monotonic()

    def elapsed(value: float | None) -> float | None:
        return round((value - started) * 1000, 3) if value is not None else None

    return {
        "request_index": metrics["request_index"],
        "duration_ms": round((ended - started) * 1000, 3),
        "event_count": metrics["event_count"],
        "reasoning_chunks": metrics["reasoning_chunks"],
        "reasoning_chars": metrics["reasoning_chars"],
        "text_chunks": metrics["text_chunks"],
        "text_chars": metrics["text_chars"],
        "tool_call_chunks": metrics["tool_call_chunks"],
        "first_event_ms": elapsed(metrics["first_event_at"]),
        "first_reasoning_ms": elapsed(metrics["first_reasoning_at"]),
        "first_text_ms": elapsed(metrics["first_text_at"]),
        "first_tool_call_ms": elapsed(metrics["first_tool_call_at"]),
        "max_inter_event_gap_ms": round(metrics["max_inter_event_gap_seconds"] * 1000, 3),
        "reasoning_content_recorded": False,
    }


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
