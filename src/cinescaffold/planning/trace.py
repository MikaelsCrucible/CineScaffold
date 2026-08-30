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
        event = {
            "trace_version": "0.1",
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": {},
        }
        encoded = self._encode_event(event, payload)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")
        if self.event_callback is not None:
            # 展示层失败不得中断实验状态机或 Trace 写入。
            try:
                self.event_callback(event_type, _sanitize(payload, self.config))
            except Exception:
                pass

    def _encode_event(self, event: dict[str, Any], payload: dict[str, Any]) -> str:
        """序列化事件并保证不超字节上限：逐级收紧截断以尽量保留内容，最后才降级为摘要。"""
        safe_payload = _sanitize(payload, self.config)
        encoded = _json_dumps({**event, "payload": safe_payload})
        if _json_bytes(encoded) <= self.config.max_event_bytes:
            return encoded
        max_chars = self.config.max_string_chars
        max_items = self.config.max_list_items
        while max_chars > 256 or max_items > 10:
            max_chars = max(256, max_chars // 2)
            max_items = max(10, max_items // 2)
            safe_payload = _sanitize(
                payload,
                self.config,
                max_chars=max_chars,
                max_items=max_items,
            )
            encoded = _json_dumps({**event, "payload": safe_payload})
            if _json_bytes(encoded) <= self.config.max_event_bytes:
                return encoded
        event["payload"] = {
            "truncated": True,
            "original_size_bytes": _json_bytes(
                _json_dumps({**event, "payload": _sanitize(payload, self.config)})
            ),
            "summary": _payload_summary(payload),
        }
        return _json_dumps(event)


class TracingModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        trace: TraceRecorder,
        *,
        record_content: bool = False,
    ) -> None:
        super().__init__(wrapped)
        self.trace = trace
        # 仅 --full-power-diagnostic 模式记录完整对话与 thinking 原文；
        # 普通运行保持精简元数据日志，不落盘消息内容。
        self.record_content = record_content
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
            **(_content_messages(messages) if self.record_content else {}),
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
                **(_content_messages(messages) if self.record_content else {}),
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
        """记录流式请求边界；完整对话与流式推理/正文原文只在诊断模式落盘。"""

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
            **(_content_messages(messages) if self.record_content else {}),
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
                **(_content_messages(messages) if self.record_content else {}),
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
            response_parts=(
                _response_parts(response) if self.record_content else _response_part_summary(response)
            ),
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
        )


class StreamTelemetryHandler:
    """消费 Agent 流事件，记录推理与输出的时序、规模及原文内容。"""

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
        # pydantic-ai 惰性进入模型 request_stream：handler 被调用时
        # active_stream_request_index 尚未设置，必须等收到首个事件后再读取关联。
        iterator = events.__aiter__()
        pending = asyncio.create_task(anext(iterator))
        request_index: int | None = None
        started: float | None = None
        metrics: dict[str, Any] | None = None
        try:
            while True:
                done, _ = await asyncio.wait(
                    {pending},
                    timeout=self.heartbeat_seconds,
                )
                if not done:
                    if metrics is not None:
                        stall_payload: dict[str, Any] = {
                            "request_index": request_index,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                            "seconds_since_last_event": round(
                                time.monotonic() - metrics["last_event_at"],
                                3,
                            ),
                            "reasoning_chunks": metrics["reasoning_chunks"],
                            "reasoning_chars": metrics["reasoning_chars"],
                            "text_chunks": metrics["text_chunks"],
                            "text_chars": metrics["text_chars"],
                        }
                        if metrics["record_content"]:
                            stall_payload.update(
                                {
                                    "reasoning_content": "".join(metrics["reasoning_content"]),
                                    "text_content": "".join(metrics["text_content"]),
                                }
                            )
                        self.trace.record("model_stream_stalled", **stall_payload)
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                if metrics is None:
                    request_index = self.model.active_stream_request_index
                    started = self.model.active_stream_started
                    if request_index is None or started is None:
                        # 非模型节点（工具调用等）的事件流无法关联请求：
                        # 排空事件但不记录，且必须推进 pending 任务，否则会
                        # 反复等待同一个已完成任务而死锁。
                        pending = asyncio.create_task(anext(iterator))
                        continue
                    metrics = _new_stream_metrics(
                        request_index,
                        record_content=getattr(self.model, "record_content", False),
                    )
                _accumulate_stream_event(metrics, event, started)
                pending = asyncio.create_task(anext(iterator))
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        if metrics is not None:
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


def _content_messages(messages: list[ModelMessage]) -> dict[str, Any]:
    return {"messages": _messages_content(messages)}


def _messages_content(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """把完整对话（含历史 thinking、工具调用与返回）转成可序列化结构。"""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            entry: dict[str, Any] = {
                "kind": "response",
                "parts": [_part_entry(part) for part in message.parts],
            }
        else:
            entry = {
                "kind": "request",
                "parts": [_part_entry(part) for part in message.parts],
            }
            instructions = getattr(message, "instructions", None)
            if instructions:
                entry["instructions"] = [_part_entry(part) for part in instructions]
        rendered.append(entry)
    return rendered


def _response_parts(response: ModelResponse) -> list[dict[str, Any]]:
    """完整记录响应各部分，包括 thinking 原文与工具调用参数。"""
    return [_part_entry(part) for part in response.parts]


def _response_part_summary(response: ModelResponse) -> list[dict[str, Any]]:
    """精简模式：只记录各部分的类型与规模，不保存原文。"""
    summaries: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            summaries.append({"kind": "tool_call", "tool_name": part.tool_name})
        elif isinstance(part, ThinkingPart):
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


def _part_entry(part: Any) -> dict[str, Any]:
    """把单个消息部分转为带完整原文的可序列化条目。"""
    kind = getattr(part, "part_kind", type(part).__name__)
    entry: dict[str, Any] = {"kind": kind}
    if isinstance(part, ThinkingPart):
        entry["content"] = part.content
        entry["content_recorded"] = True
        entry["content_length"] = len(part.content)
    elif isinstance(part, ToolCallPart):
        entry["tool_name"] = part.tool_name
        entry["tool_call_id"] = part.tool_call_id
        entry["args"] = part.args
        entry["content_recorded"] = True
    else:
        content = getattr(part, "content", None)
        if isinstance(content, (bytes, bytearray)) and getattr(part, "transcript", None):
            # 语音/音频部分优先记录可读 transcript，不倾倒二进制。
            content = getattr(part, "transcript")
        if content is not None:
            entry["content"] = content
            entry["content_recorded"] = True
            if isinstance(content, str):
                entry["content_length"] = len(content)
        tool_name = getattr(part, "tool_name", None)
        if tool_name is not None:
            entry["tool_name"] = tool_name
        tool_call_id = getattr(part, "tool_call_id", None)
        if tool_call_id is not None:
            entry["tool_call_id"] = tool_call_id
        outcome = getattr(part, "outcome", None)
        if outcome is not None:
            entry["outcome"] = outcome
    return entry


def _new_stream_metrics(
    request_index: int,
    *,
    record_content: bool,
) -> dict[str, Any]:
    now = time.monotonic()
    return {
        "request_index": request_index,
        "record_content": record_content,
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
        "reasoning_content": [],
        "text_content": [],
    }


def _accumulate_stream_event(
    metrics: dict[str, Any],
    event: Any,
    started: float,
) -> None:
    """在此处读取流事件；同时累积推理与正文原文，供 Trace 完整落盘。"""

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
            _append_stream_content(metrics, "reasoning", part.content)
        elif isinstance(part, TextPart):
            _count_stream_content(metrics, "text", part.content, now)
            _append_stream_content(metrics, "text", part.content)
        elif isinstance(part, ToolCallPart):
            metrics["tool_call_chunks"] += 1
            metrics["first_tool_call_at"] = metrics["first_tool_call_at"] or now
    elif isinstance(event, PartDeltaEvent):
        delta = event.delta
        if isinstance(delta, ThinkingPartDelta):
            _count_stream_content(metrics, "reasoning", delta.content_delta, now)
            _append_stream_content(metrics, "reasoning", delta.content_delta)
        elif isinstance(delta, TextPartDelta):
            _count_stream_content(metrics, "text", delta.content_delta, now)
            _append_stream_content(metrics, "text", delta.content_delta)
        elif isinstance(delta, ToolCallPartDelta):
            metrics["tool_call_chunks"] += 1
            metrics["first_tool_call_at"] = metrics["first_tool_call_at"] or now


def _append_stream_content(
    metrics: dict[str, Any],
    kind: str,
    content: str | None,
) -> None:
    if not content or not metrics.get("record_content", False):
        return
    metrics[f"{kind}_content"].append(content)


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

    payload: dict[str, Any] = {
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
    }
    if metrics["record_content"]:
        payload.update(
            {
                "reasoning_content": "".join(metrics["reasoning_content"]),
                "text_content": "".join(metrics["text_content"]),
                "reasoning_content_recorded": True,
                "text_content_recorded": True,
            }
        )
    else:
        payload.update(
            {
                "reasoning_content_recorded": False,
                "text_content_recorded": False,
            }
        )
    return payload


def _sanitize(
    value: Any,
    config: TraceConfig,
    key: str | None = None,
    *,
    max_chars: int | None = None,
    max_items: int | None = None,
) -> Any:
    char_limit = max_chars if max_chars is not None else config.max_string_chars
    item_limit = max_items if max_items is not None else config.max_list_items
    if key and any(marker in key.lower() for marker in ("api_key", "authorization", "password", "secret")):
        return "<redacted>"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize(
                item_value,
                config,
                str(item_key),
                max_chars=max_chars,
                max_items=max_items,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        selected = list(value[:item_limit])
        result = [
            _sanitize(item, config, max_chars=max_chars, max_items=max_items)
            for item in selected
        ]
        if len(value) > item_limit:
            result.append({"truncated_items": len(value) - item_limit})
        return result
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str) and len(value) > char_limit:
        return value[:char_limit] + f"…<truncated {len(value) - char_limit} chars>"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return [repr(item) for item in value]
    if isinstance(value, (complex, Path)):
        return str(value)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_bytes(encoded: str) -> int:
    return len(encoded.encode("utf-8"))


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(payload),
        "value_types": {key: type(value).__name__ for key, value in payload.items()},
    }
