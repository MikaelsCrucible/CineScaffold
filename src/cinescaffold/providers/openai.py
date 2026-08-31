from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from cinescaffold.errors import ProviderError
from cinescaffold.providers.base import ProviderResponse
from cinescaffold.providers.http import HttpTransport, post_json


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_tokens: int = 8192,
        reasoning_effort: Literal[
            "none", "low", "medium", "high", "xhigh", "max"
        ] | None = None,
        transport: HttpTransport = post_json,
    ) -> None:
        if not api_key:
            raise ProviderError("缺少 OPENAI_API_KEY")
        if not model:
            raise ProviderError("OpenAI Provider 必须指定模型")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.transport = transport

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "cinematic_brief_v0_3",
                    "strict": True,
                    "schema": _prepare_api_schema(schema),
                }
            },
            "max_output_tokens": self.max_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        response = self.transport(
            f"{self.base_url}/responses",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
        content = _extract_output_text(response)
        return ProviderResponse(
            content=_parse_json_object(content),
            response_id=_optional_string(response.get("id")),
            raw_metadata={"status": response.get("status"), "usage": response.get("usage")},
        )


def _extract_output_text(response: dict[str, Any]) -> str:
    if response.get("status") not in (None, "completed"):
        raise ProviderError(f"OpenAI 响应未完成：{response.get('status')}")
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "refusal":
                raise ProviderError(f"OpenAI 拒绝请求：{part.get('refusal', '')}")
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise ProviderError("OpenAI 响应中没有可用的 output_text")


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderError("OpenAI 结构化输出不是合法 JSON") from error
    if not isinstance(value, dict):
        raise ProviderError("OpenAI 结构化输出根节点不是对象")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _prepare_api_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """移除不参与约束的 Schema 文档元数据。"""
    prepared = deepcopy(schema)
    prepared.pop("$schema", None)
    prepared.pop("$id", None)
    prepared.pop("title", None)
    return prepared
