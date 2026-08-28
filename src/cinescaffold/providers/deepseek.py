from __future__ import annotations

import json
from typing import Any

from cinescaffold.errors import ProviderError
from cinescaffold.providers.base import ProviderResponse
from cinescaffold.providers.http import HttpTransport, post_json


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        max_tokens: int = 8192,
        transport: HttpTransport = post_json,
    ) -> None:
        if not api_key:
            raise ProviderError("缺少 DEEPSEEK_API_KEY")
        if not model:
            raise ProviderError("DeepSeek Provider 必须指定模型")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.transport = transport

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\n必须遵循的 JSON Schema：\n{schema_text}",
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        response = self.transport(
            f"{self.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
        content = _extract_message_content(response)
        return ProviderResponse(
            content=_parse_json_object(content),
            response_id=_optional_string(response.get("id")),
            raw_metadata={"usage": response.get("usage")},
        )


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderError("DeepSeek 响应中没有 choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ProviderError("DeepSeek JSON 输出被 max_tokens 截断")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("DeepSeek 返回了空 JSON 内容")
    return content


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderError("DeepSeek JSON Output 不是合法 JSON") from error
    if not isinstance(value, dict):
        raise ProviderError("DeepSeek JSON Output 根节点不是对象")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
