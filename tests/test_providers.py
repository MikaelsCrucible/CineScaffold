from __future__ import annotations

import json
from io import BytesIO
import unittest
from unittest.mock import patch
from typing import Any
from urllib.error import HTTPError

from cinescaffold.errors import (
    ProviderError,
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderTimeoutError,
)
from cinescaffold.providers.deepseek import DeepSeekProvider
from cinescaffold.providers.http import post_json
from cinescaffold.providers.openai import OpenAIProvider
from cinescaffold.schema import load_schema
from tests.helpers import ROOT, valid_model_output


class ProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema(ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json")
        self.output = valid_model_output()

    def test_openai_uses_responses_structured_output(self) -> None:
        captured: dict[str, Any] = {}

        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "id": "resp_test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(self.output)}
                        ],
                    }
                ],
            }

        provider = OpenAIProvider("secret", "test-model", transport=transport)
        response = provider.generate("系统", "用户", self.schema)

        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["payload"]["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            captured["payload"]["text"]["format"]["name"],
            "cinematic_brief_v0_4",
        )
        self.assertTrue(captured["payload"]["text"]["format"]["strict"])
        self.assertNotIn("$schema", captured["payload"]["text"]["format"]["schema"])
        self.assertEqual(captured["payload"]["max_output_tokens"], 8192)
        self.assertNotIn("reasoning", captured["payload"])
        self.assertFalse(captured["payload"]["store"])
        self.assertEqual(response.response_id, "resp_test")

    def test_openai_uses_requested_reasoning_effort_and_output_limit(self) -> None:
        captured: dict[str, Any] = {}

        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            captured["payload"] = payload
            return {
                "id": "resp_test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(self.output)}
                        ],
                    }
                ],
            }

        provider = OpenAIProvider(
            "secret",
            "gpt-5.6-luna",
            max_tokens=65536,
            reasoning_effort="medium",
            transport=transport,
        )
        provider.generate("系统", "用户", self.schema)

        self.assertEqual(captured["payload"]["max_output_tokens"], 65536)
        self.assertEqual(captured["payload"]["reasoning"], {"effort": "medium"})

    def test_deepseek_uses_chat_json_output(self) -> None:
        captured: dict[str, Any] = {}

        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "id": "chat_test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.output)},
                    }
                ],
            }

        provider = DeepSeekProvider("secret", "test-model", transport=transport)
        response = provider.generate("系统", "用户", self.schema)

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured["payload"])
        self.assertIn("JSON Schema", captured["payload"]["messages"][0]["content"])
        self.assertEqual(response.response_id, "chat_test")

    def test_deepseek_semantic_thinking_can_be_enabled_explicitly(self) -> None:
        captured: dict[str, Any] = {}

        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            captured["payload"] = payload
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.output)},
                    }
                ]
            }

        provider = DeepSeekProvider(
            "secret",
            "test-model",
            thinking_mode="enabled",
            reasoning_effort="low",
            transport=transport,
        )
        provider.generate("系统", "用户", self.schema)

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "low")

    def test_deepseek_empty_content_is_diagnostic_error(self) -> None:
        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            return {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}

        provider = DeepSeekProvider("secret", "test-model", transport=transport)
        with self.assertRaisesRegex(ProviderError, "空 JSON"):
            provider.generate("系统", "用户", self.schema)

    def test_deepseek_truncated_content_is_diagnostic_error(self) -> None:
        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            return {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "{\"summary\":"}}
                ]
            }

        provider = DeepSeekProvider("secret", "test-model", transport=transport)
        with self.assertRaisesRegex(ProviderError, "截断"):
            provider.generate("系统", "用户", self.schema)

    def test_openai_incomplete_response_is_diagnostic_error(self) -> None:
        def transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            return {"id": "resp_test", "status": "incomplete", "output": []}

        provider = OpenAIProvider("secret", "test-model", transport=transport)
        with self.assertRaisesRegex(ProviderError, "未完成"):
            provider.generate("系统", "用户", self.schema)

    def test_http_provider_rejections_are_reported_without_retrying(self) -> None:
        cases = (
            (401, "invalid authentication", "provider_authentication_failed"),
            (429, "rate limit exceeded", "provider_rate_limited"),
            (402, "insufficient balance", "provider_balance_exhausted"),
            (503, "server overloaded", "provider_overloaded"),
        )
        for status_code, detail, failure_code in cases:
            with self.subTest(status_code=status_code):
                error = HTTPError(
                    "https://provider.invalid/v1/test",
                    status_code,
                    "provider rejected request",
                    hdrs=None,
                    fp=BytesIO(detail.encode("utf-8")),
                )
                with patch(
                    "cinescaffold.providers.http.urlopen",
                    side_effect=error,
                ) as mocked_urlopen:
                    with self.assertRaisesRegex(
                        ProviderError,
                        rf"HTTP {status_code}.*{detail}",
                    ) as raised:
                        post_json(
                            "https://provider.invalid/v1/test",
                            {"Authorization": "Bearer test-secret"},
                            {"model": "test-model"},
                            1.0,
                        )

                mocked_urlopen.assert_called_once()
                self.assertIsInstance(raised.exception, ProviderHTTPError)
                self.assertEqual(raised.exception.status_code, status_code)
                self.assertEqual(raised.exception.failure_code, failure_code)
                self.assertEqual(
                    raised.exception.confirmed_not_billed,
                    status_code != 503,
                )

    def test_http_provider_timeout_is_normalized(self) -> None:
        with patch(
            "cinescaffold.providers.http.urlopen",
            side_effect=TimeoutError("simulated timeout"),
        ) as mocked_urlopen:
            with self.assertRaisesRegex(ProviderTimeoutError, "墙钟超时"):
                post_json(
                    "https://provider.invalid/v1/test",
                    {"Authorization": "Bearer test-secret"},
                    {"model": "test-model"},
                    0.1,
                )

        mocked_urlopen.assert_called_once()

    def test_http_keep_alive_bytes_do_not_extend_wall_clock_timeout(self) -> None:
        response = _ChunkedResponse([b"\n", b"\n", b"\n"])
        with (
            patch("cinescaffold.providers.http.urlopen", return_value=response),
            patch(
                "cinescaffold.providers.http.time.monotonic",
                side_effect=(0.0, 0.2, 0.8, 1.01),
            ),
        ):
            with self.assertRaisesRegex(ProviderTimeoutError, "墙钟超时"):
                post_json(
                    "https://provider.invalid/v1/test",
                    {"Authorization": "Bearer test-secret"},
                    {"model": "test-model"},
                    1.0,
                )

        self.assertEqual(response.read_count, 2)

    def test_http_incremental_reader_accepts_keep_alive_before_json(self) -> None:
        response = _ChunkedResponse([b"\n", b'{"ok":', b"true}", b""])
        with (
            patch("cinescaffold.providers.http.urlopen", return_value=response),
            patch(
                "cinescaffold.providers.http.time.monotonic",
                side_effect=(0.0, 0.1, 0.2, 0.3, 0.4),
            ),
        ):
            result = post_json(
                "https://provider.invalid/v1/test",
                {"Authorization": "Bearer test-secret"},
                {"model": "test-model"},
                1.0,
            )

        self.assertEqual(result, {"ok": True})

    def test_http_network_error_is_typed(self) -> None:
        from urllib.error import URLError

        with patch(
            "cinescaffold.providers.http.urlopen",
            side_effect=URLError("unreachable"),
        ):
            with self.assertRaisesRegex(ProviderNetworkError, "网络错误"):
                post_json(
                    "https://provider.invalid/v1/test",
                    {},
                    {"model": "test-model"},
                    1.0,
                )


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)
        self.read_count = 0

    def __enter__(self) -> "_ChunkedResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read1(self, _size: int) -> bytes:
        self.read_count += 1
        return next(self._chunks)

    def read(self, _size: int = -1) -> bytes:
        raise AssertionError("incremental read1 must be used")


if __name__ == "__main__":
    unittest.main()
