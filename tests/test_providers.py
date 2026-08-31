from __future__ import annotations

import json
import unittest
from typing import Any

from cinescaffold.errors import ProviderError
from cinescaffold.providers.deepseek import DeepSeekProvider
from cinescaffold.providers.openai import OpenAIProvider
from cinescaffold.schema import load_schema
from tests.helpers import ROOT, valid_model_output


class ProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema(ROOT / "schemas/cinematic_brief_model_output.schema.json")
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
            "cinematic_brief_v0_3",
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


if __name__ == "__main__":
    unittest.main()
