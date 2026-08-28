from __future__ import annotations

from copy import deepcopy
from typing import Any

from cinescaffold.providers.base import ProviderResponse


class MockProvider:
    name = "mock"
    model = "mock-cinematic-brief-v0.1"

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
            }
        )
        return ProviderResponse(
            content=deepcopy(self.response),
            response_id="mock-response-001",
            raw_metadata={"mock": True},
        )
