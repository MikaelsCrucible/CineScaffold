from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    content: dict[str, Any]
    response_id: str | None
    raw_metadata: dict[str, Any]


class StructuredOutputProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> ProviderResponse:
        ...
