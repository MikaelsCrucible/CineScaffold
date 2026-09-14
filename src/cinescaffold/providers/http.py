from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cinescaffold.errors import (
    ProviderError,
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderTimeoutError,
)


HttpTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url=url, data=body, headers=headers, method="POST")
    deadline = time.monotonic() + timeout
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = _read_response_body(response, deadline).decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(error.code, detail) from error
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise ProviderTimeoutError("API 请求超过墙钟超时") from error
        raise ProviderNetworkError(f"API 网络错误：{error.reason}") from error
    except TimeoutError as error:
        raise ProviderTimeoutError("API 请求超过墙钟超时") from error

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProviderError("API 返回的响应不是合法 JSON") from error
    if not isinstance(value, dict):
        raise ProviderError("API 响应根节点不是对象")
    return value


def _read_response_body(response: Any, deadline: float) -> bytes:
    """Read incrementally so keep-alive whitespace cannot renew the deadline."""

    chunks: list[bytes] = []
    read = getattr(response, "read1", response.read)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeoutError("API 请求超过墙钟超时")
        _set_response_socket_timeout(response, remaining)
        chunk = read(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _set_response_socket_timeout(response: Any, timeout: float) -> None:
    # urllib does not expose a total-response deadline. Its socket is available
    # through HTTPResponse.fp on CPython, so cap each read by the remaining time.
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(timeout)
