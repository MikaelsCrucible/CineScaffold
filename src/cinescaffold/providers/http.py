from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cinescaffold.errors import ProviderError, ProviderHTTPError


HttpTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(error.code, detail) from error
    except URLError as error:
        raise ProviderError(f"API 网络错误：{error.reason}") from error
    except TimeoutError as error:
        raise ProviderError("API 请求超时") from error

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProviderError("API 返回的响应不是合法 JSON") from error
    if not isinstance(value, dict):
        raise ProviderError("API 响应根节点不是对象")
    return value
