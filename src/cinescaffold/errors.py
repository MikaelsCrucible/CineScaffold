class CineScaffoldError(Exception):
    """项目内可预期错误的基类。"""


class ConfigurationError(CineScaffoldError):
    """配置缺失或无效。"""


class ProviderError(CineScaffoldError):
    """模型服务调用或响应错误。"""

    failure_code = "provider_error"
    retryable = False


_CONFIRMED_NONBILLABLE_HTTP_STATUSES = frozenset(
    {400, 401, 402, 403, 404, 405, 409, 413, 415, 422, 429}
)


class ProviderHTTPError(ProviderError):
    """Provider returned a typed HTTP rejection instead of a model response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.confirmed_not_billed = http_status_is_confirmed_not_billed(status_code)
        self.failure_code = provider_http_failure_code(status_code)
        self.retryable = status_code in {408, 429, 500, 502, 503, 504}
        super().__init__(f"API 返回 HTTP {status_code}：{detail}")


class ProviderTimeoutError(ProviderError):
    """Provider did not complete within the configured wall-clock deadline."""

    failure_code = "provider_timeout"
    retryable = True


class ProviderNetworkError(ProviderError):
    """Provider could not be reached due to a transport-level failure."""

    failure_code = "provider_network_error"
    retryable = True


def provider_http_failure_code(status_code: int) -> str:
    """Map a typed HTTP rejection to a stable, secret-free failure code."""

    if status_code in {401, 403}:
        return "provider_authentication_failed"
    if status_code == 402:
        return "provider_balance_exhausted"
    if status_code == 429:
        return "provider_rate_limited"
    if status_code == 503:
        return "provider_overloaded"
    if status_code in {408, 500, 502, 504}:
        return "provider_unavailable"
    if 400 <= status_code < 500:
        return "provider_request_rejected"
    return "provider_http_error"


def http_status_is_confirmed_not_billed(status_code: object) -> bool:
    """Return true only for request-layer rejections with no inference result."""

    return (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and status_code in _CONFIRMED_NONBILLABLE_HTTP_STATUSES
    )


class SchemaValidationError(CineScaffoldError):
    """结构化输出不符合 Schema。"""


class PromptTemplateError(CineScaffoldError):
    """提示模板缺少必要占位符。"""


class ExecutionError(CineScaffoldError):
    """Blender 执行链路中的可预期错误。"""
