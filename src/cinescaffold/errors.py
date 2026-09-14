class CineScaffoldError(Exception):
    """项目内可预期错误的基类。"""


class ConfigurationError(CineScaffoldError):
    """配置缺失或无效。"""


class ProviderError(CineScaffoldError):
    """模型服务调用或响应错误。"""


_CONFIRMED_NONBILLABLE_HTTP_STATUSES = frozenset(
    {400, 401, 402, 403, 404, 405, 409, 413, 415, 422, 429}
)


class ProviderHTTPError(ProviderError):
    """Provider returned a typed HTTP rejection instead of a model response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.confirmed_not_billed = http_status_is_confirmed_not_billed(status_code)
        super().__init__(f"API 返回 HTTP {status_code}：{detail}")


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
