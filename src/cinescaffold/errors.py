class CineScaffoldError(Exception):
    """项目内可预期错误的基类。"""


class ConfigurationError(CineScaffoldError):
    """配置缺失或无效。"""


class ProviderError(CineScaffoldError):
    """模型服务调用或响应错误。"""


class SchemaValidationError(CineScaffoldError):
    """结构化输出不符合 Schema。"""


class PromptTemplateError(CineScaffoldError):
    """提示模板缺少必要占位符。"""


class ExecutionError(CineScaffoldError):
    """Blender 执行链路中的可预期错误。"""
