from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from cinescaffold.config import LoadedConfig, SECRET_KEYS, public_config_values
from cinescaffold.platforms import open_local_path
from cinescaffold.textual_six import (
    SECTION_FIELDS,
    TextualSixDimensions,
    parse_textual_six,
)
from cinescaffold.ui_state import (
    UiRunInputs,
    UiRunMetrics,
    UiSessionController,
    event_message,
    event_progress,
)
from cinescaffold.workflow import WorkflowRunner


PROVIDER_OPTIONS = {"mock": "Mock（离线）", "openai": "OpenAI", "deepseek": "DeepSeek"}
STAGE_PROVIDER_OPTIONS = {"": "沿用通用默认", **PROVIDER_OPTIONS}
THINKING_OPTIONS = {"": "供应商默认", "disabled": "关闭", "enabled": "开启"}
EFFORT_OPTIONS = {
    "": "供应商默认",
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def launch_ui(
    loaded_config: LoadedConfig,
    *,
    config_path: Path,
    project_root: Path,
    host: str,
    port: int,
    show: bool,
) -> None:
    """启动只监听本机的 CineScaffold 浏览器界面。"""

    try:
        from nicegui import ui
    except ImportError as error:
        raise RuntimeError(
            "UI 依赖尚未安装；请使用 requirements-ui.lock 或 pip install -e '.[ui]'"
        ) from error

    controller = UiSessionController(
        loaded_config=loaded_config,
        config_path=config_path,
        project_root=project_root,
    )
    _register_page(ui, controller)
    ui.run(
        host=host,
        port=port,
        title="CineScaffold Studio",
        language="zh-CN",
        dark=True,
        show=show,
        reload=False,
        uvicorn_logging_level="warning",
        show_welcome_message=False,
    )


def _register_page(ui: Any, controller: UiSessionController) -> None:
    ui.add_css(_CSS, shared=True)

    @ui.page("/")
    def index() -> None:
        state: dict[str, Any] = {
            "running": False,
            "started": 0.0,
            "progress": 0.0,
            "summary": None,
        }
        settings_inputs: dict[str, Any] = {}

        with ui.header().classes("cs-header items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("movie_filter", size="28px").classes("text-amber-300")
                with ui.column().classes("gap-0"):
                    ui.label("CineScaffold Studio").classes("text-lg font-semibold tracking-wide")
                    ui.label("从电影意图到可验证白模预演").classes("text-xs text-slate-400")
            with ui.row().classes("items-center gap-2"):
                config_chip = ui.chip(
                    "配置已加载" if controller.loaded_config.path else "使用默认配置",
                    icon="verified_user",
                    color="positive" if controller.loaded_config.path else "grey-8",
                ).props("outline")
                ui.button("设置", icon="tune", on_click=lambda: settings_dialog.open()).props(
                    "flat"
                )

        with ui.column().classes("w-full max-w-[1500px] mx-auto px-5 py-6 gap-5"):
            with ui.row().classes("w-full items-end justify-between hero-panel"):
                with ui.column().classes("gap-2 max-w-3xl"):
                    ui.label("把想法变成能被镜头验证的空间").classes("hero-title")
                    ui.label(
                        "选择任意中间表示作为起点；每一步都保存、计时并记录成本。"
                    ).classes("text-slate-300 text-base")
                status_badge = ui.badge("准备就绪", color="grey-8").classes("status-badge")

            with ui.row().classes("w-full gap-5 items-stretch cs-main-grid"):
                with ui.card().classes("cs-card cs-builder"):
                    _section_title(ui, "01", "选择起点", "四种入口，共用同一条执行管线")
                    source_toggle = ui.toggle(
                        {
                            "text": "自然语言",
                            "textual_six": "文本六维",
                            "brief": "Cinematic Brief",
                            "scene_ir": "Scene IR",
                        },
                        value="text",
                    ).props("spread no-caps").classes("w-full source-toggle")

                    input_panels: dict[str, Any] = {}
                    input_elements: dict[str, Any] = {}
                    textual_six_inputs: dict[str, Any] = {}
                    with ui.column().classes("w-full mt-3"):
                        with ui.column().classes("w-full gap-2") as natural_panel:
                            input_panels["text"] = natural_panel
                            input_elements["text"] = ui.textarea(
                                "自然语言描述",
                                placeholder="例如：一个男人站在荒漠里，远处有巨大的飞船……",
                            ).props("outlined autogrow").classes("w-full cs-textarea")
                            ui.label("系统会同时生成并展示文本六维与 JSON 六维。 ").classes(
                                "helper"
                            )
                        with ui.column().classes("w-full gap-2") as six_panel:
                            input_panels["textual_six"] = six_panel
                            ui.label("六个维度标签固定；只需填写每项正文。 ").classes("helper")
                            for field, label in SECTION_FIELDS:
                                with ui.column().classes("w-full gap-1 textual-six-section"):
                                    ui.label(f"{label}：").classes("textual-six-label")
                                    textual_six_inputs[field] = ui.textarea(
                                        placeholder=f"填写{label}",
                                    ).props("outlined autogrow hide-bottom-space").classes(
                                        "w-full textual-six-input"
                                    )
                            _upload_textual_six(ui, textual_six_inputs)
                        with ui.column().classes("w-full gap-2") as brief_panel:
                            input_panels["brief"] = brief_panel
                            input_elements["brief"] = ui.textarea(
                                "Cinematic Brief JSON",
                                placeholder="粘贴 JSON，或从文件导入",
                            ).props("outlined autogrow").classes("w-full cs-textarea mono-input")
                            _upload(ui, input_elements["brief"], ".json", "导入 Brief JSON")
                        with ui.column().classes("w-full gap-2") as ir_panel:
                            input_panels["scene_ir"] = ir_panel
                            input_elements["scene_ir"] = ui.textarea(
                                "Scene IR JSON",
                                placeholder="粘贴已提交的 Scene IR，或从文件导入",
                            ).props("outlined autogrow").classes("w-full cs-textarea mono-input")
                            _upload(ui, input_elements["scene_ir"], ".json", "导入 Scene IR")

                    def show_source(value: str) -> None:
                        for kind, panel in input_panels.items():
                            panel.set_visibility(kind == value)

                    source_toggle.on_value_change(lambda event: show_source(str(event.value)))
                    show_source("text")

                    ui.separator().classes("my-2 opacity-30")
                    _section_title(ui, "02", "输出设置", "默认生成快速 Workbench 预演")
                    default_output = time.strftime("runs/ui/%Y%m%d-%H%M%S")
                    output_input = ui.input(
                        "输出目录",
                        value=default_output,
                    ).props("outlined").classes("w-full")
                    with ui.row().classes("w-full gap-4"):
                        render_select = ui.select(
                            {"preview": "快速预览", "control": "完整控制视频"},
                            value="preview",
                            label="渲染规格",
                        ).props("outlined").classes("grow")
                        overwrite_switch = ui.switch("覆盖本次管线产物", value=False).classes(
                            "self-center"
                        )
                    run_button = ui.button(
                        "开始生成白模视频",
                        icon="play_arrow",
                    ).classes("w-full run-button").props("unelevated no-caps")

                with ui.column().classes("gap-5 cs-monitor"):
                    with ui.card().classes("cs-card w-full"):
                        _section_title(ui, "LIVE", "运行进度", "可关闭页面前先等待任务完成")
                        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
                        with ui.row().classes("w-full justify-between gap-2 stage-row"):
                            stage_labels = {
                                "semantic": _stage_pill(ui, "六维解析"),
                                "planning": _stage_pill(ui, "场景规划"),
                                "execution": _stage_pill(ui, "Blender"),
                            }
                        with ui.grid(columns=2).classes("w-full gap-3 mt-2 metric-grid"):
                            elapsed_value = _metric(ui, "总耗时", "0.0 s")
                            token_value = _metric(ui, "Tokens", "0")
                            semantic_cost_value = _metric(ui, "六维成本", "—")
                            planning_cost_value = _metric(ui, "规划成本", "—")
                        event_log = ui.log(max_lines=240).classes("w-full event-log")

                    with ui.card().classes("cs-card w-full") as semantic_result_card:
                        semantic_result_card.set_visibility(False)
                        _section_title(ui, "DATA", "六维语义", "人类表示与机器权威表示并列保存")
                        with ui.tabs().classes("w-full") as result_tabs:
                            text_tab = ui.tab("文本六维")
                            json_tab = ui.tab("JSON 六维")
                        with ui.tab_panels(result_tabs, value=text_tab).classes("w-full result-tabs"):
                            with ui.tab_panel(text_tab):
                                textual_result = ui.textarea(value="").props("readonly outlined autogrow").classes(
                                    "w-full mono-input"
                                )
                            with ui.tab_panel(json_tab):
                                json_result = ui.textarea(value="").props("readonly outlined autogrow").classes(
                                    "w-full mono-input"
                                )

                    with ui.card().classes("cs-card w-full") as artifact_card:
                        artifact_card.set_visibility(False)
                        _section_title(ui, "DONE", "生成结果", "直接预览，或打开本地产物")
                        artifact_content = ui.column().classes("w-full gap-3")

        with ui.dialog() as settings_dialog, ui.card().classes("settings-dialog"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-0"):
                    ui.label("运行设置").classes("text-xl font-semibold")
                    ui.label("空白 API Key 表示保留现有密钥；页面不会读取其明文。 ").classes(
                        "helper"
                    )
                ui.button(icon="close", on_click=settings_dialog.close).props("flat round")
            with ui.scroll_area().classes("w-full settings-scroll"):
                _settings_form(ui, controller, settings_inputs)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=settings_dialog.close).props("flat")
                ui.button(
                    "仅应用本次",
                    icon="bolt",
                    on_click=lambda: apply_settings(False),
                ).props("outline")
                ui.button(
                    "保存到配置文件",
                    icon="save",
                    on_click=lambda: apply_settings(True),
                )

        def collect_settings() -> dict[str, str | None]:
            updates: dict[str, str | None] = {}
            for key, element in settings_inputs.items():
                value = element.value
                if key in SECRET_KEYS and not str(value or "").strip():
                    updates[key] = None
                else:
                    updates[key] = str(value or "").strip()
            return updates

        def apply_settings(save: bool) -> None:
            try:
                updates = collect_settings()
                if save:
                    loaded = controller.save_settings(updates)
                    config_chip.set_text("配置已保存")
                    config_chip.props("color=positive")
                    ui.notify(f"已安全保存到 {loaded.path}", type="positive")
                else:
                    controller.apply_settings(updates)
                    config_chip.set_text("本次设置已应用")
                    config_chip.props("color=warning")
                    ui.notify("设置仅对当前 UI 会话生效", type="positive")
                settings_dialog.close()
            except Exception as error:
                ui.notify(str(error), type="negative", multi_line=True)

        def set_stage(name: str, status: str) -> None:
            label = stage_labels[name]
            label.classes(remove="stage-idle stage-active stage-done stage-error")
            label.classes(f"stage-{status}")

        def on_pipeline_event(event_type: str, payload: dict[str, Any]) -> None:
            state["progress"] = event_progress(event_type, state["progress"])
            progress.set_value(state["progress"])
            message = event_message(event_type, payload)
            if message:
                event_log.push(message)
            if event_type == "pipeline_semantic_started":
                set_stage("semantic", "active")
            elif event_type == "pipeline_semantic_completed":
                set_stage("semantic", "done")
                _load_semantic_artifacts(
                    payload,
                    textual_result,
                    json_result,
                    semantic_result_card,
                )
            elif event_type == "pipeline_input_ready":
                set_stage("semantic", "done")
            elif event_type == "pipeline_planning_started":
                set_stage("planning", "active")
            elif event_type == "pipeline_planning_completed":
                set_stage("planning", "done" if payload.get("status") == "success" else "error")
            elif event_type == "pipeline_execution_started":
                set_stage("execution", "active")
            elif event_type == "pipeline_execution_completed":
                set_stage("execution", "done" if payload.get("status") == "success" else "error")
            elif event_type == "pipeline_failed":
                stage = payload.get("stage")
                if stage in stage_labels:
                    set_stage(stage, "error")

        async def run_pipeline() -> None:
            if state["running"]:
                return
            try:
                textual_six = ""
                if source_toggle.value == "textual_six":
                    textual_six = TextualSixDimensions.from_field_values(
                        {
                            field: textual_six_inputs[field].value
                            for field, _ in SECTION_FIELDS
                        }
                    ).render()
                inputs = UiRunInputs(
                    source_kind=source_toggle.value,
                    natural_text=input_elements["text"].value or "",
                    textual_six=textual_six,
                    brief_json=input_elements["brief"].value or "",
                    scene_ir_json=input_elements["scene_ir"].value or "",
                    output_dir=output_input.value or "",
                    render_profile=render_select.value,
                    overwrite=overwrite_switch.value,
                )
                source, run_config = controller.build_run(inputs)
                state.update(running=True, started=time.monotonic(), progress=0.0, summary=None)
                run_button.disable()
                status_badge.set_text("正在运行")
                status_badge.props("color=warning")
                event_log.clear()
                artifact_card.set_visibility(False)
                semantic_result_card.set_visibility(False)
                progress.set_value(0)
                for stage in stage_labels:
                    set_stage(stage, "idle")
                summary = await WorkflowRunner(
                    run_config,
                    progress_callback=on_pipeline_event,
                ).run(source)
                state["summary"] = summary
                metrics = UiRunMetrics.from_summary(summary)
                elapsed_value.set_text(f"{metrics.elapsed_seconds:.1f} s")
                token_value.set_text(f"{metrics.semantic_tokens + metrics.planning_tokens:,}")
                semantic_cost_value.set_text(metrics.semantic_cost)
                planning_cost_value.set_text(metrics.planning_cost)
                _render_artifacts(ui, summary, artifact_card, artifact_content)
                if summary.get("status") == "success":
                    status_badge.set_text("生成成功")
                    status_badge.props("color=positive")
                    ui.notify("白模视频生成完成", type="positive")
                else:
                    status_badge.set_text("运行失败")
                    status_badge.props("color=negative")
                    ui.notify(str(summary.get("error") or "运行失败"), type="negative", multi_line=True)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                status_badge.set_text("输入或配置错误")
                status_badge.props("color=negative")
                ui.notify(str(error), type="negative", multi_line=True)
            finally:
                state["running"] = False
                run_button.enable()

        def update_elapsed() -> None:
            if state["running"]:
                elapsed_value.set_text(f"{time.monotonic() - state['started']:.1f} s")

        run_button.on_click(run_pipeline)
        ui.timer(0.5, update_elapsed)


def _section_title(ui: Any, eyebrow: str, title: str, subtitle: str) -> None:
    with ui.row().classes("w-full items-start gap-3"):
        ui.label(eyebrow).classes("eyebrow")
        with ui.column().classes("gap-0"):
            ui.label(title).classes("section-title")
            ui.label(subtitle).classes("helper")


def _stage_pill(ui: Any, text: str) -> Any:
    return ui.label(text).classes("stage-pill stage-idle")


def _metric(ui: Any, label: str, value: str) -> Any:
    with ui.column().classes("metric-card gap-1"):
        ui.label(label).classes("helper")
        return ui.label(value).classes("metric-value")


def _upload(ui: Any, target: Any, accept: str, label: str) -> None:
    async def uploaded(event: Any) -> None:
        try:
            target.set_value(await event.file.text())
            ui.notify(f"已导入 {event.file.name}", type="positive")
        except Exception as error:
            ui.notify(f"读取文件失败：{error}", type="negative")

    ui.upload(
        label=label,
        on_upload=uploaded,
        auto_upload=True,
        max_file_size=10_000_000,
    ).props(f'accept="{accept}" flat color=grey-5').classes("compact-upload")


def _upload_textual_six(ui: Any, targets: dict[str, Any]) -> None:
    async def uploaded(event: Any) -> None:
        try:
            parsed = parse_textual_six(await event.file.text())
            for field, value in parsed.field_values().items():
                targets[field].set_value(value)
            ui.notify(f"已导入 {event.file.name}", type="positive")
        except Exception as error:
            ui.notify(f"文本六维无效：{error}", type="negative", multi_line=True)

    ui.upload(
        label="导入文本六维",
        on_upload=uploaded,
        auto_upload=True,
        max_file_size=10_000_000,
    ).props('accept=".txt" flat color=grey-5').classes("compact-upload")


def _settings_form(ui: Any, controller: UiSessionController, elements: dict[str, Any]) -> None:
    values = public_config_values(controller.loaded_config)

    def text(key: str, label: str, *, password: bool = False) -> Any:
        value = "" if password else str(values.get(key, ""))
        placeholder = "已配置；留空保持不变" if password and values.get(f"{key}_configured") else ""
        element = ui.input(label, value=value, placeholder=placeholder).props(
            "outlined dense" + (" type=password" if password else "")
        ).classes("w-full")
        elements[key] = element
        return element

    def select(key: str, label: str, options: dict[str, str]) -> Any:
        value = str(values.get(key, ""))
        element = ui.select(options, value=value if value in options else "", label=label).props(
            "outlined dense emit-value map-options"
        ).classes("w-full")
        elements[key] = element
        return element

    with ui.expansion("模型与凭据", icon="key", value=True).classes("w-full settings-group"):
        with ui.grid(columns=2).classes("w-full gap-3 settings-grid"):
            select("provider", "通用默认 Provider", PROVIDER_OPTIONS)
            text("model", "通用默认模型")
            text("api_key", "通用默认 API Key", password=True)
            text("base_url", "通用默认 Base URL")
            text("openai_api_key", "OpenAI API Key", password=True)
            text("openai_base_url", "OpenAI Base URL")
            text("deepseek_api_key", "DeepSeek API Key", password=True)
            text("deepseek_base_url", "DeepSeek Base URL")
    with ui.expansion("自然语言 / 文本六维解析", icon="translate").classes("w-full settings-group"):
        with ui.grid(columns=2).classes("w-full gap-3 settings-grid"):
            select("semantic_provider", "Provider", STAGE_PROVIDER_OPTIONS)
            text("semantic_model", "模型")
            select("semantic_thinking_mode", "Thinking", THINKING_OPTIONS)
            select("semantic_reasoning_effort", "推理强度", EFFORT_OPTIONS)
            text("semantic_max_tokens", "最大输出 Tokens")
            text("semantic_timeout", "请求超时（秒）")
            text("semantic_input_cost_per_million", "输入价格 / 1M")
            text("semantic_output_cost_per_million", "输出价格 / 1M")
            text("semantic_cache_read_cost_per_million", "缓存读取价格 / 1M")
            text("semantic_cache_write_cost_per_million", "缓存写入价格 / 1M")
            text("semantic_cost_currency", "币种")
            text("semantic_price_source", "价格来源")
    with ui.expansion("场景规划 Agent", icon="account_tree").classes("w-full settings-group"):
        with ui.grid(columns=2).classes("w-full gap-3 settings-grid"):
            select("planning_provider", "Provider", STAGE_PROVIDER_OPTIONS)
            text("planning_model", "模型")
            select("planning_thinking_mode", "Thinking", THINKING_OPTIONS)
            select("planning_reasoning_effort", "推理强度", EFFORT_OPTIONS)
            text("planning_model_max_tokens", "单次响应 Tokens")
            text("planning_max_requests", "最多请求数")
            text("planning_max_tool_calls", "最多工具调用")
            text("planning_max_input_tokens", "累计输入 Tokens")
            text("planning_max_context_tokens", "单次上下文 Tokens")
            text("planning_max_output_tokens", "累计输出 Tokens")
            text("planning_max_total_tokens", "累计总 Tokens")
            text("planning_max_seconds", "最长运行秒数")
            text("planning_max_commit_attempts", "最多提交次数")
            text("planning_trace_max_event_bytes", "单个 Trace 事件字节")
            text("planning_trace_max_string_chars", "Trace 字符串长度")
            text("planning_input_cost_per_million", "输入价格 / 1M")
            text("planning_output_cost_per_million", "输出价格 / 1M")
            text("planning_cache_read_cost_per_million", "缓存读取价格 / 1M")
            text("planning_cache_write_cost_per_million", "缓存写入价格 / 1M")
            text("planning_cost_currency", "币种")
            text("planning_price_source", "价格来源")
    with ui.expansion("Blender 与渲染", icon="view_in_ar").classes("w-full settings-group"):
        with ui.grid(columns=2).classes("w-full gap-3 settings-grid"):
            text("execution_blender_path", "Blender 可执行文件")
            text("execution_mcp_command", "blender-mcp 可执行文件（可选）")
            select(
                "execution_build_backend",
                "场景构建后端",
                {"": "默认（Background）", "background": "Background", "mcp": "MCP"},
            )
            text("execution_build_timeout_seconds", "场景构建超时（秒）")
            select(
                "execution_render_backend",
                "渲染后端",
                {"": "默认（Background）", "background": "Background", "mcp": "MCP"},
            )
            select(
                "execution_render_profile",
                "默认渲染规格",
                {"": "默认", "preview": "Preview", "control": "Control"},
            )
            text("execution_render_timeout_seconds", "渲染超时（秒）")


def _load_semantic_artifacts(
    payload: dict[str, Any],
    textual_result: Any,
    json_result: Any,
    card: Any,
) -> None:
    textual_path = Path(payload["textual_six_path"])
    brief_path = Path(payload["brief_path"])
    textual_result.set_value(textual_path.read_text(encoding="utf-8"))
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    json_result.set_value(json.dumps(brief, ensure_ascii=False, indent=2))
    card.set_visibility(True)


def _render_artifacts(ui: Any, summary: dict[str, Any], card: Any, content: Any) -> None:
    artifacts = summary.get("artifacts") or {}
    output_dir = Path(summary.get("output_dir") or ".")
    content.clear()
    with content:
        video_value = artifacts.get("video")
        if video_value and Path(video_value).is_file():
            ui.video(Path(video_value), controls=True).classes("w-full result-video")
        else:
            ui.label("本次运行没有生成可预览视频。 ").classes("helper")
        with ui.row().classes("w-full gap-2"):
            ui.button(
                "打开输出文件夹",
                icon="folder_open",
                on_click=lambda: _open_with_notice(ui, output_dir),
            ).props("outline")
            scene_blend = output_dir / "execution/scene.blend"
            if scene_blend.is_file():
                ui.button(
                    "打开 Blender 场景",
                    icon="view_in_ar",
                    on_click=lambda: _open_with_notice(ui, scene_blend),
                ).props("outline")
        ui.label(f"输出目录：{output_dir}").classes("helper break-all")
    card.set_visibility(True)


def _open_with_notice(ui: Any, path: Path) -> None:
    try:
        open_local_path(path)
    except Exception as error:
        ui.notify(str(error), type="negative")


_CSS = """
:root { --cs-ink:#0b0f14; --cs-card:#121922; --cs-line:#263343; --cs-gold:#e7b65b; }
body { background: radial-gradient(circle at 18% -10%, #243141 0, #0b0f14 42%, #07090c 100%); color:#edf2f7; }
.cs-header { background:rgba(9,13,18,.9); border-bottom:1px solid #263343; backdrop-filter:blur(16px); padding:10px 24px; }
.hero-panel { padding:28px 30px; border:1px solid #2a3848; border-radius:22px; background:linear-gradient(125deg,rgba(37,49,64,.84),rgba(14,20,28,.92)); box-shadow:0 24px 80px rgba(0,0,0,.28); }
.hero-title { font-size:clamp(1.8rem,3vw,3rem); font-weight:650; letter-spacing:-.035em; line-height:1.05; }
.status-badge { font-size:.82rem; padding:8px 12px; }
.cs-main-grid { display:grid !important; grid-template-columns:minmax(0,1.25fr) minmax(390px,.75fr); align-items:start !important; }
.cs-builder,.cs-monitor { min-width:0; }
.cs-card { background:rgba(18,25,34,.93) !important; border:1px solid var(--cs-line); border-radius:18px !important; padding:22px !important; box-shadow:0 18px 50px rgba(0,0,0,.22) !important; }
.eyebrow { color:var(--cs-gold); border:1px solid rgba(231,182,91,.42); border-radius:999px; padding:3px 8px; font-size:.68rem; letter-spacing:.12em; }
.section-title { font-size:1.15rem; font-weight:650; }
.helper { color:#94a3b8; font-size:.78rem; line-height:1.45; }
.source-toggle { background:#0c1219; border:1px solid #263343; border-radius:12px; padding:4px; }
.cs-textarea textarea { min-height:190px !important; line-height:1.65 !important; }
.textual-six-section { padding:10px 12px 12px; border:1px solid #263343; border-radius:12px; background:#0c1219; }
.textual-six-label { color:#e7b65b; font-size:.82rem; font-weight:700; letter-spacing:.02em; }
.textual-six-input textarea { min-height:58px !important; line-height:1.55 !important; }
.mono-input textarea { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.82rem; }
.compact-upload { width:max-content; min-height:36px; }
.run-button { background:linear-gradient(100deg,#c98a32,#efc46e) !important; color:#18130b !important; font-weight:750; min-height:50px; border-radius:12px !important; }
.stage-row { margin-top:12px; }
.stage-pill { border-radius:999px; padding:6px 10px; font-size:.72rem; border:1px solid; text-align:center; flex:1; }
.stage-idle { color:#718096; border-color:#334155; background:#10161e; }
.stage-active { color:#f5cf82; border-color:#b7792f; background:#33230f; }
.stage-done { color:#91e2b2; border-color:#2e8b57; background:#10271c; }
.stage-error { color:#ff9f9f; border-color:#a83e3e; background:#311616; }
.metric-card { background:#0c1219; border:1px solid #263343; border-radius:12px; padding:12px; }
.metric-value { font-size:1rem; font-weight:650; }
.event-log { height:290px; background:#090d12 !important; border:1px solid #263343; border-radius:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.73rem; }
.result-tabs { background:transparent !important; }
.result-video { border-radius:12px; overflow:hidden; background:#000; }
.settings-dialog { width:min(920px,94vw) !important; max-width:920px !important; max-height:92vh; background:#111923 !important; border:1px solid #304054; border-radius:18px !important; }
.settings-scroll { height:66vh; padding-right:8px; }
.settings-group { border-bottom:1px solid #263343; }
@media (max-width:980px) { .cs-main-grid { grid-template-columns:1fr; } .cs-monitor { width:100%; } }
@media (max-width:640px) { .settings-grid,.metric-grid { grid-template-columns:1fr !important; } .hero-panel { padding:22px; } .cs-card { padding:16px !important; } }
"""
