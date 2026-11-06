"""光子 DSE 工具箱的 LangChain 工具绑定层。

``photon_tuner.toolbox`` 中的领域工具处理器（``analysis_tools`` /
``device_tools`` / ``common_tool``）与 LangChain 无关；本子包是绑定层，将它们
重新暴露为 LangChain ``BaseTool``，使 LangGraph 运行时能够将它们绑定到聊天
模型。每个处理器仍通过可变的 :class:`RunState` 接收共享的
:class:`~photon_tuner.toolbox.common_tool.context.ToolContext`，使三个编辑工具
原地修改 ``current_hardware``。

``toolbox/__init__.py`` 不导入本子包，因此 ``import photon_tuner.toolbox`` 保持
无 LangChain 依赖，仅用于域测试套件；只有需要运行时工具的代码才导入
``toolbox.bindings``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, create_model

from photon_tuner.common.tool_types import ToolSpec
from photon_tuner.toolbox.analysis_tools import AnalysisToolbox, _REGISTRY as ANALYSIS_REGISTRY
from photon_tuner.toolbox.common_tool.context import ToolContext
from photon_tuner.toolbox.device_tools import DeviceToolbox, _REGISTRY as DEVICE_REGISTRY


@dataclass
class RunState:
    """每次迭代可变、被所有工具共享的运行状态。

    单个 :class:`ToolContext` 字段会随流水线推进而被原地更新（evaluation /
    model_analysis / bottleneck_state / optical_analysis / current_hardware），
    编辑工具会重新赋值 ``current_hardware``。
    """

    context: ToolContext


# --------------------------------------------------------------------------- #
# JSON Schema -> pydantic（ToolSpec.input_schema 方言很小：对象属性为
# {string|integer|number|boolean|array|object}）。
# --------------------------------------------------------------------------- #

_JSON_TYPE_TO_PYDANTIC = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": List[Any],
    "object": Dict[str, Any],
}


def json_schema_to_pydantic(name: str, schema: Dict[str, Any]) -> type[BaseModel]:
    """从简单的 JSON Schema 对象构建 pydantic 模型。

    每个属性都是 Optional，默认值为 ``None``：本项目的工具刻意宽松——它们在
    处理器*内部*校验必需参数并返回错误字典（从不抛异常），以便代理能够自我
    纠正。若在此将字段设为必需，会使 ``StructuredTool`` 在处理器错误返回路径
    运行之前就抛出 pydantic ``ValidationError``。
    """
    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in (schema.get("properties") or {}).items():
        json_type = prop_schema.get("type", "string")
        ptype = _JSON_TYPE_TO_PYDANTIC.get(json_type, Any)
        fields[prop_name] = (Optional[ptype], None)
    return create_model(f"{name.title().replace('_', '')}Args", **fields)


def _wrap(
    toolbox: Any,
    run_state: RunState,
    spec: ToolSpec,
    handler: Callable[[Any, ToolContext, Dict[str, Any]], Any],
) -> BaseTool:
    args_model = json_schema_to_pydantic(spec.name, spec.input_schema)

    def _call(**kwargs: Any) -> Any:
        # StructuredTool 会为模型未提供的参数填入 pydantic 默认值 (None)；
        # 领域处理器期望通过 ``arguments.get(key, default)`` 让缺失的键回退到
        # 自身默认值，因此在委托前丢弃值为 None 的条目。
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = handler(toolbox, run_state.context, args)
        except Exception as exc:  # noqa: BLE001
            # 错误的参数（如空器件名）绝不能中止运行：将错误返回给代理，使其
            # 自我纠正并继续迭代。领域处理器对无效输入抛出 ValueError/KeyError，
            # 这些是可恢复的，因此将其作为工具输出返回。
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "tool": spec.name,
                "arguments": args,
                "hint": "The call was rejected; fix the arguments and retry, or use a "
                        "read tool to discover the legal values first.",
            }
        return getattr(result, "payload", result)

    _call.__name__ = spec.name
    return StructuredTool.from_function(
        func=_call,
        name=spec.name,
        description=spec.description,
        args_schema=args_model,
    )


def wrap_registry(
    toolbox: Any,
    run_state: RunState,
    registry: Dict[str, Any],
    allowed_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    """将 ``allowed_names``（或全部）注册表条目封装为 LangChain 工具。"""
    selected = allowed_names if allowed_names is not None else list(registry)
    return [
        _wrap(toolbox, run_state, registry[name][0], registry[name][1])
        for name in selected
        if name in registry
    ]


def build_analysis_tools(
    search_space: Dict[str, Any],
    run_state: RunState,
    allowed_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    toolbox = AnalysisToolbox(search_space=search_space)
    return wrap_registry(toolbox, run_state, ANALYSIS_REGISTRY, allowed_names)


def build_device_tools(
    devicelib_root: str,
    search_space: Dict[str, Any],
    run_state: RunState,
    allowed_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    toolbox = DeviceToolbox(devicelib_root, search_space=search_space)
    return wrap_registry(toolbox, run_state, DEVICE_REGISTRY, allowed_names)
