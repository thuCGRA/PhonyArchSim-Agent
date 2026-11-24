"""把原子技能按需暴露为 LangChain StructuredTool。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from langchain_core.tools import BaseTool, StructuredTool

from photon_tuner.skills import SkillRegistry, SkillRuntime
from photon_tuner.toolbox.bindings import json_schema_to_pydantic


def _invoke_tool(all_tools: Mapping[str, Any], name: str, arguments: Mapping[str, Any]) -> Any:
    tool = all_tools[name]
    if hasattr(tool, "invoke"):
        return tool.invoke(dict(arguments))
    return tool(**dict(arguments))


def build_atomic_skill_tools(
    registry: SkillRegistry,
    all_tools: Mapping[str, Any],
    allowed_names: Optional[List[str]] = None,
) -> List[BaseTool]:
    """构建技能工具；调用时仍使用同一批底层领域工具与运行状态。"""
    selected = allowed_names or [skill.name for skill in registry.list()]
    built: List[BaseTool] = []
    for name in selected:
        skill = registry.get(name)
        runtime = SkillRuntime(lambda tool_name, args: _invoke_tool(all_tools, tool_name, args))
        args_model = json_schema_to_pydantic(skill.name, dict(skill.input_schema))

        def _call(_skill=skill, _runtime=runtime, **kwargs: Any) -> Dict[str, Any]:
            inputs = {key: value for key, value in kwargs.items() if value is not None}
            result = _runtime.execute(_skill, inputs)
            return {
                "skill": result.name,
                "conclusion": dict(result.conclusion),
                "evidence": dict(result.evidence),
                "trace": [
                    {
                        "step": item.step,
                        "tool_name": item.tool_name,
                        "arguments": dict(item.arguments),
                    }
                    for item in result.trace
                ],
            }

        _call.__name__ = skill.name
        built.append(
            StructuredTool.from_function(
                func=_call,
                name=skill.name,
                description=skill.description,
                args_schema=args_model,
            )
        )
    return built
