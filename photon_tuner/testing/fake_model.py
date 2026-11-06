"""LangGraph DSE 管道的离线假聊天模型。

阶段 1（角色 agent）：假模型输出纯文本回答（携带确定性 mock 内容的 JSON blob）
而非工具调用，使 agent 恰好在一轮模型调用内结束。阶段 2（结构化抽取）：
重写 ``with_structured_output`` 将该 blob 解析回所请求的 pydantic 模型 ——
无网络，完全确定性。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

# task -> 符合 schema 的 mock 字段（编排器的结构化抽取会将这些字段
# 直接映射到对应的 pydantic 输出模型）。
_MOCK: Dict[str, Dict[str, Any]] = {
    "model_level_analysis": {
        "candidate_layers": [
            {"layer_id": 0, "layer_name": "conv1", "group": "convolution",
             "rank_metric": "energy_pj", "rank_value": 1.0,
             "impact_types": ["energy"], "root_causes": ["device dynamic energy"]},
            {"layer_id": 1, "layer_name": "linear", "group": "linear",
             "rank_metric": "latency_s", "rank_value": 1.0,
             "impact_types": ["latency"], "root_causes": ["memory loading"]},
        ],
        "summary": "Mock model-level analysis: energy is the dominant cost driver.",
        "global_findings": ["photonic crossbar energy dominates the objective"],
        "selected_views": ["energy", "area"],
        "analysis_base": {},
        "llm_notes": {"mock": True},
    },
    "architecture_analysis": {
        "primary_impact": "energy",
        "dominant_root_cause": "device",
        "layer_diagnoses": [
            {"layer_id": 0, "layer_name": "conv1", "impact_types": ["energy"],
             "root_causes": ["MZM dynamic energy"],
             "ratios": {"compute": 0.5, "memory": 0.1, "communication": 0.05,
                        "buffer": 0.05, "device": 0.3}},
        ],
        "retrieval_description": "conv1 energy bottleneck driven by MZM and laser power",
        "root_cause_summary": {"compute": 0.5, "memory": 0.1, "communication": 0.05,
                               "buffer": 0.05, "device": 0.3},
        "recommended_focus": ["mzm efficiency", "laser wall-plug efficiency"],
        "llm_notes": {"mock": True},
    },
    "topology_analysis": {
        "loss_budget_diagnosis": "core path loss is dominated by the on-chip laser and MRR rerouters.",
        "dominant_loss_contributors": [
            {"device_type": "on_chip_laser", "device_name": "Customized_Laser", "insertion_loss_dB": 2.0},
            {"device_type": "mrr_rerouter", "device_name": "LT_MRR_REROUTER", "insertion_loss_dB": 1.86},
            {"device_type": "mzm", "device_name": "LT_MZM", "insertion_loss_dB": 1.2},
        ],
        "power_budget": {
            "laser_output_dbm": 23.5,
            "path_loss_db": 6.22,
            "pd_sensitivity_dbm": -25.0,
            "link_margin_db": 42.28,
        },
        "power_budget_margin": "6.22 dB core path loss leaves limited optical power margin.",
        "recommended_template": None,
        "circuit_recommendations": [],
        "retrieval_description": "reduce dominant optical path insertion loss",
        "summary": "Mock topology analysis: laser and MRR rerouters dominate path loss.",
    },
    "device_analysis": {
        "recommended_link_edits": [
            {"level": "core", "device_type": "on_chip_laser",
             "device_name": "Customized_Laser", "reason": "dominant loss contributor"},
        ],
        "retrieval_description": "select lower-loss on-chip laser variant",
        "summary": "Mock device analysis: switch to a lower-loss laser variant.",
    },
    "solution_generation": {
        "strategy": "shrink the tile count to cut energy and area",
        "actions": [
            {"name": "step_hardware_parameter",
             "arguments": {"parameter": "cores_per_tile", "direction": -1}},
        ],
        "link_edits": [
            {"name": "modify_device_parameter",
             "arguments": {"level": "core", "device_type": "on_chip_laser",
                           "device_name": "Customized_Laser"}},
        ],
        "rationale": "Mock solution: reduce cores_per_tile 2->1 lowers energy and area.",
        "coupling_rationale": "Mock: no work_freq/precision change, so the rate-chain and precision constraints stay satisfied.",
        "notes": {"expected_objective_direction": "decrease"},
    },
}


def _detect_task(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") != "human":
            continue
        try:
            payload = json.loads(str(m.content))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("task"):
            return str(payload["task"])
    return "unknown"


class FakeToolCallingModel(BaseChatModel):
    """Deterministic model: task-driven prose in phase 1, direct mock in phase 2."""

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": "fake"}

    def bind_tools(self, tools, **kwargs: Any):
        # 假模型不调用真实工具；忽略绑定的工具接口。
        return self

    def with_structured_output(self, schema, **kwargs: Any):
        def _extract(messages, **kw) -> Any:
            for m in reversed(messages):
                try:
                    payload = json.loads(str(m.content))
                except (TypeError, ValueError):
                    continue
                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload.pop("task", None)
                    return schema(**payload)
            return schema()

        return RunnableLambda(_extract)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        task = _detect_task(messages)
        fields = _MOCK.get(task, {"summary": f"unknown task {task}", "candidate_layers": []})
        text = json.dumps({"task": task, **fields}, ensure_ascii=False)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
