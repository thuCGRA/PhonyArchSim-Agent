"""OPEarch-Agent 的默认原子技能。

每个技能仅形成一个专业结论；它们不调用其他技能。跨技能的分支、循环与返工由
``photon_tuner.workflows`` 和运行时编排器负责。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from photon_tuner.skills.core import (
    AtomicSkillSpec,
    InputRef,
    SkillCategory,
    SkillRegistry,
    SkillStep,
)


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = _number(mapping.get(key))
        if value is not None:
            return value
    return None


def _memory_dominance(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    scope = (evidence["summarize_layer_ranks"] or {}).get("energy_scope") or {}
    share = _number(scope.get("memory_share"))
    threshold = float(inputs.get("threshold", 0.5))
    return {
        "memory_share": share,
        "threshold": threshold,
        "is_memory_dominated": share is not None and share >= threshold,
        "basis": "memory energy / total energy",
    }


def _latency_hotspot(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    layers = list((evidence["summarize_layer_ranks"] or {}).get("layers") or [])
    if not layers:
        return {"hotspot": None, "reason": "no layer evidence"}

    def score(item: Mapping[str, Any]) -> float:
        return float(item.get("switching_cycles") or 0.0) + float(item.get("memory_latency_ms") or 0.0)

    best = max(layers, key=score)
    return {
        "hotspot": best.get("layer_name"),
        "group": best.get("group"),
        "switching_cycles": best.get("switching_cycles"),
        "memory_latency_ms": best.get("memory_latency_ms"),
        "rule": "maximum switching_cycles + memory_latency_ms",
    }


def _dominant_device(metric: str):
    tool_name = f"summarize_device_{metric}_breakdown"
    share_key = f"total_{metric}_share"

    def evaluate(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        devices = list((evidence[tool_name] or {}).get("devices") or [])
        if not devices:
            return {"device": None, "reason": f"no per-device {metric} evidence"}
        best = max(devices, key=lambda item: float(item.get(share_key) or 0.0))
        return {
            "device": best.get("device_name"),
            "device_type": best.get("device_type"),
            "level": best.get("level"),
            "share": best.get(share_key),
            "metric": metric,
        }

    return evaluate


def _power_budget(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["summarize_topology_evidence"] or {}
    power_inputs = payload.get("power_budget_inputs") or {}
    laser_cfg = ((power_inputs.get("laser") or {}).get("cfgs") or {})
    detector_cfg = ((power_inputs.get("photodetector") or {}).get("cfgs") or {})
    laser = _first_number(laser_cfg, ("power", "output_power", "laser_power", "power_dbm"))
    sensitivity = _first_number(detector_cfg, ("sensitivity", "sensitivity_dbm", "min_rx_power"))
    il = payload.get("insertion_loss") or {}
    core_loss = _number(il.get("core_insertion_loss")) or 0.0
    node_loss = _number(il.get("node_insertion_loss")) or 0.0
    required = float(inputs.get("required_margin_db", 0.0))
    margin = None if laser is None or sensitivity is None else laser - core_loss - node_loss - sensitivity
    return {
        "laser_output_dbm": laser,
        "path_loss_db": core_loss + node_loss,
        "detector_sensitivity_dbm": sensitivity,
        "link_margin_db": margin,
        "required_margin_db": required,
        "feasible": margin is not None and margin >= required,
    }


def _dominant_loss(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["summarize_topology_evidence"] or {}
    rows = list(payload.get("core_path_detail") or []) + list(payload.get("node_path_detail") or [])
    if not rows:
        return {"component": None, "reason": "no per-component loss evidence"}
    best = max(rows, key=lambda item: float(item.get("insertion_loss_dB") or 0.0))
    return {
        "component": best.get("instance"),
        "device_type": best.get("device_type"),
        "device_name": best.get("device_name"),
        "insertion_loss_db": best.get("insertion_loss_dB"),
    }


def _best_baseline(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    states = (evidence["compare_search_states"] or {}).get("states") or {}
    candidates = []
    for name, body in states.items():
        objective = _number((body.get("metrics") or {}).get("objective"))
        if objective is not None:
            candidates.append((objective, name))
    if not candidates:
        return {"source": None, "reason": "no comparable objective values"}
    objective, source = min(candidates)
    return {"source": source, "objective": objective, "rule": "minimum evaluated objective"}


def _objective_improvement(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    states = (evidence["compare_search_states"] or {}).get("states") or {}
    current = _number(((states.get("current") or {}).get("metrics") or {}).get("objective"))
    reference_name = str(inputs.get("reference", "best"))
    reference = _number(((states.get(reference_name) or {}).get("metrics") or {}).get("objective"))
    delta = None if current is None or reference is None or reference <= 0 else (reference - current) / reference
    return {
        "current_objective": current,
        "reference": reference_name,
        "reference_objective": reference,
        "relative_improvement": delta,
        "improved": delta is not None and delta > 0,
    }


def _device_choice(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    device_type = str(inputs["device_type"])
    priority = str(inputs.get("priority", "balanced"))
    body = (evidence["summarize_device_frontier"] or {}).get(device_type) or {}
    frontier = list(body.get("frontier_power_area") or [])
    tradeoffs = body.get("tradeoffs") or {}
    if not frontier:
        return {"device_type": device_type, "selected": None, "reason": "no comparable compatible frontier"}

    def value(name: str, key: str) -> float:
        number = _number((tradeoffs.get(name) or {}).get(key))
        return number if number is not None else float("inf")

    if priority == "energy":
        selected = min(frontier, key=lambda name: value(name, "energy_cost"))
    elif priority == "area":
        selected = min(frontier, key=lambda name: value(name, "area_mm2"))
    else:
        energy_order = {name: rank for rank, name in enumerate(sorted(frontier, key=lambda n: value(n, "energy_cost")))}
        area_order = {name: rank for rank, name in enumerate(sorted(frontier, key=lambda n: value(n, "area_mm2")))}
        selected = min(frontier, key=lambda name: energy_order[name] + area_order[name])
    return {
        "device_type": device_type,
        "priority": priority,
        "selected": selected,
        "tradeoff": tradeoffs.get(selected, {}),
        "frontier": frontier,
    }


def _parameter_effectiveness(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    parameter = str(inputs["parameter"])
    payload = evidence["searchable_parameters"] or {}
    effective = parameter in (payload.get("searchable") or [])
    disabled = (payload.get("disabled") or {}).get(parameter)
    return {
        "parameter": parameter,
        "effective": effective,
        "status": "effective" if effective else (disabled or {}).get("status", "unknown"),
        "reason": "" if effective else (disabled or {}).get("reason", "parameter is not in the active search space"),
        "action": "" if effective else (disabled or {}).get("action", "query summarize_search_space"),
    }


def _candidate_legality(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["validate_coupling"] or {}
    violations = list(payload.get("violations") or [])
    return {"legal": bool(payload.get("consistent", not violations)), "violations": violations}


def _novelty(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["summarize_design_identity"] or {}
    return {
        "fingerprint": payload.get("fingerprint"),
        "novel": not bool(payload.get("seen")),
        "known_fingerprint_count": payload.get("known_fingerprint_count", 0),
    }


def _case_reuse(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    cases = list((evidence["retrieve_similar_cases"] or {}).get("cases") or [])
    improved = [case for case in cases if (case.get("outcome") or {}).get("label") == "improved"]
    selected = max(improved, key=lambda item: float(item.get("score") or 0.0)) if improved else None
    return {
        "reusable": selected is not None,
        "selected_case": selected,
        "retrieved_count": len(cases),
        "improved_count": len(improved),
    }


def _memory_maturity(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["summarize_case_memory"] or {}
    improved = int(payload.get("improved_case_count") or 0)
    target = max(1, int(inputs.get("target", 2)))
    rho = min(improved / target, 1.0)
    return {"improved_case_count": improved, "target": target, "maturity": rho, "mature": rho >= 1.0}


def _convergence(inputs: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = evidence["summarize_schedule_signals"] or {}
    return {
        "converged": bool(payload.get("converged")),
        "recent_improvements": payload.get("best_so_far_relative_improvements", []),
        "window": payload.get("convergence_window"),
        "threshold": payload.get("convergence_delta"),
    }


def _skill(
    *, name: str, goal: str, description: str, category: SkillCategory,
    evidence: tuple[str, ...], steps: tuple[SkillStep, ...], rule: str,
    evaluator, roles: tuple[str, ...], input_schema: Optional[Mapping[str, Any]] = None,
    output_schema: Optional[Mapping[str, Any]] = None,
) -> AtomicSkillSpec:
    return AtomicSkillSpec(
        name=name,
        goal=goal,
        description=description,
        category=category,
        preconditions=("required tools are bound", "current context is available"),
        required_evidence=evidence,
        input_schema=input_schema or {"type": "object", "properties": {}},
        output_schema=output_schema or {"type": "object"},
        steps=steps,
        decision_rule=rule,
        failure_policy="Return a typed skill error; do not invent missing evidence.",
        evaluator=evaluator,
        allowed_roles=roles,
    )


DEFAULT_SKILLS = (
    _skill(name="assess_memory_energy_dominance", goal="判断当前设计是否由存储能耗主导", description="Use the reported compute/memory split to classify memory-energy dominance.", category=SkillCategory.DIAGNOSIS, evidence=("compute and memory energy",), steps=(SkillStep("summarize_layer_ranks", {"top_layers": 1}),), rule="memory_share >= threshold", evaluator=_memory_dominance, roles=("bottleneck", "architecture", "evaluate"), input_schema={"type": "object", "properties": {"threshold": {"type": "number"}}}),
    _skill(name="identify_latency_hotspot", goal="识别当前工作负载的主要时延热点层", description="Identify one latency hotspot from layer switching and memory evidence.", category=SkillCategory.DIAGNOSIS, evidence=("ranked layer costs",), steps=(SkillStep("summarize_layer_ranks", {"top_layers": 16}),), rule="maximize switching_cycles + memory_latency_ms", evaluator=_latency_hotspot, roles=("bottleneck", "architecture", "evaluate")),
    _skill(name="identify_dominant_energy_device", goal="识别能耗占比最高的器件", description="Select the device with the largest reported energy share.", category=SkillCategory.DIAGNOSIS, evidence=("per-device energy share",), steps=(SkillStep("summarize_device_energy_breakdown"),), rule="argmax total_energy_share", evaluator=_dominant_device("energy"), roles=("bottleneck", "architecture", "device", "evaluate")),
    _skill(name="identify_dominant_area_device", goal="识别面积占比最高的器件", description="Select the device with the largest reported area share.", category=SkillCategory.DIAGNOSIS, evidence=("per-device area share",), steps=(SkillStep("summarize_device_area_breakdown"),), rule="argmax total_area_share", evaluator=_dominant_device("area"), roles=("bottleneck", "architecture", "device", "evaluate")),
    _skill(name="assess_optical_power_budget", goal="判断当前光链路是否满足接收功率预算", description="Calculate laser-to-detector link margin from reported power and path loss.", category=SkillCategory.DIAGNOSIS, evidence=("laser output", "path loss", "detector sensitivity"), steps=(SkillStep("summarize_topology_evidence"),), rule="laser_output - path_loss - detector_sensitivity >= required_margin", evaluator=_power_budget, roles=("topology", "evaluate"), input_schema={"type": "object", "properties": {"required_margin_db": {"type": "number"}}}),
    _skill(name="identify_dominant_optical_loss", goal="识别光路中插损最大的组件", description="Select the largest measured component contribution on the critical optical path.", category=SkillCategory.DIAGNOSIS, evidence=("per-component path loss",), steps=(SkillStep("summarize_topology_evidence"),), rule="argmax insertion_loss_dB", evaluator=_dominant_loss, roles=("topology", "device", "evaluate")),
    _skill(name="choose_analysis_baseline", goal="选择后续诊断使用的已评估基点", description="Choose the evaluated current/previous/best state with the lowest objective.", category=SkillCategory.DECISION, evidence=("evaluated search states",), steps=(SkillStep("compare_search_states"),), rule="argmin objective", evaluator=_best_baseline, roles=("bottleneck", "architecture", "evaluate")),
    _skill(name="assess_objective_improvement", goal="判断当前设计相对指定基点是否改善", description="Compute the relative objective improvement against an evaluated reference state.", category=SkillCategory.DECISION, evidence=("current and reference objectives",), steps=(SkillStep("compare_search_states"),), rule="(reference-current)/reference > 0", evaluator=_objective_improvement, roles=("bottleneck", "architecture", "candidate", "evaluate"), input_schema={"type": "object", "properties": {"reference": {"type": "string"}}}),
    _skill(name="select_device_frontier_candidate", goal="按指定优化方向选择一个兼容器件变体", description="Choose one real compatible variant from the device Pareto frontier.", category=SkillCategory.DECISION, evidence=("compatible device frontier",), steps=(SkillStep("summarize_device_frontier", {"device_type": InputRef("device_type")}),), rule="minimum requested cost or minimum joint rank on the Pareto frontier", evaluator=_device_choice, roles=("device", "candidate", "evaluate"), input_schema={"type": "object", "properties": {"device_type": {"type": "string"}, "priority": {"type": "string"}}, "required": ["device_type"]}),
    _skill(name="assess_parameter_effectiveness", goal="判断一个参数在当前拓扑和后端下是否真正生效", description="Determine whether changing one parameter can affect the active simulator.", category=SkillCategory.VERIFICATION, evidence=("parameter scope and backend status",), steps=(SkillStep("searchable_parameters"),), rule="parameter is in searchable; otherwise return the declared disable reason", evaluator=_parameter_effectiveness, roles=("architecture", "topology", "candidate", "evaluate"), input_schema={"type": "object", "properties": {"parameter": {"type": "string"}}, "required": ["parameter"]}),
    _skill(name="assess_candidate_legality", goal="判断当前候选设计是否满足跨层耦合约束", description="Classify the current candidate as legal only when no deterministic violation exists.", category=SkillCategory.VERIFICATION, evidence=("cross-layer violations",), steps=(SkillStep("validate_coupling"),), rule="legal iff violations is empty", evaluator=_candidate_legality, roles=("topology", "candidate", "evaluate")),
    _skill(name="assess_design_novelty", goal="判断当前设计点是否已经被评估", description="Use the canonical hardware fingerprint to prevent duplicate evaluations.", category=SkillCategory.VERIFICATION, evidence=("hardware fingerprint and evaluated set",), steps=(SkillStep("summarize_design_identity"),), rule="novel iff fingerprint is not in the evaluated set", evaluator=_novelty, roles=("candidate", "evaluate")),
    _skill(name="assess_case_reusability", goal="判断检索到的历史案例是否包含可复用的已验证改善", description="Accept a precedent only when retrieval finds an objectively improved case.", category=SkillCategory.LEARNING, evidence=("retrieval score and verified case outcome",), steps=(SkillStep("retrieve_similar_cases", {"query": InputRef("query"), "top_k": InputRef("top_k", 5), "min_score": InputRef("min_score", 0.0)}),), rule="reusable iff at least one retrieved case has outcome=improved", evaluator=_case_reuse, roles=("candidate", "evaluate"), input_schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}, "min_score": {"type": "number"}}, "required": ["query"]}),
    _skill(name="assess_memory_maturity", goal="判断案例记忆是否积累了足够的有效先验", description="Compute memory maturity from simulator-verified improved precedents.", category=SkillCategory.LEARNING, evidence=("improved case count",), steps=(SkillStep("summarize_case_memory"),), rule="mature iff improved_case_count >= target", evaluator=_memory_maturity, roles=("candidate", "evaluate"), input_schema={"type": "object", "properties": {"target": {"type": "integer"}}}),
    _skill(name="assess_objective_convergence", goal="判断搜索目标是否在指定窗口内收敛", description="Use best-so-far relative improvements and the configured threshold to classify convergence.", category=SkillCategory.SCHEDULING, evidence=("objective history and scheduler thresholds",), steps=(SkillStep("summarize_schedule_signals"),), rule="all gains in the full window are <= convergence_delta", evaluator=_convergence, roles=("bottleneck", "candidate", "evaluate")),
)


def build_default_skill_registry() -> SkillRegistry:
    return SkillRegistry(DEFAULT_SKILLS)
