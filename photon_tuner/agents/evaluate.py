"""各阶段证据评估闸门（8 行验收协议）。

* 确定性行先以纯代码运行，不唤醒评判智能体：1 模式、4 证据来源、5 编辑一致性
  （仅 solution）、5.5 结构合法性（topology + solution）。
* 语义行（2 完整性、3 证据完整性、6 推理、7 适用性）通过只读的
  ``evaluate_agent``（一个能实际检查工具结果、器件库与 SimPhony 报告的 LangGraph
  智能体）运行，且仅在全部确定性行通过后执行。
* 第 8 行（accept / request-info / re-execute / escalate-planning）是基于 1-7 行的
  确定性策略。

编排器驱动两阶段流程（``_run_stage_gate``）：先运行 ``deterministic_checks`` +
``structure_legality_check``，再调用 evaluate 智能体处理语义行，最后调用 ``decide``。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from photon_tuner.agents.outputs import EvaluateOutput

# 由各阶段闸门检查输出的角色。``solution`` 额外携带编辑一致性（第 5 行）与
# 跨层耦合行。
GATED_ROLES = ("model", "architecture", "topology", "device", "solution")

# 适用确定性结构/耦合合法性行（5.5）的角色。
STRUCTURE_LEGALITY_ROLES = ("topology", "solution")

# L4 电路级调节项，其推荐值必须是枚举候选值。``mesh_rank`` 是 mesh 的酉秩 N
# （整数），但同样要求其值属于设计空间候选。
_L4_PARAMS = ("sharing_policy", "temporal_accum", "laser_distribution", "mesh_rank")

# 指示智能体对其实可获取的证据放弃的措辞。
_PLACEHOLDER_PHRASES = (
    "not exposed", "not available", "cannot be stated", "can not be stated",
    "unavailable", "not retrievable", "not provided by any tool",
)

# topology 阶段必须以真实 dB 数值填充的键。
_REQUIRED_POWER_BUDGET_KEYS = (
    "laser_output_dbm", "path_loss_db", "pd_sensitivity_dbm", "link_margin_db",
)

# architecture 阶段可在 ``circuit_recommendations`` 中推荐的参数。这些是它实际
# 拥有的系统/光核/存储级调节项；器件变体（phase_shifter、coupler、laser 等）归
# device 阶段，topology L4 调节项（sharing_policy、temporal_accum、
# laser_distribution）归 topology 阶段。该集合在运行时按层从设计空间派生；这里
# 是回退用的静态列表。
#
# 注意：这只是"架构阶段有权动"的全集。它在**当前拓扑族下**是否真正生效，由
# ``photon_tuner.core.topology_space.PARAM_SCOPE`` 决定——参见
# :func:`_ineffective_param_fail`。例如网格拓扑下 ``core_height`` 被 ``mesh_rank``
# 取代、``dataflow`` 被物理固定，推荐它们不会有任何效果。
_ARCH_LEGAL_PARAMS = {
    "tiles", "cores_per_tile", "dataflow",
    "precision_in_bit", "precision_w_bit", "precision_out_bit",
    "hbm_bandwidth", "glb1_cache_size", "glb2_cache_size", "rf_cache_size",
    "core_height", "core_width", "num_wavelength", "work_freq",
}


def _context_core_kind(
    context: Optional[Dict[str, Any]],
    output: Any = None,
) -> Optional[str]:
    """推断该阶段推荐所针对的 ``core_kind``。

    优先级：输出里的 ``recommended_template``（拓扑阶段提出的**目标**结构，即设计点
    换过去之后的状态）→ ``context["hardware"]`` 的当前 ``core_kind`` → ``None``。
    这样"推荐切到网格 + 同时又调 core_height"这类自相矛盾的组合会被判失败，而不是
    在切换后静默失效。
    """
    if output is not None:
        recommended = getattr(output, "recommended_template", None)
        if recommended:
            return str(recommended)
    hardware = (context or {}).get("hardware")
    if isinstance(hardware, dict):
        topo = ((hardware.get("arch") or {}).get("topology") or {})
        value = topo.get("core_kind")
        if value is not None:
            return str(value)
    return None


def _forced_value_fail(
    param: str,
    value: Any,
    core_kind: Optional[str],
    hardware: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """若 ``param=value`` 与该拓扑的物理强制值冲突，返回拒绝理由；否则 ``None``。

    与 :func:`_ineffective_param_fail` 互补：那里拦"该参数完全无效"，这里拦
    "参数有效但被推荐成了拓扑装不出来的值"（例如网格下 num_wavelength=12）。

    提供 ``hardware`` 时按该设计点解析派生强制（如权重阵列的
    ``num_wavelength == mesh_rank``）；否则只检查常量强制。
    """
    from photon_tuner.core.topology_space import (
        FAMILY_LABELS,
        core_family,
        effective_forced_params,
        forced_params,
    )

    if value is None:
        return None
    forced = (
        effective_forced_params(hardware)
        if isinstance(hardware, dict)
        else forced_params(core_kind)
    )
    if param not in forced:
        return None
    want = forced[param]
    if str(value) == str(want):
        return None
    family = core_family(core_kind if core_kind is not None else (hardware or {}).get(
        "arch", {}).get("topology", {}).get("core_kind"))
    label = FAMILY_LABELS.get(family, family)
    return (
        f"{param}={value!r} is physically impossible under {label} "
        f"(core_kind={core_kind!r}): the netlist builder forces {want!r}. "
        f"Omit this lever and let topology handle it."
    )


def _ineffective_param_fail(param: str, core_kind: Optional[str]) -> Optional[str]:
    """若 ``param`` 在当前拓扑下不生效，返回可操作的拒绝理由；否则 ``None``。"""
    from photon_tuner.core.topology_space import FAMILY_LABELS, param_status

    st = param_status(param, core_kind)
    if st.get("status") == "effective":
        return None
    label = FAMILY_LABELS.get(st.get("family"), st.get("family"))
    reason = st.get("reason", "")
    action = st.get("action", "")
    by = st.get("by")
    detail = f"{param!r} has no effect under {label} (core_kind={core_kind!r})"
    if by:
        detail += f": it is superseded by {by!r}"
    if reason:
        detail += f" — {reason}"
    if action:
        detail += f" Suggested: {action}."
    return detail

# 各角色的语义检查清单。每个清单原样注入 evaluate 智能体的载荷，使评审者明确
# 该阶段的契约。中间阶段保持轻量清单。
STAGE_CHECKLISTS: Dict[str, List[str]] = {
    "model": [
        "Candidate layers are ranked and each rank is supported by summarized metrics / energy / area / insertion-loss evidence.",
        "An analysis base was selected (select_analysis_base / compare_search_states were used).",
        "No claim asserts a fact absent from the tool results.",
    ],
    "architecture": [
        "primary_impact and dominant_root_cause follow from the layer/operator evidence.",
        "root_cause_summary fractions are coherent (roughly sum to ~1.0).",
        "The recommended focus areas map to the diagnosed root cause.",
        "The named root cause is the *measured* dominant contributor: its share in the "
        "ranked device energy/area evidence is the largest, and no focus area targets a "
        "component whose measured share is negligible relative to the objective.",
    ],
    "topology": [
        "The insertion-loss diagnosis matches the longest-path IL report (which component loses how many dB).",
        "The power-budget margin is consistent with laser output vs path loss vs PD sensitivity.",
        "Any recommended structural change (template / layout) is one of the legal enumerated templates — no invented netlists.",
    ],
    "device": [
        "Every recommended variant actually exists in the device library (list_device_candidates / inspect_device_config).",
        "Area / insertion-loss / power of a variant was inspected before recommending it.",
        "No parameter value is invented beyond the library cfgs.",
    ],
    "solution": [
        "Actions and link_edits use only the documented tools and arguments.",
        "The proposed move is consistent with the bottleneck state and optical analysis.",
        "The cross-layer coupling is legal: template/node/device mapping, work_freq <= min(chain), and precision compatibility are respected.",
    ],
}


def _row(row: Any, name: str, status: str, detail: str) -> Dict[str, Any]:
    return {"row": row, "name": name, "status": status, "detail": detail}


def deterministic_checks(
    role: str,
    output: Any,
    messages: Optional[List[Any]] = None,
    applied_edits: Optional[List[str]] = None,
    failed_edits: Optional[List[Dict[str, str]]] = None,
    declared_edits: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """运行第 1、4 行（所有角色）与第 5 行（仅 solution）。"""
    checks: List[Dict[str, Any]] = []

    # 第 1 行——输出格式/模式合规（编排器已将提案强转为 pydantic 模型；此处防御性复检）。
    schema_ok = output is not None and isinstance(output, BaseModel)
    checks.append(_row(
        1, "output_schema", "pass" if schema_ok else "fail",
        f"structured output type={type(output).__name__}" if schema_ok else "missing structured output",
    ))

    # 第 4 行——证据真实性/来源：会话记录中至少有一次带对应工具结果的真实工具调用。
    if messages:
        tool_calls = [
            getattr(m, "tool_calls", None)
            for m in messages
            if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
        ]
        tool_results = [m for m in messages if getattr(m, "type", "") == "tool"]
        n_calls = sum(len(tc) for tc in tool_calls if tc)
        evidence_ok = n_calls > 0 and len(tool_results) > 0
        checks.append(_row(
            4, "evidence_source", "pass" if evidence_ok else "fail",
            f"tool_calls={n_calls} tool_results={len(tool_results)}",
        ))
    else:
        checks.append(_row(4, "evidence_source", "fail", "no tool transcript: claims are unverifiable"))

    # 第 5 行——声明编辑 vs 实际应用的编辑（仅 solution，当编排器传入声明/应用的编辑集时）。
    if role == "solution" and declared_edits is not None:
        failed_names = {e.get("name") for e in (failed_edits or [])}
        declared = set(declared_edits)
        applied = set(applied_edits or [])
        consistent = declared.issubset(applied | failed_names)
        checks.append(_row(
            5, "edit_consistency", "pass" if consistent else "fail",
            f"declared={sorted(declared)} applied={sorted(applied)} failed={sorted(failed_names)}; "
            "every actions/link_edits entry must be {'name': <tool>, 'arguments': {...}} "
            "(tools: modify_hardware_parameter, step_hardware_parameter, modify_device_parameter)",
        ))

    return checks


def structure_legality_check(
    role: str,
    output: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """确定性跨层结构/耦合合法性检查（第 5.5 行）。

    包含两项检查，均为确定性且基于证据：

    1. 若上游输出带有 ``recommended_template``（topology 的结构提案），它必须是
       合法的模板名且可被仿真器评估——否则提案将产生仿真器无法运行的设计点。
    2. ``context["hardware"]`` 中的硬件点必须满足耦合矩阵（类型兼容 + 所需器件
       槽位）及数值跨层约束（经由 ``context["design_space"]``）。

    缺少 ``context`` 时降级为通过（调用方未传入硬件，例如裸单元测试），使闸门保持
    非致命。
    """
    from photon_tuner.core.topology_space import (
        CORE_KINDS,
        _is_evaluable,
        physical_violations,
    )

    context = context or {}
    mode = str(context.get("mode", "real"))
    fails: List[str] = []

    recommended = getattr(output, "recommended_template", None)
    if recommended:
        rec = str(recommended)
        if rec not in CORE_KINDS:
            fails.append(
                f"recommended_template {rec!r} is not a legal template; legal: {sorted(CORE_KINDS)}"
            )
        elif not _is_evaluable(CORE_KINDS[rec], mode):
            fails.append(
                f"recommended_template {rec!r} is not evaluable in {mode} mode "
                f"(this kind has no real ONNArchSimulator netlist path)"
            )

    hardware = context.get("hardware")
    if hardware is not None:
        for v in physical_violations(hardware, context.get("design_space"), mode=mode):
            fails.append(v.get("message", str(v)))

    # L4 电路推荐合法性：每个推荐的 sharing_policy / temporal_accum /
    # laser_distribution 值必须是枚举的搜索空间候选之一。这样可确定性捕获捏造值
    # （如 "full_shared"、"H"），而非仅依赖语义评判。
    #
    # 同时校验拓扑族有效性：``mesh_rank`` 在 crossbar 家族下没有意义，而
    # ``core_height``/``num_wavelength``/``dataflow`` 在 mesh 与权重阵列下不生效。
    # 推荐一个不生效的参数等于让下游编辑白做——这里直接拒绝并给出替代建议。
    design_space = context.get("design_space")
    recs = getattr(output, "circuit_recommendations", None) or []
    # 拓扑阶段若提出了目标模板，则以目标结构为准判断参数是否生效。
    target_core_kind = _context_core_kind(context, output)
    if design_space is not None and recs:
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            param = str(rec.get("parameter", ""))
            if param not in _L4_PARAMS and param not in _ARCH_LEGAL_PARAMS:
                continue
            ineffective = _ineffective_param_fail(param, target_core_kind)
            if ineffective:
                fails.append(f"circuit_recommendations {ineffective}")
                continue
            value = rec.get("value")
            forced_fail = _forced_value_fail(param, value, target_core_kind, hardware)
            if forced_fail:
                fails.append(f"circuit_recommendations {forced_fail}")
                continue
            try:
                legal = [str(c) for c in design_space.candidates(param)]
            except Exception:  # noqa: BLE001 — 未知参数降级为不检查
                continue
            if legal and str(value) not in legal:
                fails.append(
                    f"circuit_recommendations {param}={value!r} is not an enumerated "
                    f"candidate; legal: {legal}"
                )

    if fails:
        return _row(5.5, "structure_legality", "fail", "; ".join(fails))
    return _row(
        5.5, "structure_legality", "pass",
        "core_kind/node_kind compatible and coupling constraints satisfied"
        if (recommended or hardware is not None)
        else "structural-edit legality: no structural proposal or hardware context supplied",
    )


def architecture_legality_check(output: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """第 5.4 行——architecture 阶段的确定性边界检查。

    architecture 阶段负责归因成本并推荐 **架构级** 调节项；它不得提出器件变体
    （属于 device 阶段）或 topology L4 调节项（属于 topology 阶段）。因此
    ``circuit_recommendations`` 仅限系统/光核/存储参数。推荐值中出现器件变体名
    （如 IHO_PS、LT_Laser）同样失败，因为值必须是枚举的搜索空间候选。
    """
    recs = getattr(output, "circuit_recommendations", None) or []
    fails: List[str] = []

    context = context or {}
    design_space = context.get("design_space")
    core_kind = _context_core_kind(context, output)
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        param = str(rec.get("parameter", ""))
        value = rec.get("value")
        if not param:
            fails.append(f"circuit_recommendations entry {rec!r} has no parameter")
            continue
        if param not in _ARCH_LEGAL_PARAMS:
            fails.append(
                f"circuit_recommendations parameter {param!r} is outside the "
                f"architecture stage's authority (legal: tiles, cores_per_tile, "
                f"dataflow, precision_*, hbm_bandwidth, glb*/rf_cache_size, "
                f"core_height/width, num_wavelength, work_freq); device variants "
                f"and topology L4 levers belong to later stages"
            )
            continue
        # 拓扑族有效性：该参数在当前 core_kind 下必须真正影响仿真结果。
        ineffective = _ineffective_param_fail(param, core_kind)
        if ineffective:
            fails.append(f"circuit_recommendations {ineffective}")
            continue
        # 物理可行性：即便参数有效，也不得推荐该拓扑装不出来的取值。
        forced_fail = _forced_value_fail(
            param, value, core_kind, context.get("hardware")
        )
        if forced_fail:
            fails.append(f"circuit_recommendations {forced_fail}")
            continue
        # 该值必须是该参数的枚举候选值。
        legal: List[str] = []
        if design_space is not None:
            try:
                legal = [str(c) for c in design_space.candidates(param)]
            except Exception:  # noqa: BLE001 — 未知参数降级为不做值检查
                legal = []
        if legal and str(value) not in legal:
            fails.append(
                f"circuit_recommendations {param}={value!r} is not an enumerated "
                f"candidate; legal: {legal}"
            )

    if fails:
        return _row(5.4, "architecture_legality", "fail", "; ".join(fails))
    return _row(
        5.4, "architecture_legality", "pass",
        "recommended levers are architecture-level with enumerated values",
    )


def _as_edit_dict(edit: Any) -> Any:
    """将提议的编辑（pydantic 模型或字典）规范化为普通字典。"""
    if hasattr(edit, "model_dump"):
        return edit.model_dump()
    return edit


def _edit_name(edit: Any) -> str:
    """返回提议编辑的工具名，兼容模型的 ``action`` 别名。"""
    edit = _as_edit_dict(edit)
    if not isinstance(edit, dict):
        return ""
    return str(edit.get("name") or edit.get("action") or "")


def _switches_core_kind(edit: Any) -> bool:
    """判断某个编辑是否执行拓扑结构（core_kind）切换。"""
    edit = _as_edit_dict(edit)
    name = _edit_name(edit)
    if name == "modify_topology_template":
        return True
    if name != "modify_hardware_parameter":
        return False
    args = (edit or {}).get("arguments", {}) or {}
    for upd in (args.get("updates") or []):
        if isinstance(upd, dict) and str(upd.get("parameter", "")) == "core_kind":
            return True
    return False


def topology_binding_check(
    role: str,
    output: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """第 5.8 行——solution 必须落实 topology 的推荐。

    当 topology 阶段指定了一个*不同*、合法且可评估的 ``core_kind`` 时，solution
    必须提出拓扑结构编辑（``modify_topology_template`` 或对 ``core_kind`` 的
    ``modify_hardware_parameter`` 更新）。
    """
    if role != "solution":
        return _row(5.8, "topology_binding", "pass", f"not applicable to role {role!r}")

    ctx = context or {}
    optical = ctx.get("optical_analysis") or {}
    recommended = optical.get("recommended_template") or ""
    if not recommended:
        return _row(5.8, "topology_binding", "pass", "no structural recommendation to honour")

    recommended = str(recommended)
    current = str(ctx.get("current_core_kind") or "dptc_crossbar")
    if recommended == current:
        return _row(5.8, "topology_binding", "pass",
                    "recommended template is already the current one")

    # 该推荐本身必须合法且可评估；否则 topology 阶段自身的结构行（5.5）已失败，
    # 不再要求 solution 遵循非法目标。
    from photon_tuner.core.topology_space import CORE_KINDS, _is_evaluable

    meta = CORE_KINDS.get(recommended)
    if meta is None or not _is_evaluable(meta, str(ctx.get("mode", "real"))):
        return _row(5.8, "topology_binding", "pass",
                    f"recommended_template {recommended!r} is not a legal evaluable template")

    actions = getattr(output, "actions", None) or []
    if any(_switches_core_kind(a) for a in actions):
        return _row(5.8, "topology_binding", "pass",
                    "solution proposes the recommended structural switch")

    return _row(
        5.8, "topology_binding", "fail",
        f"topology stage recommended core_kind={recommended!r} (current={current!r}) "
        f"but the solution proposed no core_kind edit; add a modify_topology_template "
        f"or a modify_hardware_parameter on core_kind/node_kind",
    )


def evidence_completeness_check(
    role: str,
    output: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """第 5.6 行——确定性“是否确实收集了证据”检查。

    捕获智能体把可获取的证据报告为不可用的常见失败。对 topology 阶段要求：

    * ``dominant_loss_contributors`` 非空且非占位（按组件 IL 分解由
      summarize_insertion_loss_detail 提供）；
    * ``power_budget`` 携带四个数值键，且 ``link_margin_db`` 与激光 − 路径损耗 −
      灵敏度一致。

    此处仅对 topology 角色设闸；其他角色保留语义行。
    """
    if role != "topology":
        return _row(5.6, "evidence_completeness", "pass", f"not applicable to role {role!r}")

    fails: List[str] = []

    def _placeholder(text: Any) -> bool:
        low = str(text or "").lower()
        return any(p in low for p in _PLACEHOLDER_PHRASES)

    contributors = getattr(output, "dominant_loss_contributors", None) or []
    if not contributors:
        fails.append(
            "dominant_loss_contributors is empty; summarize_insertion_loss_detail "
            "exposes core_path_detail/node_path_detail with per-component dB"
        )
    else:
        for entry in contributors:
            if not isinstance(entry, dict):
                continue
            if _placeholder(entry.get("device_name")) or _placeholder(entry.get("device_type")):
                fails.append(
                    f"dominant_loss_contributors entry {entry!r} uses a placeholder instead "
                    f"of a real component"
                )
                break

    for field in ("loss_budget_diagnosis", "power_budget_margin"):
        if _placeholder(getattr(output, field, "")):
            fails.append(
                f"{field} reports evidence as unavailable ({getattr(output, field, '')!r}); "
                f"laser power and PD sensitivity are exposed by inspect_device_config"
            )

    budget = getattr(output, "power_budget", None) or {}
    missing = [k for k in _REQUIRED_POWER_BUDGET_KEYS if k not in budget]
    if missing:
        fails.append(
            f"power_budget is missing {missing}; supply the dB numbers from "
            f"inspect_device_config (laser power / PD sensitivity) and the IL report"
        )
    else:
        try:
            expected = (
                float(budget["laser_output_dbm"])
                - float(budget["path_loss_db"])
                - float(budget["pd_sensitivity_dbm"])
            )
            if abs(expected - float(budget["link_margin_db"])) > 0.01:
                fails.append(
                    f"power_budget.link_margin_db={budget['link_margin_db']} is inconsistent "
                    f"with laser_output_dbm - path_loss_db - pd_sensitivity_dbm = {expected:.4f}"
                )
        except (TypeError, ValueError):
            fails.append("power_budget values must all be numeric dB values")

    if fails:
        return _row(5.6, "evidence_completeness", "fail", "; ".join(fails))
    return _row(
        5.6, "evidence_completeness", "pass",
        "per-component loss attributed and power budget computed from device cfgs",
    )


def semantic_fails(verdict: Optional[EvaluateOutput]) -> List[Dict[str, Any]]:
    """将 EvaluateOutput 的四条语义行展平为失败检查字典。"""
    if verdict is None:
        return []
    fails: List[Dict[str, Any]] = []
    for row, field in (
        (2, "content_completeness"),
        (3, "evidence_completeness"),
        (6, "reasoning_correctness"),
        (7, "evidence_applicability"),
    ):
        text = str(getattr(verdict, field, ""))
        if text.lower().startswith("fail"):
            fails.append(_row(row, field, "fail", text))
    return fails


def decide(
    checks: List[Dict[str, Any]],
    verdict: Optional[EvaluateOutput] = None,
) -> Dict[str, Any]:
    """第 8 行——基于 1-7 行的验收决策（规则驱动）。

    - 第 1/5/5.5 行（模式、编辑一致性、结构合法性）为硬性：任一失败 → ``re-execute``。
    - 第 4 行（证据真实性）→ ``request-info``（收集证据）。
    - 语义行 2/3/6/7 → ``re-execute`` 并附评判反馈。
    """
    fails: List[Dict[str, Any]] = [c for c in checks if c.get("status") == "fail"]
    fails += semantic_fails(verdict)

    if not fails:
        return {"decision": "accept", "reasons": []}

    # 第 1/5/5.5/5.6/5.7 行为确定性且硬性：任一失败即 re-execute，并附具体原因
    # （兼作返工的 gate_feedback）。
    hard = [f for f in fails if f.get("row") in (1, 5, 5.4, 5.5, 5.6, 5.7, 5.8)]
    if hard:
        return {
            "decision": "re-execute",
            "reasons": [f"{f['name']}: {f['detail']}" for f in hard],
        }
    if any(f.get("row") == 4 for f in fails):
        return {
            "decision": "request-info",
            "reasons": [f"{f['name']}: {f['detail']}" for f in fails],
        }
    # 语义行携带可用于定向返工的可执行反馈。
    feedback = getattr(verdict, "feedback", []) if verdict is not None else []
    return {
        "decision": "re-execute",
        "reasons": [f"{f['name']}: {f['detail']}" for f in fails] + feedback,
    }
