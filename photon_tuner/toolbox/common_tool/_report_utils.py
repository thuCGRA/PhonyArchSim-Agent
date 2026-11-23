"""分析工具与器件工具共用的报告读取辅助函数。

两个工具箱通过同一入口读取 SimPhony 报告，而非由器件工具从分析模块导入
私有辅助函数（``_load_report``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from photon_tuner.common.io import read_json
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.common.utils import layer_group
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.toolbox.common_tool.context import ToolContext


def _load_report(context: ToolContext) -> Dict[str, Any]:
    if context.evaluation is None:
        raise ValueError("No evaluation available in tool context")
    report_path = context.evaluation.detail_files.get("report")
    if report_path is None or not Path(report_path).exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    return read_json(report_path)


def _metrics_payload(context: ToolContext, evaluation: EvaluationResult) -> Dict[str, Any]:
    metrics = metrics_with_objective(
        evaluation.metrics, evaluation.hardware, context.search_space
    )
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _active_evaluation(context: ToolContext) -> EvaluationResult:
    """返回当前选定的评估基准（默认：current）。"""
    source = context.active_base.get("source", "current")
    if source in context.evaluation_bases:
        return context.evaluation_bases[source]
    return context.evaluation or next(iter(context.evaluation_bases.values()))


def _flatten_cycles(partition_cycles: Dict[str, Any]) -> Dict[str, Any]:
    """将 {sub_arch: {layer: cycle}} 展平为 {layer: cycle}。"""
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in partition_cycles.items():
        if isinstance(layers, dict):
            flattened.update(layers)
    return flattened


def _flatten_memory(latency: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in latency.items():
        if isinstance(layers, dict):
            flattened.update(layers)
    return flattened


def _flatten_memory_energy(energy: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, layers in energy.items():
        if isinstance(layers, dict):
            for layer, value in layers.items():
                flattened[layer] = value
    return flattened


def _rank_layers(report: Dict[str, Any], context: ToolContext) -> List[Dict[str, Any]]:
    """按成本对层排序，优先采用仿真器*原生*的逐层能耗。

    分析镜像会报告 ``energy.by_layer``（在其迭代层时采集）。存在时直接使用。
    较旧的报告和真实 ONN 仿真器——它们只按器件细分能耗——回退到按周期比例的
    估算，并通过 ``energy_source`` 标记。
    """
    cycles = _flatten_cycles(report.get("workload", {}).get("partition_cycles", {}))
    memory_latency = _flatten_memory(report.get("memory", {}).get("latency", {}))
    memory_energy = _flatten_memory_energy(report.get("memory", {}).get("energy", {}))
    native_energy = _flatten_memory_energy(report.get("energy", {}).get("by_layer", {}))

    metrics = report.get("metrics", {})
    total_energy = float(metrics.get("energy_mj", 1.0))
    total_area = float(metrics.get("area_mm2", 1.0))
    # 逐层延迟估算需要工作频率 (GHz)；从已评估的硬件中读取，以便反映步进
    # 频率的变化。
    freq = _work_freq_g_hz(report, context)

    switching_cycles = {name: _cycle_switching(cycle) for name, cycle in cycles.items()}
    total_switching = sum(switching_cycles.values()) or 1

    ranks = []
    for name, cycle in cycles.items():
        share = switching_cycles[name] / total_switching
        if name in native_energy:
            # 原生 ``energy.by_layer`` 是 SimPhony 的 pJ 明细；``metrics`` 是规范
            # 单位 mJ。两者混用会让排序把 pJ 的数当成 mJ（差 1e9 倍），瓶颈定位
            # 直接错乱——这里显式换算到 mJ。
            energy_mj = float(native_energy[name]) * 1e-9
            energy_source = "simulated"
        else:
            energy_mj = total_energy * share
            energy_source = "estimated"
        ranks.append({
            "layer_name": name,
            "group": layer_group(name),
            "cycles": cycle,
            "switching_cycles": switching_cycles[name],
            # cycles / freq[GHz] = ns；ns -> ms 乘 1e-6。
            "latency_ms": switching_cycles[name] / freq * 1e-6,
            # 报告 memory.latency 单位为 s -> ms。
            "memory_latency_ms": _memory_latency_value(memory_latency.get(name)) * 1e3,
            "energy_mj": energy_mj,
            "energy_source": energy_source,
            "area_share_mm2": total_area * share,
        })
    ranks.sort(key=lambda item: item["energy_mj"], reverse=True)
    for index, item in enumerate(ranks):
        item["layer_id"] = index
    return ranks


def top_layer_energy_share(report: Dict[str, Any], context: ToolContext) -> Optional[float]:
    """顶层算子在**逐层能耗**中的占比（0–1）；无法计算时返回 ``None``。

    这是"瓶颈有多集中"的语义化度量：0.35 表示最贵的层只占三成、其余七成分散在别的层里。
    相比之下用 ``1 / 层数`` 作代理没有量程（78 层时恒为 0.013），既无法设阈值也无法解释。

    ``None``（而不是 0.0）是刻意的：调用方必须区分"测得集中度为零"与"没有报告可算"。
    """
    try:
        ranks = _rank_layers(report, context)
    except Exception:  # noqa: BLE001 — 缺报告/字段时不应让调度崩掉
        return None
    if not ranks:
        return None
    energies = [float(row.get("energy_mj", 0.0) or 0.0) for row in ranks]
    total = sum(energies)
    if total <= 0:
        return None
    return max(energies) / total


def _memory_energy_by_level(report: Dict[str, Any]) -> Dict[str, float]:
    """把 ``memory.energy``（pJ，按存储层级组织）换算成 mJ。

    注意：它是 ``{sub_arch: {HBM|GLB2|GLB1|RF: pJ}}``，**按存储层级**而非按
    NN 层组织。早先的逐层 ``memory_energy_mj`` 用层名去查这张表，永远查不到，
    于是恒为 0——智能体会据此认为存储能耗为零，而实际它才是能耗主体。
    """
    levels: Dict[str, float] = {}
    for sub_arch in (report.get("memory", {}).get("energy", {}) or {}).values():
        if not isinstance(sub_arch, dict):
            continue
        for level, value in sub_arch.items():
            try:
                levels[str(level)] = levels.get(str(level), 0.0) + float(value) * 1e-9
            except (TypeError, ValueError):
                continue
    return dict(sorted(levels.items(), key=lambda kv: kv[1], reverse=True))


def _energy_split_mj(report: Dict[str, Any]) -> Dict[str, float]:
    """返回 ``{compute, memory, total}``（mJ），供工具在载荷里显式声明口径。"""
    metrics = report.get("metrics", {}) or {}
    try:
        compute = float(metrics.get("energy_compute_mj", metrics.get("energy_mj", 0.0)))
        total = float(metrics.get("energy_mj", 0.0))
    except (TypeError, ValueError):
        compute, total = 0.0, 0.0
    if "energy_memory_mj" in metrics:
        memory = float(metrics["energy_memory_mj"])
    else:
        # 旧报告（能耗口径修正之前）没有这个字段：从 memory.energy 重新求，
        # 而不是静默当成 0——把"存储能耗为零"当成事实正是早先的 bug。
        memory = sum(_memory_energy_by_level(report).values())
        total = compute + memory
    return {"compute": compute, "memory": memory, "total": total}


def _work_freq_g_hz(report: Dict[str, Any], context: ToolContext) -> float:
    """从硬件读取交叉开关时钟频率 (GHz)，并带兜底值。"""
    candidates = [
        (context.current_hardware or {}).get("arch", {}).get("optical_core", {}).get("work_freq"),
        (report.get("hardware") or {}).get("arch", {}).get("optical_core", {}).get("work_freq"),
    ]
    for value in candidates:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 5.0


def _cycle_switching(cycle: Any) -> float:
    """从 SimPhony 周期元组计算 iter_N * iter_D * iter_M。"""
    if isinstance(cycle, (list, tuple)) and len(cycle) >= 3:
        try:
            return float(cycle[0]) * float(cycle[1]) * float(cycle[2])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _memory_latency_value(value: Any) -> float:
    if isinstance(value, dict):
        # 分析镜像：{"input_latency_s": ..., "total_size": ...}。
        explicit = value.get("input_latency_s", value.get("total"))
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                return 0.0
        # 真实 ONN 仿真器：按内存层级的键 (GLB2_input_latency,
        # GLB1_output_latency, RF_input_latency, ...)。将它们求和：这些层级是
        # 单层数据搬运的串行层次阶段。
        total = 0.0
        for key, item in value.items():
            if isinstance(key, str) and key.endswith("_latency"):
                try:
                    total += float(item)
                except (TypeError, ValueError):
                    continue
        return total
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _flatten_sub_archs(by_device: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for sub_arch, devices in by_device.items():
        if isinstance(devices, dict):
            flattened.update(devices)
    return flattened


def _parse_device_key(device: str) -> Dict[str, Any]:
    """拆分仿真器的 ``<level>_<type>_<idx>_<instance>-<Variant>`` 键。

    代理以“激光器占主导”等方式推理，因此暴露各部分而非原始复合字符串。
    当键不匹配该形状时（如分析镜像键）返回 ``{}``，保留 ``device`` 作为标识。
    """
    head, _, variant = str(device).partition("-")
    parts = head.split("_")
    idx_pos = next((i for i in range(1, len(parts)) if parts[i].isdigit()), None)
    if idx_pos is None or idx_pos < 2 or idx_pos + 1 >= len(parts):
        return {}
    return {
        "level": parts[0],
        "device_type": "_".join(parts[1:idx_pos]),
        "instance": "_".join(parts[idx_pos + 1:]),
        "device_name": variant or None,
    }


def _summarize_device_records(by_device: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    rows = []
    total = 0.0
    for device, record in by_device.items():
        if not isinstance(record, dict):
            continue
        value = float(record.get(metric, 0.0) or 0.0)
        total += value
        row = {
            "device": device,
            metric: record.get(metric, 0.0),
            "count": record.get("count"),
            "chip_type": record.get("chip_type"),
        }
        row.update(_parse_device_key(device))
        rows.append(row)
    rows.sort(key=lambda row: float(row[metric] or 0.0), reverse=True)
    for row in rows:
        row[f"{metric}_share"] = (
            round(float(row[metric] or 0.0) / total, 6) if total else 0.0
        )
    return rows
