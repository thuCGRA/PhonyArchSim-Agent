"""光子设计空间探索的目标函数。

标量化采用归一化延迟、能耗与成本三者的乘积：::

    objective = latency_s * pJ_per_mac * area_um2

对光子加速器而言，成本项为芯片面积（一阶制造成本的近似），能耗项为每 MAC
能耗（论文中常引用的指标）。对于固定工作负载，
``pJ_per_mac = energy_pj / total_macs``，因此该目标与
``latency_s * energy_pj * area_um2`` 的排序一致（相差常数
``1 / total_macs``），同时归一化了目标的量级，使其可在不同工作负载间比较。

值越小越优。两项扩展让标量化真正利用 SimPhony 上报的权衡（例如
``num_wavelength``：更多波长可降低延迟/能耗，但会增大面积）：

* **可配置权重** -- ``search_space["objective_weights"]`` 将
  ``{latency, energy, area}``（或对应的指标键）映射为指数，形成加权几何平均。
  ``energy`` / ``energy_pj`` 是 ``pJ_per_mac`` 的别名。
* **插入损耗预算惩罚** -- 可选的单边软惩罚，将最长光路限制在
  ``search_space["il_budget"]``（dB）以内，用于约束光链路预算
  （激光功率与光电探测器灵敏度），而非数字吞吐目标。

权重只会重新分配指标模型中已存在的权衡，无法在 SimPhony 公式单调的区域
凭空制造权衡（例如 ``work_freq`` 受器件带宽硬约束限制，而非本标量化）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


DEFAULT_METRIC_KEYS = ("latency_s", "pJ_per_mac", "area_um2")
_WEIGHT_ALIASES = {
    "latency": "latency_s",
    "latency_s": "latency_s",
    "energy": "pJ_per_mac",
    "energy_pj": "pJ_per_mac",
    "pJ_per_mac": "pJ_per_mac",
    "area": "area_um2",
    "area_um2": "area_um2",
}


def objective_weights(search_space: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """返回 ``{metric_key: exponent}``，默认指数为 1.0（等权重）。

    接收 ``search_space["objective_weights"]``，其键既可以是短名称
    （``latency`` / ``energy`` / ``area``），也可以是指标键本身。
    """
    raw = (search_space or {}).get("objective_weights") if isinstance(search_space, Mapping) else None
    weights = {key: 1.0 for key in DEFAULT_METRIC_KEYS}
    if not isinstance(raw, Mapping):
        return weights
    for name, exponent in raw.items():
        key = _WEIGHT_ALIASES.get(str(name))
        if key is None:
            continue
        weights[key] = _positive_float(exponent, default=1.0)
    return weights


def base_objective(
    metrics: Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """三个必需正指标的加权乘积。"""
    weights = weights or {key: 1.0 for key in DEFAULT_METRIC_KEYS}
    value = 1.0
    for key in DEFAULT_METRIC_KEYS:
        value *= _required_positive_metric(metrics, key) ** float(weights.get(key, 1.0))
    return value


def objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> float:
    return base_objective(metrics, objective_weights(search_space)) * il_budget_penalty(
        metrics, search_space
    )["il_budget_penalty"]


def metrics_with_objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """返回附加了标量化目标值的指标字典。"""
    weights = objective_weights(search_space)
    enriched: Dict[str, float] = {key: float(metrics[key]) for key in metrics}
    enriched["base_objective"] = base_objective(metrics, weights)
    # 权重以标量键（而非嵌套字典）暴露，保证本函数幂等：调优器会对
    # 已扩充的指标再次调用本函数，若使用字典值会使上面的 ``float(...)``
    # 抛出 TypeError。
    for key, weight in weights.items():
        enriched[f"weight_{key}"] = weight
    enriched.update(il_budget_penalty(metrics, search_space))
    enriched["objective"] = objective(metrics, hardware, search_space)
    return enriched


def il_budget_penalty(
    metrics: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """光链路超过插入损耗预算时的单边软惩罚。

    约束光链路预算——即激光在仍能满足光电探测器灵敏度的前提下所能克服的
    最大总插入损耗（dB）。当设计的 ``core_insertion_loss`` 超过
    ``search_space["il_budget"]`` 时，仅对超预算一侧进行惩罚（预算不足已通过
    延迟计入成本），并使用有界二次项，使其保持软约束。
    """
    ss = search_space if isinstance(search_space, Mapping) else {}
    target = _positive_float(ss.get("il_budget"), default=None)
    actual = _positive_float(metrics.get("core_insertion_loss"), default=None)
    if target is None or actual is None:
        return {"il_budget_penalty": 1.0}
    lam = _positive_float(ss.get("il_budget_lambda"), default=1.0)
    ratio = actual / target
    penalty = 1.0 + lam * max(0.0, ratio - 1.0) ** 2
    return {
        "now_insertion_loss": actual,
        "il_budget": target,
        "il_budget_penalty": penalty,
    }


def _required_positive_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        raw = metrics[key]
    except KeyError as exc:
        raise KeyError(f"Evaluation metrics missing required key: {key}") from exc
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation metric {key} must be numeric, got {raw!r}") from exc
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"Evaluation metric {key} must be finite and positive, got {value!r}")
    return value


def _positive_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0.0 or not math.isfinite(parsed):
        return default
    return parsed
