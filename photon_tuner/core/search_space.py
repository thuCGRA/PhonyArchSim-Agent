"""光子设计搜索空间与硬件操作 —— 搜索空间/搜索操作的**唯一** Python 实现。

## 单一数据源

搜索空间的**全部候选值**只在一处声明：仓库根目录的 ``configs/search_space.json``。
本模块负责读取、校验并暴露它；"某个参数的候选是什么"不应在任何别处硬编码。

两处例外，均有意为之：

* ``core_kind`` / ``node_kind`` 的候选来自拓扑注册表
  （:mod:`photon_tuner.core.topology_space` 的 ``CORE_KINDS`` / ``NODE_KINDS``）——
  那里同时承载 ``evaluable`` / ``real`` 标志、core→node 耦合矩阵与器件槽位，是这些
  种类的物理事实来源。JSON 里再列一遍就会静默漂移，因此本模块**注入**这两个参数的
  候选；若 JSON 显式给出 ``values`` 会直接报错。
* 器件变体候选来自 SimPhony 器件库 YAML，运行时发现
  （:func:`discover_device_catalog`）。JSON 中的器件清单只是文档快照，不参与搜索。

## 内容

本模块合并了原先的两个子模块：

1. **搜索空间模型**（原 ``search_space/design_space.py``）—— 分层 :class:`DesignSpace`，
   提供 ``candidates`` / ``step`` / ``modify`` / ``refine`` / ``violations`` /
   ``normalize``，以及按拓扑族收窄候选的 ``effective_candidates``。
2. **硬件操作**（原 ``search_space/hardware_ops.py``）—— agent 直接调用的 ``dict``
   硬件辅助函数：``materialize_hardware`` / ``step_arch_parameter`` /
   ``modify_arch_parameter`` / ``switch_topology_template`` / ``set_device_choice``，
   以及器件库发现与 ``hardware_summary``。

两层共享同一份规格：:class:`DesignSpace` 从 ``configs/search_space.json`` 构建，
``hardware_ops`` 的扁平回退空间由**同一份**规格派生，因此不存在"两套候选"。
"""

from __future__ import annotations

import copy
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import yaml

from photon_tuner.common.paths import CONFIG_ROOT
from photon_tuner.core.topology import (
    DEFAULT_LASER_DISTRIBUTION,
    DEFAULT_SHARING_POLICY,
    DEFAULT_TEMPORAL_ACCUM,
    LASER_DISTRIBUTIONS,
    SHARING_POLICIES,
    TEMPORAL_ACCUM,
)
from photon_tuner.core.topology_space import (
    COUPLING_MATRIX,
    CORE_KINDS,
    coerce_forced_values,
    default_core_kind_candidates,
    default_node_kind_candidates,
    forced_value_violations,
    is_param_effective,
    param_status,
)

# --------------------------------------------------------------------------- #
# 搜索空间规格：从 configs/search_space.json 载入（唯一数据源）。
# --------------------------------------------------------------------------- #

SEARCH_SPACE_JSON: Path = CONFIG_ROOT / "search_space.json"

# 候选由拓扑注册表派生、不允许在 JSON 中重复声明的参数。
_REGISTRY_DERIVED_PARAMS: Dict[str, Any] = {
    "core_kind": default_core_kind_candidates,
    "node_kind": default_node_kind_candidates,
}


def _inject_registry_candidates(spec: Dict[str, Any]) -> Dict[str, Any]:
    """把 ``core_kind`` / ``node_kind`` 的候选从拓扑注册表注入分层规格。

    对这两个参数只允许单一来源：

    * 规格里**没有** ``values`` —— 注入注册表候选（JSON 的常态）。
    * 规格里**已有** ``values`` —— 必须是注册表派生值的等值快照（例如已经注入过的
      规格再次传入 :class:`DesignSpace`），否则报错。这样重复声明不会被静默接受，
      也不会因注入不幂等而误报。
    """
    for layer in (spec.get("layers") or {}).values():
        for param, derive in _REGISTRY_DERIVED_PARAMS.items():
            p_spec = (layer.get("params") or {}).get(param)
            if p_spec is None:
                continue
            expected = list(derive())
            declared = p_spec.get("values")
            if declared and [str(v) for v in declared] != [str(v) for v in expected]:
                raise ValueError(
                    f"search space declares candidates for {param!r} that disagree "
                    f"with the topology registry: declared={declared}, "
                    f"registry={expected}. Edit topology_space.CORE_KINDS / "
                    f"NODE_KINDS instead of duplicating the list here."
                )
            p_spec["values"] = expected
    return spec


def load_search_space(path: Optional[Any] = None) -> Dict[str, Any]:
    """读取并校验一份搜索空间 JSON，返回可直接交给 :class:`DesignSpace` 的规格。

    ``path`` 省略时读取 :data:`SEARCH_SPACE_JSON`。返回值中的 ``core_kind`` /
    ``node_kind`` 候选已由拓扑注册表注入。
    """
    target = Path(path) if path is not None else SEARCH_SPACE_JSON
    with open(target, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    return _inject_registry_candidates(spec)


#: 默认分层规格（公开常量，等价于 ``load_search_space()``）。
LAYERED_DESIGN_SPACE: Dict[str, Any] = load_search_space()

# ``map`` 是从 ``hardware`` 字典根到参数值的路径。
# ``refine`` 标记可自适应细化步长的线性数值参数。
# （原 ``mode`` 字段已移除——仿真只走 real，不存在"该引擎是否消费"的区分。）



# --------------------------------------------------------------------------- #
# 声明式分层规格（默认值；可从 JSON 加载/覆盖）。
# --------------------------------------------------------------------------- #

# ``map`` 是从 ``hardware`` 字典根到参数值的路径。
# ``mode``: "both" = 解析镜像与 real SimPhony 均消费；
#          "real" = 仅 real CACTI 存储模型消费。
# ``refine`` 标记可自适应细化步长的线性数值参数。


def _lookup(hw: Dict[str, Any], path: Iterable[str]) -> Any:
    node: Any = hw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _assign(hw: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    node = hw
    *parents, leaf = list(path)
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def _core_kind_of(hardware: Optional[Dict[str, Any]]) -> Optional[str]:
    """从 hardware 字典读取 ``arch.topology.core_kind``（缺失时返回 ``None``）。"""
    if not isinstance(hardware, dict):
        return None
    topo = ((hardware.get("arch") or {}).get("topology") or {})
    value = topo.get("core_kind")
    return str(value) if value is not None else None


class DesignSpace:
    """A hierarchical, multi-granularity, adaptive search space."""

    def __init__(
        self,
        spec: Optional[Dict[str, Any]] = None,
        device_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.spec = _inject_registry_candidates(
            copy.deepcopy(spec or LAYERED_DESIGN_SPACE)
        )
        self.layers: Dict[str, Any] = self.spec.get("layers", {})
        # {器件类型: {器件名: {"cfgs": {...}, "type": ...}}}
        self.device_catalog: Dict[str, Any] = device_catalog or {}

    # -- 查询 ---------------------------------------------------------------- #

    def _iter_params(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        for layer_name, layer in self.layers.items():
            for param_name, spec in (layer.get("params") or {}).items():
                yield layer_name, param_name, spec

    def params(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(self._iter_params())

    def layer_of(self, param: str) -> Optional[str]:
        for layer_name, param_name, _ in self._iter_params():
            if param_name == param:
                return layer_name
        return None

    def param_spec(self, param: str) -> Dict[str, Any]:
        for _, param_name, spec in self._iter_params():
            if param_name == param:
                return spec
        raise KeyError(f"Unknown design-space parameter: {param}")

    def candidates(self, param: str) -> List[Any]:
        spec = self.param_spec(param)
        return list(spec.get("values", []))

    def effective_candidates(
        self,
        param: str,
        hardware: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """返回该参数在 ``hardware`` 当前拓扑族下的**可搜索**候选。

        若该参数在当前拓扑下不生效（被 ``mesh_rank`` 取代、物理固定、后端未接通，
        或该族无此概念），返回空列表——调用方据此跳过该维度，而不是在无效轴上
        消耗评估预算。``hardware`` 省略时视为无拓扑信息，退化为 :meth:`candidates`。
        """
        if hardware is None:
            return self.candidates(param)
        core_kind = _core_kind_of(hardware)
        if not is_param_effective(param, core_kind):
            return []
        return self.candidates(param)

    def param_effectiveness(
        self,
        param: str,
        hardware: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """返回该参数在当前拓扑族下的状态与可操作建议（供工具/闸门使用）。"""
        return param_status(param, _core_kind_of(hardware) if hardware is not None else None)

    # -- 硬件读写 ----------------------------------------------------------- #

    def get(self, hw: Dict[str, Any], param: str) -> Any:
        return _lookup(hw, self.param_spec(param)["map"])

    def set(self, hw: Dict[str, Any], param: str, value: Any) -> None:
        _assign(hw, self.param_spec(param)["map"], value)

    # -- 可行性校验 --------------------------------------------------------- #

    def _device_cfg(self, hw: Dict[str, Any], device_type: str) -> Dict[str, Any]:
        devices = hw.get("devices", {})
        name = (devices.get("core", {}).get(device_type)
                or devices.get("node", {}).get(device_type))
        if not name:
            return {}
        variant = self.device_catalog.get(device_type, {}).get(str(name), {})
        return dict(variant.get("cfgs", {}) or {})

    def max_work_freq(self, hw: Dict[str, Any]) -> Optional[float]:
        """MZM 符号速率上限（modulation_speed / testing_bit），单位 GHz。"""
        cfg = self._device_cfg(hw, "mzm")
        if not cfg or "modulation_speed" not in cfg:
            return None
        testing_bit = float(cfg.get("testing_bit", 1) or 1)
        return float(cfg["modulation_speed"]) / testing_bit

    def min_work_freq_ceiling(self, hw: Dict[str, Any]) -> Optional[float]:
        """最紧的器件时钟上限，单位 GHz。

        ``min(MZM symbol rate, DAC sample_rate, ADC sample_rate, photodetector
        bandwidth)`` —— 转换器与探测器各自独立构成上限，
        ``normalize`` 需钳位到其中最紧的一个。
        """
        ceilings: List[float] = []
        mzm = self.max_work_freq(hw)
        if mzm is not None:
            ceilings.append(mzm)
        for device_type, cfg_key in (
            ("dac", "sample_rate"),
            ("adc", "sample_rate"),
            ("photodetector", "bandwidth"),
        ):
            cfg = self._device_cfg(hw, device_type)
            limit = cfg.get(cfg_key)
            if limit is not None:
                ceilings.append(float(limit))
        return min(ceilings) if ceilings else None

    def violations(self, hw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """返回硬件字典违反的可行性约束。"""
        out: List[Dict[str, Any]] = []
        core = hw.get("arch", {}).get("optical_core", {})
        precision = core.get("precision", {})

        max_freq = self.max_work_freq(hw)
        freq = core.get("work_freq")
        if max_freq is not None and freq is not None and float(freq) > max_freq:
            out.append({
                "param": "work_freq",
                "constraint": "work_freq <= mzm.symbol_rate",
                "current": float(freq),
                "limit": max_freq,
                "message": f"work_freq {freq} GHz exceeds MZM symbol rate {max_freq} GHz",
            })

        # 转换器采样率与探测器带宽构成独立的时钟上限
        # （上面的 MZM 检查仅是调制器自身的符号速率上限）。
        # 它们从物理上约束 ``work_freq`` 旋钮的取值。
        for device_type, cfg_key in (("dac", "sample_rate"), ("adc", "sample_rate")):
            dev_cfg = self._device_cfg(hw, device_type)
            limit = dev_cfg.get(cfg_key)
            if limit is not None and freq is not None and float(freq) > float(limit):
                out.append({
                    "param": "work_freq",
                    "constraint": f"work_freq <= {device_type}.{cfg_key}",
                    "current": float(freq),
                    "limit": float(limit),
                    "message": f"work_freq {freq} GHz exceeds {device_type} {cfg_key} {limit} GHz",
                })

        pd_cfg = self._device_cfg(hw, "photodetector")
        pd_bw = pd_cfg.get("bandwidth")
        if pd_bw is not None and freq is not None and float(freq) > float(pd_bw):
            out.append({
                "param": "work_freq",
                "constraint": "work_freq <= photodetector.bandwidth",
                "current": float(freq),
                "limit": float(pd_bw),
                "message": f"work_freq {freq} GHz exceeds photodetector bandwidth {pd_bw} GHz",
            })

        dac_cfg = self._device_cfg(hw, "dac")
        in_bit = precision.get("in_bit")
        if "prec" in dac_cfg and in_bit is not None and int(in_bit) > int(dac_cfg["prec"]):
            out.append({
                "param": "precision_in_bit",
                "constraint": "in_bit <= dac.prec",
                "current": int(in_bit),
                "limit": int(dac_cfg["prec"]),
                "message": f"in_bit {in_bit} exceeds DAC precision {dac_cfg['prec']}",
            })

        adc_cfg = self._device_cfg(hw, "adc")
        out_bit = precision.get("out_bit")
        if "prec" in adc_cfg and out_bit is not None and int(out_bit) > int(adc_cfg["prec"]):
            out.append({
                "param": "precision_out_bit",
                "constraint": "out_bit <= adc.prec",
                "current": int(out_bit),
                "limit": int(adc_cfg["prec"]),
                "message": f"out_bit {out_bit} exceeds ADC precision {adc_cfg['prec']}",
            })

        return out

    def forced_violations(self, hw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """本设计点声称的值与其拓扑物理强制值不一致的部分。

        装配器会忽略这些声明的值并按拓扑写死，因此它们属于"物理上不成立的组合"：
        若不在此拦下，求值会产出**从未被仿真过**的配置的数值。
        """
        return forced_value_violations(hw)

    # -- 归一化 ------------------------------------------------------------- #

    def _clamp_to_candidates(self, hw: Dict[str, Any], param: str) -> None:
        candidates = self.candidates(param)
        if not candidates:
            return
        spec = self.param_spec(param)
        if spec.get("kind") == "enum":
            current = str(self.get(hw, param))
            if current not in [str(c) for c in candidates]:
                self.set(hw, param, candidates[0])
            return
        try:
            current = float(self.get(hw, param))
        except (TypeError, ValueError):
            return
        best = min(candidates, key=lambda c: abs(float(c) - current))
        self.set(hw, param, best)

    def normalize(self, hw: Dict[str, Any]) -> Dict[str, Any]:
        """将每个可搜索参数钳位到候选值并施加可行性约束。"""
        result = copy.deepcopy(hw)
        for _, param, _ in self._iter_params():
            self._clamp_to_candidates(result, param)

        # 可行性：钳位到不超过最紧器件上限的最大候选值
        # （对应 real SimPhony 的断言，而非直接报错）。
        max_freq = self.min_work_freq_ceiling(result)
        if max_freq is not None:
            feasible = [c for c in self.candidates("work_freq") if float(c) <= max_freq]
            if feasible:
                current = float(result["arch"]["optical_core"].get("work_freq", 0))
                result["arch"]["optical_core"]["work_freq"] = min(
                    feasible, key=lambda c: abs(float(c) - current)
                )
        for param, device in (("precision_in_bit", "dac"), ("precision_out_bit", "adc")):
            cfg = self._device_cfg(result, device)
            if "prec" in cfg:
                feasible = [c for c in self.candidates(param) if int(c) <= int(cfg["prec"])]
                if feasible:
                    current = int(result["arch"]["optical_core"]["precision"].get(
                        param.replace("precision_", ""), 0))
                    result["arch"]["optical_core"]["precision"][param.replace("precision_", "")] = min(
                        feasible, key=lambda c: abs(int(c) - current)
                    )
        # 拓扑物理强制值：归一化后的设计点必须就是装配器实际会建出来的那一个。
        # 例如网格下 dataflow 只能是 weight_stationary、num_wavelength 只能是 1；
        # 不做这步，设计点会一直记录着从未生效的取值。
        result = coerce_forced_values(result)
        return result

    # -- 编辑 --------------------------------------------------------------- #

    def step(self, hw: Dict[str, Any], param: str, direction: int) -> Dict[str, Any]:
        """将某个参数步进到下一个/上一个候选值（并归一化）。"""
        result = copy.deepcopy(hw)
        candidates = self.candidates(param)
        if not candidates:
            return result
        spec = self.param_spec(param)
        if spec.get("kind") == "enum":
            current = str(self.get(result, param))
            if current not in [str(c) for c in candidates]:
                self.set(result, param, candidates[0])
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                self.set(result, param, candidates[index])
            return self.normalize(result)
        ordered = sorted(set(candidates), key=lambda v: float(v))
        try:
            current = float(self.get(result, param))
        except (TypeError, ValueError):
            self.set(result, param, ordered[0])
            return self.normalize(result)
        if current not in [float(v) for v in ordered]:
            self.set(result, param, min(ordered, key=lambda v: abs(float(v) - current)))
        else:
            index = [float(v) for v in ordered].index(current)
            index = max(0, min(len(ordered) - 1, index + direction))
            self.set(result, param, ordered[index])
        return self.normalize(result)

    def modify(self, hw: Dict[str, Any], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用显式的 ``{parameter, value}`` 更新（然后归一化）。"""
        result = copy.deepcopy(hw)
        for update in updates:
            param = str(update.get("parameter", ""))
            value = update.get("value")
            spec = self.param_spec(param)  # 未知参数时抛出 KeyError
            if spec.get("kind") == "enum":
                self.set(result, param, str(value))
            else:
                self.set(result, param, int(round(float(value))))
        return self.normalize(result)

    # -- 自适应细化 --------------------------------------------------------- #

    def refine(self, center_hw: Dict[str, Any], factor: int = 2) -> "DesignSpace":
        """返回围绕 ``center_hw`` 的细粒度副本。

        线性 ``refine`` 参数在局部窗口 ``[center - step, center + step]`` 内细分步长；
        几何/枚举参数保持原网格。
        """
        new = copy.deepcopy(self)
        new.device_catalog = self.device_catalog
        for layer_name, param_name, spec in new._iter_params():
            if not spec.get("refine") or spec.get("kind") not in ("int", "bits"):
                continue
            values = [float(v) for v in spec.get("values", [])]
            if not values:
                continue
            step = float(spec.get("step", 1))
            try:
                center = float(self.get(center_hw, param_name))
            except (TypeError, ValueError, KeyError):
                continue
            new_step = max(1.0, step / max(1, factor))
            lo = max(min(values), center - step)
            hi = min(max(values), center + step)
            refined: List[float] = []
            v = lo
            while v <= hi + 1e-9:
                refined.append(round(v))
                v += new_step
            if center not in refined:
                refined.append(round(center))
            spec["values"] = sorted(set(int(v) for v in refined))
            spec["step"] = int(new_step)
        return new

    # -- 面向 LLM 的摘要 ---------------------------------------------------- #

    def summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for layer_name, param_name, spec in self._iter_params():
            rows.append({
                "layer": layer_name,
                "parameter": param_name,
                "kind": spec.get("kind"),
                "values": spec.get("values"),
                "step": spec.get("step"),
                "constraint": spec.get("constraint"),
                "refine": bool(spec.get("refine")),
            })
        return rows


def build_design_space(
    search_space: Optional[Dict[str, Any]] = None,
    device_catalog: Optional[Dict[str, Any]] = None,
) -> DesignSpace:
    """从 JSON/字典规格构建 :class:`DesignSpace`。

    同时接受分层格式（``{"layers": {...}}``）与扁平格式（``{"arch": {...}}``）。
    扁平格式会映射到分层默认值上，以保留存储与可行性能力。
    """
    if search_space and search_space.get("layers"):
        return DesignSpace(spec=search_space, device_catalog=device_catalog)

    # 扁平规格：从分层默认值出发，再收窄其实际提及参数的候选网格。
    spec = copy.deepcopy(LAYERED_DESIGN_SPACE)
    flat = (search_space or {}).get("arch") or {}
    if flat:
        aliases = {
            "tiles": "tiles",
            "cores_per_tile": "cores_per_tile",
            "core_height": "core_height",
            "core_width": "core_width",
            "num_wavelength": "num_wavelength",
            "work_freq": "work_freq",
            "dataflow": "dataflow",
            "hbm_bandwidth": "hbm_bandwidth",
            "glb1_cache_size": "glb1_cache_size",
            "glb2_cache_size": "glb2_cache_size",
            "rf_cache_size": "rf_cache_size",
            "sharing_policy": "sharing_policy",
            "temporal_accum": "temporal_accum",
            "laser_distribution": "laser_distribution",
        }
        for layer in spec["layers"].values():
            for param, p_spec in (layer.get("params") or {}).items():
                key = aliases.get(param)
                if key and key in flat:
                    p_spec["values"] = list(flat[key])
        prec = flat.get("precision") or {}
        for bit_key, param in (("in_bit", "precision_in_bit"),
                               ("w_bit", "precision_w_bit"),
                               ("out_bit", "precision_out_bit")):
            if bit_key in prec:
                spec["layers"]["architecture"]["params"][param]["values"] = list(prec[bit_key])
    return DesignSpace(spec=spec, device_catalog=device_catalog)


# --------------------------------------------------------------------------- #
# 硬件操作（原 search_space/hardware_ops.py）
# --------------------------------------------------------------------------- #


def _flat_arch_space(search_space: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """返回搜索空间规格的扁平 ``{参数: 候选列表}`` 架构空间。

    同时接受分层格式（``{"layers": {...}}``）与扁平格式（``{"arch": {...}}``）。
    ``precision_*`` 参数归入 step/modify 辅助函数所用的嵌套 ``precision`` 表。
    ``search_space`` 省略时使用 :data:`LAYERED_DESIGN_SPACE`——即同一份 JSON，
    因此不存在第二套候选。
    """
    spec = search_space or LAYERED_DESIGN_SPACE
    layers = spec.get("layers")
    if not layers:
        return dict(
            spec.get("arch") or _flat_arch_space(LAYERED_DESIGN_SPACE)
        )
    flat: Dict[str, Any] = {}
    for layer in layers.values():
        for param, p_spec in (layer.get("params") or {}).items():
            values = list((p_spec or {}).get("values", []))
            if not values:
                continue
            if param.startswith("precision_"):
                flat.setdefault("precision", {})[param.replace("precision_", "")] = values
            else:
                flat[param] = values
    return flat


#: 扁平架构空间——**派生**自 :data:`LAYERED_DESIGN_SPACE`（即同一份 JSON），
#: 不再有独立的字面量副本。保留该名称供既有导入使用。
DEFAULT_ARCH_SEARCH_SPACE: Dict[str, Any] = _flat_arch_space(LAYERED_DESIGN_SPACE)


# 暴露给方案生成 agent 的可调类别。顺序即 ``orchestrator._repair_duplicate``
# 逃离重复设计时的步进优先级；成员集合必须与搜索空间规格一致
# （由 tests/test_search_space_single_source.py 守护）。
ARCH_PARAM_KEYS = [
    "tiles",
    "cores_per_tile",
    "core_height",
    "core_width",
    "num_wavelength",
    "work_freq",
    "dataflow",
    "precision_in_bit",
    "precision_w_bit",
    "precision_out_bit",
    "hbm_bandwidth",
    "glb1_cache_size",
    "glb2_cache_size",
    "rf_cache_size",
    "sharing_policy",
    "temporal_accum",
    "laser_distribution",
    "core_kind",
    "node_kind",
    "mesh_rank",
]

# 存储层级参数名 -> (存储层级, 该层级字典中的字段)。
_MEMORY_PARAM_MAP = {
    "hbm_bandwidth": ("HBM", "bandwidth"),
    "glb1_cache_size": ("GLB1", "cache_size"),
    "glb2_cache_size": ("GLB2", "cache_size"),
    "rf_cache_size": ("RF", "cache_size"),
}

# 基线器件选择，对应 SimPhony 项目中的 configs/design/cores/dptc.yml 与
# configs/design/nodes/dode.yml。
DEFAULT_CORE_DEVICES = {
    "mzm": "LT_MZM",
    "dac": "DAC_2",
    "adc": "ADC_SAR_1",
    "tia": "TIA_1",
    "waveguide": "WG_Single_Strip",
    "mrr_rerouter": "LT_MRR_REROUTER",
    "laser_splitter": "Customized_Laser_Splitter",
    "micro_comb": "LT_Micro_Comb",
    "on_chip_laser": "Customized_Laser",
}

DEFAULT_NODE_DEVICES = {
    "phase_shifter": "MEMS_PS",
    "photodetector": "LT_PD",
    "coupler": "LT_DC_2x2",
    "y_branch": "LT_Y_Branch",
}

# Mesh 范式的器件表。MZI-mesh 的 core/node 电路结构与 crossbar 不同：
# core 承载离片激光器与输出光电探测器，node 仅为耦合器 + 移相器。
# 这些默认值复用既有库变体（无新增硬件）；mesh 评估器
# （analytic.build_mesh_sub_arch）从 ``hardware["devices"]["mesh"]`` 读取它们。
DEFAULT_MESH_CORE_DEVICES = {
    "off_chip_laser": "LT_Laser",
    "photodetector": "LT_PD",
    "adc": "ADC_SAR_1",
    "tia": "TIA_1",
    "dac": "DAC_2",
    "waveguide": "WG_Single_Strip",
    # crossing-bearing mesh（Reck/butterfly）的布线交叉；Clements 无交叉时不注入。
    "crossing": "WG_Crossing_Parabolic",
}

DEFAULT_MESH_NODE_DEVICES = {
    "coupler": "LT_DC_2x2",
    "phase_shifter": "IHO_PS",
}

# WDM 权重阵列范式（MRR 权重组 / PCM 交叉开关 / 单 WDM 链路）的器件表。
# core 承载微梳（WDM 源）、PD、电子器件；node 为权重/调制器单元。
DEFAULT_WEIGHT_ARRAY_CORE_DEVICES = {
    "off_chip_laser": "LT_Laser",
    "micro_comb": "MC_Si3N4_DKS_100GHz",
    "photodetector": "LT_PD",
    "adc": "ADC_SAR_1",
    "tia": "TIA_1",
    "dac": "DAC_2",
    "waveguide": "WG_Single_Strip",
}

DEFAULT_WEIGHT_ARRAY_NODE_DEVICES = {
    "weight_mrr": "WG_MRR_WEIGHT_TAIT",
    "pcm_cell": "PCM_CELL_FELDMANN",
    "mzm": "LT_MZM",
}


def default_hardware() -> Dict[str, Any]:
    """返回基线光子配置（SimPhony LT 设计）。

    hardware 字典对应分层设计空间，含四个部分：``arch.system``（分块/数据流）、
    ``arch.optical_core``（crossbar 几何/精度/时钟）、``arch.topology``
    （共享/时域累加/激光分布）与 ``arch.memory``，外加顶层的 ``devices`` 表
    用于库变体选择。
    """
    return {
        "arch": {
            "system": {
                "tiles": 4,
                "cores_per_tile": 2,
                "dataflow": "output_stationary",
                "op_type": "MVM",
            },
            "optical_core": {
                "type": "Crossbar",
                "coherent": True,
                "universal": True,
                "height": 12,
                "width": 12,
                "num_wavelength": 12,
                "work_freq": 5,
                "precision": {"in_bit": 4, "w_bit": 4, "out_bit": 4},
            },
            "topology": {
                "core_kind": "dptc_crossbar",
                "node_kind": "dode",
                "sharing_policy": DEFAULT_SHARING_POLICY,
                "temporal_accum": DEFAULT_TEMPORAL_ACCUM,
                "laser_distribution": DEFAULT_LASER_DISTRIBUTION,
                # MZI-mesh 酉变换阶数 (N)。crossbar 下无效；mesh 评估器读取该值
                # （缺省时回退到 optical_core.height）。
                "mesh_rank": 16,
            },
            "memory": {
                "HBM": {"read_energy": 62.4, "write_energy": 62.4, "bandwidth": 1024},
                "GLB2": {"cache_size": 485, "block_size": 8},
                "GLB1": {"cache_size": 485, "block_size": 8},
                "RF": {"cache_size": 485, "block_size": 8},
            },
        },
        "devices": {
            "core": {
                **dict(DEFAULT_CORE_DEVICES),
                # 实例级器件类型开关（默认保持 LT 基线：
                # 在片激光器 + 波导激光分束器）。
                "laser_source": "on_chip",
                "splitter_kind": "laser_splitter",
            },
            "node": dict(DEFAULT_NODE_DEVICES),
            # Mesh 范式的器件表：无论当前 core_kind 为何都会填充，
            # 使 crossbar -> mesh 的拓扑切换无需再编辑器件即拥有合法且有来源的器件集。
            "mesh": {
                "core": dict(DEFAULT_MESH_CORE_DEVICES),
                "node": dict(DEFAULT_MESH_NODE_DEVICES),
            },
            # WDM 权重阵列范式（MRR/PCM/WDM 链路）：无论当前 core_kind 为何都会填充。
            "weight_array": {
                "core": dict(DEFAULT_WEIGHT_ARRAY_CORE_DEVICES),
                "node": dict(DEFAULT_WEIGHT_ARRAY_NODE_DEVICES),
            },
        },
    }


# --------------------------------------------------------------------------- #
# 器件库发现（读取 SimPhony 的器件 YAML）。
# --------------------------------------------------------------------------- #

# 从可搜索器件面中隐藏的理想化/占位变体。
#
# 这些条目存在于 SimPhony 的 YAML 器件库中，但不得提供给器件级搜索
# （目录列表、agent 工具箱、证据探针），因为它们是物理上不可实现的占位符
# （无 ref、参数不可实现），会在目标轴上严格支配所有真实变体——P_pi=0 /
# response_time=0 的移相器、无插损（视为 0 dB）的激光器、10 nm 宽的 Y 分支、
# 以及占用 2x2 槽位的 1x1 光栅耦合器。若允许搜索，器件搜索会退化
# （优化器总是选中同一个理想赢家），并让 DSE 产出引用不存在器件的硬件配置。
#
# 这些变体仍保留在 YAML 器件库中（仿真器回读时仍可解析），
# 但发现、探针与工具箱不再暴露它们。
EXCLUDED_SEARCH_VARIANTS: Dict[str, frozenset] = {
    "phase_shifter": frozenset({"Customized_PS"}),        # 理想：P_pi=0, response_time=0, 无 ref
    "on_chip_laser": frozenset({"LT_Laser"}),             # 理想：无插损 -> 0 dB, 无 ref
    "y_branch": frozenset({"Customized_Y_Branch"}),       # 占位：10-nm 宽度, 无 ref
    "coupler": frozenset({"Customized_DC_1x1"}),          # 槽位无效：1x1 光栅占用 2x2 槽位
}


def discover_device_catalog(devicelib_root: str) -> Dict[str, Dict[str, Any]]:
    """将 ``devicelib_root`` 下每个器件 YAML 载入嵌套字典。

    返回 ``{器件类型: {器件名: {"cfgs": {...}, "type": ...}}}``，
    并排除 :data:`EXCLUDED_SEARCH_VARIANTS` 中的理想化变体。
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    files = glob.glob(os.path.join(devicelib_root, "*", "*.yml"))
    for path in sorted(files):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for device_type, variants in data.items():
            bucket = catalog.setdefault(str(device_type), {})
            excluded = EXCLUDED_SEARCH_VARIANTS.get(str(device_type), frozenset())
            for name, body in variants.items():
                if isinstance(body, dict) and str(name) not in excluded:
                    bucket[str(name)] = body
    return catalog


def discover_device_variants(devicelib_root: str) -> Dict[str, List[str]]:
    """从器件库返回 ``{器件类型: [变体名列表]}``。"""
    catalog = discover_device_catalog(devicelib_root)
    return {
        device_type: sorted(variants.keys())
        for device_type, variants in catalog.items()
    }


# --------------------------------------------------------------------------- #
# 硬件归一化 / 步进辅助函数。
# --------------------------------------------------------------------------- #


def materialize_hardware(hardware: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """返回填入架构默认值的深拷贝。"""
    base = default_hardware()
    if hardware:
        _deep_update(base, dict(hardware))
    return base


def normalize_hardware_to_design_space(
    hardware: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """将标量架构值钳位到最近的合法候选值。"""
    arch_space = _flat_arch_space(search_space)
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]

    def _clamp(key: str, candidates: List[Any], target: Dict[str, Any]) -> None:
        if key in target and candidates:
            try:
                current = float(target[key])
            except (TypeError, ValueError):
                return
            target[key] = min(candidates, key=lambda c: abs(float(c) - current))

    _clamp("tiles", arch_space.get("tiles", []), system)
    _clamp("cores_per_tile", arch_space.get("cores_per_tile", []), system)
    _clamp("height", arch_space.get("core_height", []), oc)
    _clamp("width", arch_space.get("core_width", []), oc)
    _clamp("num_wavelength", arch_space.get("num_wavelength", []), oc)
    _clamp("work_freq", arch_space.get("work_freq", []), oc)

    precision_space = arch_space.get("precision", {})
    precision = oc.setdefault("precision", {})
    for bit_key in ("in_bit", "w_bit", "out_bit"):
        candidates = precision_space.get(bit_key, [])
        if bit_key in precision and candidates:
            try:
                current = float(precision[bit_key])
            except (TypeError, ValueError):
                continue
            precision[bit_key] = min(candidates, key=lambda c: abs(float(c) - current))

    memory = arch.setdefault("memory", {})
    for param, (level, field) in _MEMORY_PARAM_MAP.items():
        candidates = arch_space.get(param, [])
        if not candidates:
            continue
        target = memory.setdefault(level, {})
        if field in target:
            try:
                current = float(target[field])
            except (TypeError, ValueError):
                continue
            target[field] = min(candidates, key=lambda c: abs(float(c) - current))

    dataflow = str(system.get("dataflow", ""))
    dataflow_candidates = arch_space.get("dataflow", [])
    if dataflow_candidates and dataflow not in dataflow_candidates:
        system["dataflow"] = dataflow_candidates[0]

    for param in ("core_kind", "node_kind", "sharing_policy", "temporal_accum", "laser_distribution"):
        candidates = arch_space.get(param, [])
        if not candidates:
            continue
        current = str(topo.get(param, ""))
        if current not in [str(c) for c in candidates]:
            topo[param] = candidates[0]

    # mesh_rank 是数值型拓扑参数（酉变换阶数 N）；与上面的标量架构参数
    # 一样，钳位到最近的合法候选值。
    mesh_candidates = arch_space.get("mesh_rank", [])
    if mesh_candidates:
        try:
            current_mesh = float(topo.get("mesh_rank", 16))
        except (TypeError, ValueError):
            current_mesh = float(mesh_candidates[0])
        topo["mesh_rank"] = min(mesh_candidates, key=lambda c: abs(float(c) - current_mesh))

    return result


def step_arch_parameter(
    hardware: Mapping[str, Any],
    parameter: str,
    direction: int,
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """返回将某个架构参数步进 +1 / -1 后的新硬件字典。"""
    arch_space = _flat_arch_space(search_space)
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]

    mapping = {
        "tiles": ("tiles", system, arch_space.get("tiles", [])),
        "cores_per_tile": ("cores_per_tile", system, arch_space.get("cores_per_tile", [])),
        "core_height": ("height", oc, arch_space.get("core_height", [])),
        "core_width": ("width", oc, arch_space.get("core_width", [])),
        "num_wavelength": ("num_wavelength", oc, arch_space.get("num_wavelength", [])),
        "work_freq": ("work_freq", oc, arch_space.get("work_freq", [])),
        "mesh_rank": ("mesh_rank", topo, arch_space.get("mesh_rank", [])),
    }
    if parameter in mapping:
        key, target, candidates = mapping[parameter]
        if not candidates:
            return result
        target[key] = _step_value(target.get(key), candidates, direction)
        return result

    precision_map = {
        "precision_in_bit": "in_bit",
        "precision_w_bit": "w_bit",
        "precision_out_bit": "out_bit",
    }
    if parameter in precision_map:
        bit_key = precision_map[parameter]
        candidates = (arch_space.get("precision") or {}).get(bit_key, [])
        precision = oc.setdefault("precision", {})
        if candidates:
            precision[bit_key] = _step_value(precision.get(bit_key), candidates, direction)
        return result

    if parameter in _MEMORY_PARAM_MAP:
        level, field = _MEMORY_PARAM_MAP[parameter]
        candidates = arch_space.get(parameter, [])
        memory = arch.setdefault("memory", {})
        target = memory.setdefault(level, {})
        if candidates:
            target[field] = _step_value(target.get(field), candidates, direction)
        return result

    if parameter == "dataflow":
        candidates = list(arch_space.get("dataflow", []))
        if candidates:
            current = str(system.get("dataflow", ""))
            if current not in [str(c) for c in candidates]:
                system["dataflow"] = candidates[0]
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                system["dataflow"] = candidates[index]
        return result

    if parameter in {"core_kind", "node_kind", "sharing_policy", "temporal_accum", "laser_distribution"}:
        candidates = list(arch_space.get(parameter, []))
        if candidates:
            current = str(topo.get(parameter, ""))
            if current not in [str(c) for c in candidates]:
                topo[parameter] = candidates[0]
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                topo[parameter] = candidates[index]
        return result

    raise ValueError(f"Unknown architecture parameter: {parameter}")


def modify_arch_parameter(
    hardware: Mapping[str, Any],
    updates: List[Dict[str, Any]],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """应用显式的架构参数更新并钳位到候选值。"""
    result = materialize_hardware(hardware)
    arch = result["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]
    precision = oc.setdefault("precision", {})

    for update in updates:
        parameter = str(update.get("parameter", ""))
        value = update.get("value")
        if parameter == "dataflow":
            system["dataflow"] = str(value)
        elif parameter == "tiles":
            system["tiles"] = _as_int(value)
        elif parameter == "cores_per_tile":
            system["cores_per_tile"] = _as_int(value)
        elif parameter == "core_height":
            oc["height"] = _as_int(value)
        elif parameter == "core_width":
            oc["width"] = _as_int(value)
        elif parameter == "num_wavelength":
            oc["num_wavelength"] = _as_int(value)
        elif parameter == "work_freq":
            oc["work_freq"] = _as_int(value)
        elif parameter in {"precision_in_bit", "precision_w_bit", "precision_out_bit"}:
            precision[parameter.replace("precision_", "")] = _as_int(value)
        elif parameter in _MEMORY_PARAM_MAP:
            level, field = _MEMORY_PARAM_MAP[parameter]
            memory = arch.setdefault("memory", {})
            memory.setdefault(level, {})[field] = _as_int(value)
        elif parameter in {"sharing_policy", "temporal_accum", "laser_distribution"}:
            topo[parameter] = str(value)
        elif parameter == "mesh_rank":
            topo["mesh_rank"] = _as_int(value)
        elif parameter in {"core_kind", "node_kind"}:
            topo[parameter] = str(value)
        else:
            raise ValueError(f"Unknown architecture parameter: {parameter}")

    return normalize_hardware_to_design_space(result, search_space)


def switch_topology_template(
    hardware: Mapping[str, Any],
    core_kind: str,
) -> Dict[str, Any]:
    """原子地切换到 ``core_kind``，并自动对齐 ``node_kind``。

    拓扑结构变更属于 *耦合* 操作：核心模板通过
    :data:`~photon_tuner.core.topology_space.COUPLING_MATRIX` 固定其节点模板，
    所需器件槽位从对应器件表（``devices.core``/``devices.node`` 与
    ``devices.mesh``）读取。仅要求 ``core_kind`` 并由本函数选定兼容的
    ``node_kind``，可避免产生半切换（耦合不一致）的设计点。

    拒绝未知或不可求值的种类（如无来源的 butterfly），但不强制模式相关的
    可求值性——该检查由 gate 的 ``structure_legality_check`` 依据仿真器模式复核。
    """
    core_kind = str(core_kind)
    if core_kind not in CORE_KINDS:
        raise ValueError(
            f"unknown core_kind {core_kind!r}; legal: {sorted(CORE_KINDS)}"
        )
    if not CORE_KINDS[core_kind].get("evaluable"):
        raise ValueError(
            f"core_kind {core_kind!r} is not evaluable (no simulator path); "
            f"cannot be selected"
        )
    allowed = COUPLING_MATRIX.get(core_kind, {}).get("node_kinds", [])
    if not allowed:
        raise ValueError(
            f"core_kind {core_kind!r} has no compatible node_kind in COUPLING_MATRIX"
        )
    result = materialize_hardware(hardware)
    topo = result["arch"].setdefault("topology", {})
    topo["core_kind"] = core_kind
    topo["node_kind"] = allowed[0]
    # 切换后必须立刻对齐该拓扑族的物理强制值，否则设计点会保留 crossbar 的
    # dataflow / 阵列宽度 / 波长数——这三者在网格与权重阵列下并不存在，
    # 会产出一个"声明值从未被仿真"的设计点。
    return coerce_forced_values(result)


def set_device_choice(
    hardware: Mapping[str, Any],
    level: str,
    device_type: str,
    device_name: str,
) -> Dict[str, Any]:
    """在 ``core``/``node`` 级为某器件类型设置器件变体。

    ``level`` 也可为 ``mesh_core`` / ``mesh_node``，以指向 MZI-mesh 器件表
    （``devices.mesh.core`` / ``devices.mesh.node``）。

    :data:`EXCLUDED_SEARCH_VARIANTS` 中列出的理想化变体会被拒绝：它们被刻意排除
    在搜索空间之外（无参考、参数理想如 ``P_pi=0``），接受选择会把不物理的设计点
    置于 Pareto 前沿。
    """
    excluded = EXCLUDED_SEARCH_VARIANTS.get(str(device_type), frozenset())
    if str(device_name) in excluded:
        raise ValueError(
            f"device variant {device_name!r} is excluded from the search space for "
            f"{device_type!r} (idealized/unsourced); choose another variant"
        )
    result = materialize_hardware(hardware)
    if level in {"core", "node"}:
        result["devices"][level][device_type] = device_name
    elif level in {"mesh_core", "mesh_node"}:
        sub = level.replace("mesh_", "")
        result["devices"]["mesh"][sub][device_type] = device_name
    else:
        raise ValueError(f"level must be 'core', 'node', 'mesh_core' or 'mesh_node', got {level!r}")
    return result


def hardware_summary(hardware: Mapping[str, Any]) -> Dict[str, Any]:
    """光子硬件配置的精简、LLM 友好摘要。"""
    hw = materialize_hardware(hardware)
    arch = hw["arch"]
    system = arch["system"]
    oc = arch["optical_core"]
    topo = arch["topology"]
    return {
        "arch": {
            "tiles": system.get("tiles"),
            "cores_per_tile": system.get("cores_per_tile"),
            "core_height": oc.get("height"),
            "core_width": oc.get("width"),
            "num_wavelength": oc.get("num_wavelength"),
            "work_freq": oc.get("work_freq"),
            "precision": oc.get("precision"),
            "core_kind": topo.get("core_kind"),
            "node_kind": topo.get("node_kind"),
            "sharing_policy": topo.get("sharing_policy"),
            "temporal_accum": topo.get("temporal_accum"),
            "laser_distribution": topo.get("laser_distribution"),
            "mesh_rank": topo.get("mesh_rank"),
            "dataflow": system.get("dataflow"),
            "memory": {level: dict(cfg) for level, cfg in arch.get("memory", {}).items()},
        },
        "core_devices": dict(hw.get("devices", {}).get("core", {})),
        "node_devices": dict(hw.get("devices", {}).get("node", {})),
        "mesh_core_devices": dict(hw.get("devices", {}).get("mesh", {}).get("core", {})),
        "mesh_node_devices": dict(hw.get("devices", {}).get("mesh", {}).get("node", {})),
    }


# --------------------------------------------------------------------------- #
# 内部辅助函数。
# --------------------------------------------------------------------------- #


def _step_value(current: Any, candidates: List[Any], direction: int) -> Any:
    ordered = sorted(set(candidates), key=lambda v: float(v))
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return ordered[0]
    if current_value not in [float(v) for v in ordered]:
        return min(ordered, key=lambda v: abs(float(v) - current_value))
    index = [float(v) for v in ordered].index(current_value)
    index = max(0, min(len(ordered) - 1, index + direction))
    return ordered[index]


def _as_int(value: Any) -> int:
    return int(round(float(value)))


def _deep_update(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)