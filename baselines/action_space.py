"""统一搜索空间 —— 传统算法基线与 LLM 多智能体共用同一搜索空间。

本模块为一次硬件编辑定义统一的动作表示，并生成当前设计点周围的合法邻域。规则如下：

* 每个动作只做一次**最小合法编辑**——把某个参数在候选网格上移动一格、切换一次
  电路模板、或替换一个器件变体；
* 所有候选设计点都经
  :meth:`~photon_tuner.core.search_space.DesignSpace.normalize` 与
  :func:`~photon_tuner.core.topology_space.validate_coupling` 校验，
  因此必然可被 SimPhony 求值；
* 动作集**只依赖当前硬件与搜索空间**，与决策策略无关；不同的搜索策略之间的差异
  只来自**决策规则**，而非「谁能到达哪里」。

## 三组动作（对应本项目分层的 architecture / topology / device）

* ``architecture`` —— 分块/cores/dataflow/精度/缓存/带宽/crossbar 几何/波长/时钟
  等参数在候选网格上 ±1 步进；
* ``topology``    —— 切换电路模板（``core_kind``↔``node_kind`` 原子耦合切换）以及
  sharing / temporal_accum / laser_distribution / mesh_rank ±1 步进；
* ``device``      —— 实例级开关（片上/片外激光、波导/MMI 分束）与器件库变体替换。

参数是否进入动作集由**搜索空间规格**决定（``configs/search_space.json``，唯一数据源），
并按当前 ``core_kind`` 所属拓扑族进一步收窄——``mesh_rank`` 在 crossbar 下无意义、
``core_height`` / ``num_wavelength`` / ``dataflow`` 在 mesh 与权重阵列下不生效
（见 ``photon_tuner.core.topology_space.PARAM_SCOPE``）。这样动作集与「真正影响
仿真结果的参数」保持一致，不会把评估预算烧在无效维度上。

## 主要接口

* :class:`Action`        —— 纯数据的动作描述（``group`` / ``label`` / ``payload``）；
* :class:`ActionSpace`   —— 由硬件、搜索空间与器件库构造，提供
  :meth:`ActionSpace.neighbors`（全部合法邻域）与
  :meth:`ActionSpace.random_neighbor`（均匀随机采样）；
* :func:`apply_action`   —— 按 ``payload`` 在当前硬件上重放一次动作。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from photon_tuner.core.search_space import (
    DesignSpace,
    materialize_hardware,
    set_device_choice,
    switch_topology_template,
)
from photon_tuner.core.topology_space import (
    WEIGHT_ARRAY_CORE_KINDS,
    evaluable_template_names,
    validate_coupling,
)

ARCH_LAYER = "architecture"
TOPOLOGY_LAYER = "topology"
DEVICE_LAYER = "device"

# core 族 -> 器件表硬件路径（对应 analytic.py 的 build_sub_arch /
# build_mesh_sub_arch / build_weight_array_sub_arch 所消费的器件表）。
_DEVICE_TABLE_PATHS = {
    "crossbar": ("core", "node"),
    "mesh": ("mesh.core", "mesh.node"),
    "weight_array": ("weight_array.core", "weight_array.node"),
}

# set_device_choice 在核心/节点表上使用的 level 名映射。
_LEVEL_FOR_SET = {
    "core": "core",
    "node": "node",
    "mesh.core": "mesh_core",
    "mesh.node": "mesh_node",
}


def canonical_fingerprint(hardware: Mapping[str, Any]) -> str:
    """确定性、可比较的硬件指纹（sha256 前 8 字节）。"""
    payload = json.dumps(dict(hardware), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _core_family(core_kind: str) -> str:
    if core_kind.startswith("mzi_mesh_"):
        return "mesh"
    if core_kind in WEIGHT_ARRAY_CORE_KINDS:
        return "weight_array"
    return "crossbar"


@dataclass(frozen=True)
class Action:
    """一次最小合法硬件编辑。``payload`` 是纯数据（可 JSON 序列化），由
    :func:`apply_action` 统一解释，保证动作可复现、可审计、可日志化。"""

    group: str
    label: str
    payload: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"group": self.group, "label": self.label, "payload": self.payload}


def _device_paths(core_kind: str) -> Tuple[str, str]:
    table = _DEVICE_TABLE_PATHS[_core_family(core_kind)]
    return f"devices.{table[0]}", f"devices.{table[1]}"


class ActionSpace:
    """从「当前硬件 + 搜索空间 + 器件库」生成合法邻域动作。"""

    def __init__(
        self,
        space: DesignSpace,
        device_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        mode: str = "real",
        include_weight_array_devices: bool = False,
    ) -> None:
        self.space = space
        self.catalog = device_catalog or (space.device_catalog or {})
        self.mode = mode
        # 是否把 ``devices.weight_array`` 表纳入器件变体替换。默认 False，此时只覆盖
        # core/node 与 mesh_core/mesh_node 两张表。
        self.include_weight_array_devices = include_weight_array_devices
        self._cache: Dict[str, List[Tuple[Action, Dict[str, Any]]]] = {}
        self._architecture_params = [
            (layer, name, spec)
            for layer, name, spec in space.params()
            if layer == ARCH_LAYER
        ]
        self._topology_params = [
            (layer, name, spec)
            for layer, name, spec in space.params()
            if layer == TOPOLOGY_LAYER
        ]
        self._templates = evaluable_template_names(mode)

    # -- 观察 ---------------------------------------------------------------- #

    def mode_summary(self) -> Dict[str, Any]:
        """返回动作空间摘要（供日志 / 审计）。"""
        return {
            "mode": self.mode,
            "architecture_params": [name for _, name, _ in self._architecture_params],
            "topology_params": [name for _, name, _ in self._topology_params],
            "templates": self._templates,
            "include_weight_array_devices": self.include_weight_array_devices,
        }

    # -- 邻域 ---------------------------------------------------------------- #

    def neighbors(self, hardware: Mapping[str, Any]) -> List[Tuple[Action, Dict[str, Any]]]:
        """返回 (动作, 结果硬件) 列表。结果已归一化且通过耦合校验，且**一定改变**
        硬件（去掉了空动作）。"""
        hw = self.space.normalize(materialize_hardware(hardware))
        key = canonical_fingerprint(hw)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        base_fp = canonical_fingerprint(hw)
        seen: Dict[str, Tuple[Action, Dict[str, Any]]] = {}

        # 1) architecture 层：网格 ±1 步进。
        for _layer, param, _spec in self._architecture_params:
            for direction in (1, -1):
                cand = self.space.step(hw, param, direction)
                self._accept(seen, base_fp, Action(ARCH_LAYER, f"arch:{param}{'+' if direction > 0 else '-'}",
                                                  {"op": "step", "param": param, "direction": direction}), cand)

        # 2) topology 层：模板切换（原子耦合）+ 其余拓扑参数 ±1。
        current_core = str(self.space.get(hw, "core_kind") or "dptc_crossbar")
        for core_kind in self._templates:
            if core_kind == current_core:
                continue
            try:
                cand = self.space.normalize(switch_topology_template(hw, core_kind))
            except ValueError:
                continue
            self._accept(seen, base_fp, Action(TOPOLOGY_LAYER, f"topo:switch:{core_kind}",
                                                {"op": "template", "core_kind": core_kind}), cand)

        for _layer, param, _spec in self._topology_params:
            if param in ("core_kind", "node_kind"):
                # node_kind 不应脱离 core_kind 单独步进（会破坏耦合），
                # core_kind 已由上面的模板切换覆盖。
                continue
            for direction in (1, -1):
                cand = self.space.step(hw, param, direction)
                self._accept(seen, base_fp, Action(TOPOLOGY_LAYER, f"topo:{param}{'+' if direction > 0 else '-'}",
                                                   {"op": "step", "param": param, "direction": direction}), cand)

        # 3) device 层：实例级开关 + 器件库变体替换。
        for variant_action, cand in self._device_actions(hw, base_fp):
            self._accept(seen, base_fp, variant_action, cand)

        result = sorted(seen.values(), key=lambda pair: (pair[0].group, pair[0].label))
        self._cache[key] = result
        return result

    def random_neighbor(
        self, hardware: Mapping[str, Any], rng: Any
    ) -> Optional[Tuple[Action, Dict[str, Any]]]:
        """均匀随机抽取一个合法邻域动作。"""
        candidates = self.neighbors(hardware)
        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]

    # -- 内部 ---------------------------------------------------------------- #

    def _accept(self, seen, base_fp, action: Action, cand: Dict[str, Any]) -> None:
        if validate_coupling(cand, self.space, self.mode):
            return
        if canonical_fingerprint(cand) == base_fp:
            return
        seen.setdefault(canonical_fingerprint(cand), (action, cand))

    def _device_actions(self, hw: Dict[str, Any], base_fp: str):
        """产生 (激光/分束开关, 器件变体替换) 两类动作。"""
        # 3a) 实例级开关（激光在片/离片、波导/MMI 分束）。这些是 device 层的搜索参数。
        for _layer, param, _spec in self.space.params():
            if _layer != DEVICE_LAYER:
                continue
            for direction in (1, -1):
                cand = self.space.step(hw, param, direction)
                yield (Action(DEVICE_LAYER, f"device:{param}{'+' if direction > 0 else '-'}",
                              {"op": "step", "param": param, "direction": direction}), cand)

        # 3b) 器件库变体替换，仅当前 core 族实际消费的表。
        core_kind = str(self.space.get(hw, "core_kind") or "dptc_crossbar")
        for hardware_path in _DEVICE_TABLE_PATHS[_core_family(core_kind)]:
            if hardware_path.startswith("weight_array") and not self.include_weight_array_devices:
                continue
            table = hw.get("devices", {})
            node = table
            for part in hardware_path.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if not isinstance(node, dict):
                continue
            for dtype, current_variant in node.items():
                if not isinstance(dtype, str) or not isinstance(current_variant, str):
                    continue
                variants = sorted(self.catalog.get(dtype, {}).keys())
                if not variants or current_variant not in variants:
                    continue
                for name in variants:
                    if name == current_variant:
                        continue
                    level = _LEVEL_FOR_SET.get(hardware_path)
                    if level is not None:
                        try:
                            cand = set_device_choice(hw, level, dtype, name)
                        except ValueError:
                            continue
                    else:
                        cand = _set_weight_array_device(hw, hardware_path, dtype, name)
                    cand = self.space.normalize(cand)
                    yield (Action(DEVICE_LAYER, f"device:{hardware_path}.{dtype}->{name}",
                                  {"op": "device", "level": hardware_path, "device_type": dtype,
                                   "device_name": name}), cand)

def _set_weight_array_device(
    hardware: Mapping[str, Any], hardware_path: str, device_type: str, device_name: str
) -> Dict[str, Any]:
    """为 ``devices.weight_array.core/node`` 表设置器件变体。"""
    result = materialize_hardware(hardware)
    node = result.setdefault("devices", {})
    for part in hardware_path.split("."):
        node = node.setdefault(part, {})
    node[device_type] = device_name
    return result


def apply_action(hardware: Mapping[str, Any], action: Action, space: DesignSpace) -> Dict[str, Any]:
    """解释 :class:`Action.payload` 并在当前硬件上应用一次动作（返回归一化结果）。"""
    hw = materialize_hardware(hardware)
    op = action.payload.get("op")
    if op == "step":
        return space.step(hw, action.payload["param"], int(action.payload["direction"]))
    if op == "template":
        return space.normalize(switch_topology_template(hw, action.payload["core_kind"]))
    if op == "device":
        level = action.payload["level"]
        dtype = action.payload["device_type"]
        name = action.payload["device_name"]
        set_level = _LEVEL_FOR_SET.get(level)
        if set_level is not None:
            return space.normalize(set_device_choice(hw, set_level, dtype, name))
        return space.normalize(_set_weight_array_device(hw, level, dtype, name))
    raise ValueError(f"unknown action op: {op!r}")