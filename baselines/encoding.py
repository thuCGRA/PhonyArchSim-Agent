"""硬件设计点的向量编码 —— 供贝叶斯优化的高斯过程代理模型使用。

编码策略：

* **数值型参数**（tiles / cores / 几何 / 波长 / 时钟 / 缓存 / 带宽 / 精度 / mesh 阶数）
  编码为其在候选网格中的**归一化序号**，落在 ``[0, 1]``。网格是单调有序的，因此
  相邻网格点映射为相邻标量，RBF/Matérn 核的平滑性假设与之相符。
* **类别型参数**（dataflow / core_kind / node_kind / sharing_policy /
  temporal_accum / laser_distribution / 器件变体 / 实例开关）编码为 **one-hot**。

由于三种电路族（crossbar / mesh / weight_array）消费不同的器件表，编码为**全部三族
都预留 slot**、非当前族的 slot 全零。这样特征维度恒定，跨模板的设计点仍可在同一
代理模型中比较；被置零的槽位不引入噪声（它们对当前模板无因果作用）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from photon_tuner.core.search_space import DesignSpace, default_hardware

# 器件表路径（与 analytic.py 的三族消费一致）。
DEVICE_TABLE_PATHS: Tuple[str, ...] = (
    "core",
    "node",
    "mesh.core",
    "mesh.node",
    "weight_array.core",
    "weight_array.node",
)

# 三族之外的 device 层实例开关参数名（DesignSpace 的 "device" 层）。
INSTANCE_SWITCH_PARAMS: Tuple[str, ...] = ("laser_source", "splitter_kind")


def _enum_slot(name: str, candidates: Sequence[Any]) -> Dict[str, Any]:
    return {"name": name, "kind": "onehot", "candidates": [str(c) for c in candidates]}


def _numeric_slot(name: str, candidates: Sequence[Any]) -> Dict[str, Any]:
    ordered = sorted({float(c) for c in candidates})
    return {"name": name, "kind": "ordinal", "candidates": ordered}


class HardwareEncoder:
    """把硬件字典编码为定长浮点向量。"""

    def __init__(
        self,
        space: DesignSpace,
        catalog: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.space = space
        self.catalog: Mapping[str, Mapping[str, Any]] = catalog or (space.device_catalog or {})
        self._slots: List[Dict[str, Any]] = []
        self._build_layout()
        self.dim = self._compute_dim()

    # -- 布局 ---------------------------------------------------------------- #

    def _build_layout(self) -> None:
        # 1) architecture 层与 2) topology 层：按搜索空间声明的顺序（确定性）。
        for layer, name, spec in self.space.params():
            if layer not in ("architecture", "topology"):
                continue
            candidates = spec.get("values") or []
            if not candidates:
                continue
            if spec.get("kind") == "enum":
                self._slots.append(_enum_slot(name, candidates))
            else:
                self._slots.append(_numeric_slot(name, candidates))

        # 3) device 层的实例开关。
        for name in INSTANCE_SWITCH_PARAMS:
            try:
                candidates = self.space.candidates(name)
            except KeyError:
                continue
            if candidates:
                self._slots.append(_enum_slot(name, candidates))

        # 4) 三族器件表的变体槽位。布局从 default_hardware() 派生（与当前设计点无关），
        #    保证特征维度恒定。
        base = default_hardware().get("devices", {}) or {}
        for path in DEVICE_TABLE_PATHS:
            node: Any = base
            for part in path.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if not isinstance(node, dict):
                continue
            for dtype in sorted(node):
                if dtype in INSTANCE_SWITCH_PARAMS:
                    continue
                variants = sorted((self.catalog.get(dtype) or {}).keys())
                if not variants:
                    continue
                self._slots.append(
                    {"name": f"{path}.{dtype}", "kind": "onehot",
                     "candidates": variants, "table": path, "device_type": dtype}
                )

    def _compute_dim(self) -> int:
        total = 0
        for slot in self._slots:
            total += len(slot["candidates"]) if slot["kind"] == "onehot" else 1
        return total

    # -- 编码 ---------------------------------------------------------------- #

    def encode(self, hardware: Mapping[str, Any]) -> List[float]:
        vector: List[float] = []
        for slot in self._slots:
            if slot["kind"] == "ordinal":
                vector.append(self._encode_ordinal(slot, hardware))
            else:
                vector.extend(self._encode_onehot(slot, hardware))
        return vector

    def encode_matrix(self, hardware_list: Sequence[Mapping[str, Any]]) -> List[List[float]]:
        return [self.encode(hw) for hw in hardware_list]

    # -- 内部 ---------------------------------------------------------------- #

    def _encode_ordinal(self, slot: Dict[str, Any], hardware: Mapping[str, Any]) -> float:
        ordered: List[float] = list(slot["candidates"])
        if len(ordered) <= 1:
            return 0.0
        try:
            value = float(self.space.get(dict(hardware), slot["name"]))
        except (KeyError, TypeError, ValueError):
            return 0.0
        # 最近网格点 → 归一化序号。
        index = min(range(len(ordered)), key=lambda i: abs(ordered[i] - value))
        return index / (len(ordered) - 1)

    def _encode_onehot(self, slot: Dict[str, Any], hardware: Mapping[str, Any]) -> List[float]:
        candidates: List[str] = list(slot["candidates"])
        current = self._current_categorical(slot, hardware)
        return [1.0 if str(current) == candidate else 0.0 for candidate in candidates]

    def _current_categorical(self, slot: Dict[str, Any], hardware: Mapping[str, Any]) -> Any:
        table = slot.get("table")
        if table is not None:
            node: Any = hardware.get("devices", {})
            for part in table.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                value = node.get(slot["device_type"])
                return None if value is None else str(value)
            return None
        try:
            return self.space.get(dict(hardware), slot["name"])
        except (KeyError, TypeError):
            return None

    # -- 诊断 ---------------------------------------------------------------- #

    def describe(self) -> List[Dict[str, Any]]:
        """返回布局描述（供报告 / 复现说明）。"""
        out: List[Dict[str, Any]] = []
        offset = 0
        for slot in self._slots:
            width = len(slot["candidates"]) if slot["kind"] == "onehot" else 1
            out.append({"name": slot["name"], "kind": slot["kind"],
                        "width": width, "offset": offset})
            offset += width
        return out