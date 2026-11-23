"""统计分层搜索空间的基数（论文 Table III 的可复现来源）。

口径（按论文 §B 的定义，逐层相乘）：

  L1 architecture : 14 个参数候选数的乘积；再按拓扑族（PARAM_SCOPE）把
                    「该族下不生效」的参数降为 1（superseded / fixed /
                    unsupported / not_applicable 一律不计为自由度）。
  L2 topology     : 合法 (core_kind, node_kind) 对数 × sharing × temporal ×
                    laser_distribution × mesh_rank；crossbar 族无 mesh_rank。
  L3 device       : 该范式 REQUIRED_DEVICE_SLOTS 里每个槽位的可搜索变体数之积，
                    × laser_source(2) × splitter_kind(2)。
  L4 device_param : 固定（库内物理参数不搜索），恒为 1。

用法：
    PYTHONPATH=. .venv-agent/bin/python scripts/analysis/space_cardinality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photon_tuner.common.paths import PROJECT_ROOT  # noqa: E402
from photon_tuner.core.search_space import (  # noqa: E402
    LAYERED_DESIGN_SPACE,
    DesignSpace,
    discover_device_catalog,
)
from photon_tuner.core.topology_space import (  # noqa: E402
    COUPLING_MATRIX,
    CORE_KINDS,
    PARAM_SCOPE,
    REQUIRED_DEVICE_SLOTS,
    TOPOLOGY_FAMILIES,
)

DEVICE_ROOT = PROJECT_ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"


def _prod(values):
    out = 1
    for value in values:
        out *= value
    return out


def main() -> int:
    space = DesignSpace(spec=LAYERED_DESIGN_SPACE)
    arch = {
        param: len(space.candidates(param))
        for layer, param, _spec in space.params()
        if layer == "architecture"
    }
    arch_total = _prod(arch.values())

    print("== L1 architecture ==")
    print(f"axes={len(arch)}  cartesian={arch_total:.4g}")
    family_arch = {}
    for family in ("crossbar_family", "mesh_family", "weight_array"):
        sizes = {}
        for param, count in arch.items():
            status = (
                PARAM_SCOPE.get(param, {}).get("families", {}).get(family, {}).get("status", "effective")
            )
            sizes[param] = count if status == "effective" else 1
        family_arch[family] = _prod(sizes.values())
        dead = [p for p, n in sizes.items() if n == 1 and arch[p] != 1]
        print(f"  {family:16s} = {family_arch[family]:.4g}  (inactive: {dead})")

    print("\n== L2 topology ==")
    pairs = sum(1 for core in CORE_KINDS for _ in COUPLING_MATRIX[core]["node_kinds"])
    knobs = _prod(
        len(space.candidates(p)) for p in ("sharing_policy", "temporal_accum", "laser_distribution")
    )
    ranks = len(space.candidates("mesh_rank"))
    topo_applicable = 0
    for core in CORE_KINDS:
        family = TOPOLOGY_FAMILIES[core]
        rank_axes = 1 if family == "crossbar_family" else ranks
        topo_applicable += len(COUPLING_MATRIX[core]["node_kinds"]) * knobs * rank_axes
    print(f"legal (core_kind,node_kind) pairs={pairs}  knobs={knobs}  mesh_rank={ranks}")
    print(f"uncoupled={len(CORE_KINDS) * len(space.candidates('node_kind')) * knobs * ranks}"
          f"  nominal={pairs * knobs * ranks}  applicable={topo_applicable}")

    print("\n== L3 device ==")
    catalog = discover_device_catalog(str(DEVICE_ROOT))
    print(f"types={len(catalog)}  searchable_variants={sum(len(v) for v in catalog.values())}")
    per_core = {}
    for core in CORE_KINDS:
        node = COUPLING_MATRIX[core]["node_kinds"][0]
        slots = REQUIRED_DEVICE_SLOTS[core][node]
        missing = [s for s in slots if not catalog.get(s)]
        size = _prod(max(1, len(catalog.get(s, {}))) for s in slots) * 2 * 2
        per_core[core] = (size, len(slots), missing)
        print(f"  {core:22s} family={TOPOLOGY_FAMILIES[core]:15s} slots={len(slots):2d} size={size:.4g}"
              f"{'  MISSING=' + str(missing) if missing else ''}")

    print("\n== joint (paradigm-conditioned) ==")
    joint = 0.0
    family_topo = {"crossbar_family": knobs, "mesh_family": knobs * ranks, "weight_array": knobs * ranks}
    for core in CORE_KINDS:
        family = TOPOLOGY_FAMILIES[core]
        size = family_arch[family] * family_topo[family] * per_core[core][0]
        joint += size
        print(f"  {core:22s} {size:.4g}")
    uncoupled = (
        arch_total
        * len(CORE_KINDS)
        * len(space.candidates("node_kind"))
        * knobs
        * ranks
        * _prod(len(v) for v in catalog.values())
    )
    print(f"joint-conditioned = {joint:.4g}")
    print(f"joint-uncoupled   = {uncoupled:.4g}")
    print(f"reduction         = {uncoupled / joint:.4g}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
