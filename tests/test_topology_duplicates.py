"""拓扑注册表的重复性守卫。

背景：``tempo_crossbar`` 曾作为第 8 个 ``core_kind`` 注册，但它复用
``dptc_crossbar`` 的 crossbar 网表，只改 ``name``/``type``/``ref``。实测网表与
energy/area/latency **逐位相同**，于是搜索空间里多出 42% 的精确副本——agent 会把
同一份评估结果当成两种不同结构，浪费预算并污染 Pareto 前沿。

这类重复是**静默**的：注册表看起来合法，``validate_coupling`` 通过，求值不报错，
只有逐位比对才暴露。因此用机械校验代替人眼审查。

注意这里比对的是**剥掉装饰字段后的结构**：``name``/``type``/``ref``/``note`` 改了
不影响任何数值，所以两个仅在这些字段上不同的 core_kind 就是重复。
"""

import itertools
import unittest
from pathlib import Path

from photon_tuner.core.search_space import materialize_hardware, switch_topology_template
from photon_tuner.core.topology_space import (
    COUPLING_MATRIX,
    CORE_KINDS,
    CORE_KIND_SEARCH_ORDER,
    NODE_KINDS,
    REQUIRED_DEVICE_SLOTS,
    TOPOLOGY_FAMILIES,
)
from photon_tuner.simulators.netlist import (
    _tunable_mzi_node,
    _weight_element_node,
    build_mesh_sub_arch,
    build_sub_arch,
    build_weight_array_sub_arch,
    load_device_catalog,
)

ROOT = str(Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony")
CATALOG = load_device_catalog(f"{ROOT}/configs/devices")

#: 纯装饰字段——改动它们不影响 energy / area / latency。
DECORATIVE = frozenset(
    {"name", "type", "ref", "note", "version", "universal", "coherent", "layout"}
)


def _strip(obj):
    """递归剥掉装饰字段，得到"影响数值的部分"。"""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in sorted(obj.items()) if k not in DECORATIVE}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    return obj


def _diff(a, b, path="", out=None):
    """返回两个结构的差异路径列表（空表示完全相同）。"""
    if out is None:
        out = []
    if type(a) is not type(b):
        out.append(path)
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}.len")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
        return out
    if a != b:
        out.append(path)
    return out


def _build_core(kind):
    hw = switch_topology_template(materialize_hardware(None), kind)
    if kind.startswith("mzi_mesh_"):
        return build_mesh_sub_arch(hw, ROOT, CATALOG, kind)
    if kind in ("mrr_weight_bank", "pcm_crossbar", "wdm_link"):
        return build_weight_array_sub_arch(hw, ROOT, CATALOG, kind)
    return build_sub_arch(hw, ROOT, CATALOG)


class TestNoDuplicateCoreKinds(unittest.TestCase):
    """任意两个 core_kind 都必须在物理结构上不同。"""

    def test_core_kind_netlists_are_pairwise_distinct(self):
        built = {kind: _strip(_build_core(kind)) for kind in CORE_KINDS}
        duplicates = []
        for a, b in itertools.combinations(sorted(built), 2):
            if not _diff(built[a], built[b]):
                duplicates.append((a, b))
        self.assertEqual(
            duplicates,
            [],
            "以下 core_kind 剥掉 name/type/ref/note 后网表完全相同，属于重复注册，"
            f"应删除其一或让其物理结构真正不同：{duplicates}",
        )

    def test_core_kind_is_not_a_single_member_dominated_set(self):
        """每种 core_kind 都必须有独占的 (core, node) 耦合项。"""
        for kind in CORE_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, COUPLING_MATRIX)
                self.assertTrue(COUPLING_MATRIX[kind]["node_kinds"], kind)

    def test_registries_cover_exactly_the_same_keys(self):
        """五个注册表必须同集合，否则新增/删除拓扑会留下悬挂条目。"""
        expected = set(CORE_KINDS)
        for name, table in (
            ("COUPLING_MATRIX", COUPLING_MATRIX),
            ("REQUIRED_DEVICE_SLOTS", REQUIRED_DEVICE_SLOTS),
            ("TOPOLOGY_FAMILIES", TOPOLOGY_FAMILIES),
            ("CORE_KIND_SEARCH_ORDER", CORE_KIND_SEARCH_ORDER),
        ):
            with self.subTest(table=name):
                self.assertEqual(set(table), expected, f"{name} 与 CORE_KINDS 不一致")

    def test_search_order_is_a_permutation_of_core_kinds(self):
        self.assertEqual(sorted(CORE_KIND_SEARCH_ORDER), sorted(CORE_KINDS))
        self.assertEqual(len(CORE_KIND_SEARCH_ORDER), len(set(CORE_KIND_SEARCH_ORDER)))


class TestNoDuplicateNodeKinds(unittest.TestCase):
    """node_kind 允许共享网表骨架，但必须实例化不同器件。"""

    NODE_BUILDERS = {
        "tunable_mzi": lambda: _tunable_mzi_node(4, "LT_DC_2x2", "IHO_PS"),
        "weight_mrr": lambda: _weight_element_node("weight_mrr", "WG_MRR_WEIGHT", 16),
        "pcm_cell": lambda: _weight_element_node("pcm_cell", "PCM_CELL", 16),
        "wdm_modulator": lambda: _weight_element_node("wdm_modulator", "MZM_1", 16),
    }

    def test_node_kinds_instantiate_distinct_devices(self):
        """权重阵列三类节点骨架相同，但必须是不同物理器件。"""
        devices = {}
        for name, build in self.NODE_BUILDERS.items():
            node = build()
            devices[name] = frozenset(
                d for group in node.get("devices", {}).values() for d in group
            )

        self.assertEqual(
            sorted(devices), sorted(k for k in NODE_KINDS if k != "dode")
        )
        for a, b in itertools.combinations(sorted(devices), 2):
            with self.subTest(a=a, b=b):
                self.assertNotEqual(
                    devices[a],
                    devices[b],
                    f"{a} 与 {b} 实例化同一组器件，数值必然相同",
                )


class TestTempoPhysicsIsStillExpressible(unittest.TestCase):
    """删除 tempo_crossbar 不应丢失任何可表达能力。"""

    def test_temporal_accum_axis_available_on_crossbar(self):
        from photon_tuner.core.search_space import build_design_space

        ds = build_design_space()
        hw = switch_topology_template(materialize_hardware(None), "dptc_crossbar")
        cands = ds.effective_candidates("temporal_accum", hw)
        self.assertIn("default", cands)
        self.assertIn("deep", cands)

    def test_slow_light_modulator_is_a_device_choice(self):
        from photon_tuner.core.search_space import set_device_choice
        from photon_tuner.simulators.analytic import evaluate

        hw = switch_topology_template(materialize_hardware(None), "dptc_crossbar")
        lit = evaluate(hw, ROOT)["metrics"]
        slow = set_device_choice(hw, "core", "mzm", "SL_MZM")
        sl = evaluate(slow, ROOT)["metrics"]
        # 慢光 EOM 通过 device 维度选型即可表达，无需新拓扑。
        self.assertNotEqual(lit["energy_pj"], sl["energy_pj"])
