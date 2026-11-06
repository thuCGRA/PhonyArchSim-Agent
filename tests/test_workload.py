"""可加载工作负载层（analytic 模式下的真实模型）的测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._subarch import sub_arch
from photon_tuner.core.search_space import default_hardware
from photon_tuner.common.utils import layer_group
from photon_tuner.core.workload import (
    DEFAULT_MINIBLOCK,
    WorkloadError,
    build_workload,
    cycle_tuple,
    im2col_shapes,
    loading_factor_calculator,
)
from photon_tuner.simulators.analytic import evaluate


ROOT = str(Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony")

#: ViT-base：patch_embed + 12 个 block × 6 个子层。
VIT_BASE_LAYER_COUNT = 1 + 12 * 6
VIT_SUB_LAYERS = ("qkv", "attn.qk", "attn.av", "out_proj", "ffn.fc1", "ffn.fc2")


class TestIm2ColShapes(unittest.TestCase):
    """移植的 reshape 逻辑必须按 SimPhony 顺序返回 (input, weight)。"""

    def test_linear_shapes(self):
        # PyTorch Linear 权重为 (out_features, in_features)。
        reshaped_input, reshaped_weight = im2col_shapes(
            "Linear", weight_shape=[2304, 768], input_shape=[197, 768]
        )
        self.assertEqual(reshaped_weight, (2304, 768))
        self.assertEqual(reshaped_input, (768, 197))

    def test_conv2d_shapes(self):
        reshaped_input, reshaped_weight = im2col_shapes(
            "Conv2d",
            weight_shape=[768, 3, 16, 16],
            input_shape=[1, 3, 224, 224],
            stride=16,
        )
        self.assertEqual(reshaped_weight, (768, 3 * 16 * 16))
        self.assertEqual(reshaped_input, (3 * 16 * 16, 1 * 14 * 14))

    def test_matmul_shapes(self):
        reshaped_input, reshaped_weight = im2col_shapes(
            "MatMul", layer_shape=[[12, 197, 64], [12, 64, 197], [12, 64, 197]]
        )
        self.assertEqual(reshaped_input, (64, 12 * 197))
        self.assertEqual(reshaped_weight, (12 * 197, 64))

    def test_unsupported_op_raises(self):
        with self.assertRaises(WorkloadError):
            im2col_shapes("Bogus")


class TestCycleArithmetic(unittest.TestCase):
    """cycle_tuple 必须复现 SimPhony 的分区算术。"""

    def test_loading_factors(self):
        self.assertEqual(loading_factor_calculator("full", "full", "full"), (1, 1))
        with self.assertRaises(WorkloadError):
            loading_factor_calculator("nonsense", "full", "full")

    def test_cycle_tuple_linear(self):
        matrices = im2col_shapes("Linear", weight_shape=[2304, 768], input_shape=[197, 768])
        cycle = cycle_tuple(
            miniblock=DEFAULT_MINIBLOCK,
            matrices=matrices,
            multi_wavelength=12,
        )
        iter_n, iter_d, iter_m, n, d, m, fw, fx = cycle
        self.assertEqual((n, d, m), (2304, 768, 197))
        # 架构维度：height = 4*2 = 8、width = 4*2 = 8、wavelength = 12
        self.assertEqual(iter_n, -(-2304 // 8))
        self.assertEqual(iter_m, -(-197 // 8))
        self.assertEqual(iter_d, -(-768 // 12))
        self.assertEqual((fw, fx), (1, 1))

    def test_cycle_tuple_rejects_mismatched_dims(self):
        # weight in_features=8，但输入最后一维为 3。
        matrices = im2col_shapes("Linear", weight_shape=[10, 8], input_shape=[4, 3])
        with self.assertRaises(WorkloadError):
            cycle_tuple(miniblock=DEFAULT_MINIBLOCK, matrices=matrices, multi_wavelength=12)


class TestWorkloadBuilders(unittest.TestCase):
    def test_vit_tiny_layer_structure(self):
        workload = build_workload("vit_tiny", {"num_wavelength": 12}, simphony_root=ROOT)
        cycles = workload["cycles"]
        # patch_embed + 4 个 block × 6 个子层。
        self.assertEqual(len(cycles), 1 + 4 * 6)
        self.assertIn("patch_embed", cycles)
        for suffix in VIT_SUB_LAYERS:
            self.assertIn(f"layers.0.{suffix}", cycles)

    def test_vit_base_layer_structure(self):
        workload = build_workload("vit_base", {"num_wavelength": 12}, simphony_root=ROOT)
        cycles = workload["cycles"]
        self.assertEqual(len(cycles), VIT_BASE_LAYER_COUNT)
        self.assertIn("patch_embed", cycles)
        for index in (0, 11):
            for suffix in VIT_SUB_LAYERS:
                self.assertIn(f"layers.{index}.{suffix}", cycles)

    def test_vit_layer_sizes_present_for_every_layer(self):
        workload = build_workload("vit_base", {"num_wavelength": 12}, simphony_root=ROOT)
        self.assertEqual(set(workload["cycles"]), set(workload["layer_sizes"]))
        for name, size in workload["layer_sizes"].items():
            for key in ("input_size", "output_size", "weight_size"):
                self.assertIn(key, size, msg=f"{name} missing {key}")
                self.assertGreater(size[key][0], 0, msg=f"{name} has non-positive {key}")

    def test_cycles_respond_to_wavelength(self):
        """更多波长必须减少 iter_D（D 维度被分块铺开）。"""
        wide = build_workload("vit_base", {"num_wavelength": 8}, simphony_root=ROOT)
        narrow = build_workload("vit_base", {"num_wavelength": 24}, simphony_root=ROOT)
        self.assertGreater(
            wide["cycles"]["layers.0.qkv"][1],
            narrow["cycles"]["layers.0.qkv"][1],
        )

    def test_unknown_workload_raises(self):
        with self.assertRaises(WorkloadError):
            build_workload("does_not_exist", simphony_root=ROOT)

    def test_vit_rejects_indivisible_embed_dim(self):
        import json
        import tempfile

        spec = {
            "default": "bad",
            "workloads": {
                "bad": {"kind": "vit", "embed_dim": 10, "depth": 1, "num_heads": 3}
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workloads.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(WorkloadError):
                build_workload("bad", {"num_wavelength": 12}, config_path=str(path))


class TestLayerGroupRegression(unittest.TestCase):
    """融合 QKV 不得被 attention 规则吞掉。"""

    def test_qkv_projection_is_reachable(self):
        self.assertEqual(layer_group("layers.0.qkv"), "qkv_projection")
        self.assertEqual(layer_group("qkv"), "qkv_projection")

    def test_other_groups(self):
        self.assertEqual(layer_group("layers.0.attn.qk"), "attention")
        self.assertEqual(layer_group("layers.0.attn.av"), "attention")
        self.assertEqual(layer_group("layers.0.out_proj"), "output_projection")
        self.assertEqual(layer_group("layers.0.ffn.fc1"), "ffn")
        self.assertEqual(layer_group("layers.0.ffn.fc2"), "ffn")
        self.assertEqual(layer_group("patch_embed"), "embedding")


class TestRealWorkloadEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = default_hardware()

    def test_default_workload_is_vit_tiny(self):
        report = evaluate(self.hardware, ROOT)
        self.assertEqual(report["workload"]["name"], "vit_tiny")
        # 默认 node 相移器为 MEMS_PS（P_pi 已按 10 nW 级订正）。
        self.assertAlmostEqual(report["metrics"]["energy_pj"], 115562731.63852626, places=3)

    def test_vit_workload_produces_varied_layer_profile(self):
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        self.assertEqual(report["workload"]["name"], "vit_base")
        cycles = sub_arch(report["workload"]["partition_cycles"])
        self.assertEqual(len(cycles), VIT_BASE_LAYER_COUNT)

        switching = {name: cycle[0] * cycle[1] * cycle[2] for name, cycle in cycles.items()}
        # ViT 有 12 个相同的 block，因此只存在少数几种不同的子层原型——但绝不止
        # 单一常量。
        self.assertGreaterEqual(len(set(switching.values())), 5)

        # 这些 block 确实以周期 6 重复。
        for suffix in VIT_SUB_LAYERS:
            self.assertEqual(
                switching[f"layers.0.{suffix}"],
                switching[f"layers.7.{suffix}"],
                msg=f"block periodicity broken for {suffix}",
            )

        top = max(switching, key=switching.get)
        bottom = min(switching, key=switching.get)
        self.assertNotEqual(layer_group(top), "")  # 健全性检查
        self.assertGreater(switching[top], switching[bottom])

    def test_vit_memory_energy_varies_per_layer(self):
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        memory_energy = sub_arch(report["memory"]["energy"])
        self.assertGreater(len(memory_energy), 1)
        self.assertGreaterEqual(len(set(memory_energy.values())), 5)

    def test_vit_operator_groups_are_dominated_by_projections_and_ffn(self):
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        cycles = sub_arch(report["workload"]["partition_cycles"])
        totals = {}
        for name, cycle in cycles.items():
            group = layer_group(name)
            totals[group] = totals.get(group, 0) + cycle[0] * cycle[1] * cycle[2]
        # 权重复用的 GEMM 主导周期预算。
        dominant = max(totals, key=totals.get)
        self.assertIn(dominant, {"attention", "qkv_projection", "ffn"})
        self.assertGreater(totals["ffn"], 0)

    def test_accepts_prebuilt_workload(self):
        prebuilt = build_workload("vit_tiny", {"num_wavelength": 12}, simphony_root=ROOT)
        report = evaluate(self.hardware, ROOT, workload=prebuilt)
        self.assertEqual(report["workload"]["name"], "vit_tiny")
        self.assertEqual(len(sub_arch(report["workload"]["partition_cycles"])), 1 + 4 * 6)

    def test_layer_profile_responds_to_hardware(self):
        """P0 目标：逐层需求画像必须随硬件变化。"""
        narrow = default_hardware()
        narrow["arch"]["optical_core"]["num_wavelength"] = 8
        wide = default_hardware()
        wide["arch"]["optical_core"]["num_wavelength"] = 24

        a = sub_arch(evaluate(narrow, ROOT, workload_name="vit_base")["workload"]["partition_cycles"])
        b = sub_arch(evaluate(wide, ROOT, workload_name="vit_base")["workload"]["partition_cycles"])

        # D 维度被铺到波长轴上：波长越多，同一层的 D 迭代越少。
        self.assertGreater(a["layers.0.qkv"][1], b["layers.0.qkv"][1])
        self.assertNotEqual(a["layers.0.attn.qk"], b["layers.0.attn.qk"])

        # 且整设计能量随之变化。
        ea = evaluate(narrow, ROOT, workload_name="vit_base")["metrics"]["energy_pj"]
        eb = evaluate(wide, ROOT, workload_name="vit_base")["metrics"]["energy_pj"]
        self.assertNotAlmostEqual(ea, eb, places=3)

    def test_area_is_workload_independent(self):
        """面积是硬件属性；只有能量/延迟会变化。"""
        tiny = evaluate(self.hardware, ROOT, workload_name="vit_tiny")
        small = evaluate(self.hardware, ROOT, workload_name="vit_small")
        self.assertAlmostEqual(
            tiny["metrics"]["area_um2"], small["metrics"]["area_um2"], places=6
        )

    def test_per_layer_energy_sums_to_total(self):
        """原生逐层分解必须无损。"""
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        by_layer = sub_arch(report["energy"]["by_layer"])
        self.assertEqual(
            set(by_layer), set(sub_arch(report["workload"]["partition_cycles"]))
        )
        self.assertAlmostEqual(
            sum(by_layer.values()), report["metrics"]["energy_pj"], places=3
        )

    def test_per_layer_energy_is_not_a_cycles_estimate(self):
        """P1：原生能量必须携带周期占比所不能承载的信息。"""
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        by_layer = sub_arch(report["energy"]["by_layer"])
        cycles = sub_arch(report["workload"]["partition_cycles"])
        switching = {name: c[0] * c[1] * c[2] for name, c in cycles.items()}
        total_switching = sum(switching.values())
        total_energy = report["metrics"]["energy_pj"]

        deviations = [
            abs(by_layer[name] - total_energy * switching[name] / total_switching)
            for name in switching
        ]
        self.assertGreater(max(deviations), 1.0)

        # ffn.fc1 与 ffn.fc2 周期数相同，因此按周期比例的估计无法区分二者——原生
        # 分解必须能够区分。
        self.assertNotAlmostEqual(
            by_layer["layers.0.ffn.fc1"], by_layer["layers.0.ffn.fc2"], places=3
        )


class TestLayerRankToolUsesNativeEnergy(unittest.TestCase):
    def test_ranks_report_simulated_energy(self):
        import tempfile

        from photon_tuner.simulators.simphony import SimphonyAdapter
        from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
        from photon_tuner.toolbox.common_tool.context import ToolContext

        hardware = default_hardware()
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root="photon_tuner/simulators/simphony",
                output_root=Path(tmp),
                mode="analytic",
                workload_name="vit_tiny",
            )
            evaluation = adapter.evaluate(hardware, 0)
            context = ToolContext(
                output_dir=Path(tmp),
                evaluation=evaluation,
                current_hardware=hardware,
                search_space={},
            )
            ranks = AnalysisToolbox(search_space={}).run(
                "summarize_layer_ranks", context, {"top_layers": 5}
            ).payload["layers"]

        self.assertTrue(ranks)
        for layer in ranks:
            self.assertEqual(layer["energy_source"], "simulated")
            self.assertGreater(layer["energy_pj"], 0.0)
        energies = [layer["energy_pj"] for layer in ranks]
        self.assertEqual(energies, sorted(energies, reverse=True))

    def test_operator_group_tool_aggregates_layers(self):
        import tempfile

        from photon_tuner.simulators.simphony import SimphonyAdapter
        from photon_tuner.toolbox.analysis_tools import AnalysisToolbox
        from photon_tuner.toolbox.common_tool.context import ToolContext

        hardware = default_hardware()
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root="photon_tuner/simulators/simphony",
                output_root=Path(tmp),
                mode="analytic",
                workload_name="vit_tiny",
            )
            evaluation = adapter.evaluate(hardware, 0)
            context = ToolContext(
                output_dir=Path(tmp),
                evaluation=evaluation,
                current_hardware=hardware,
                search_space={},
            )
            groups = AnalysisToolbox(search_space={}).run(
                "summarize_operator_groups", context, {"top_layers_per_group": 2}
            ).payload["groups"]

        self.assertTrue(groups)
        names = {group["group"] for group in groups}
        for expected in ("ffn", "qkv_projection", "attention", "embedding"):
            self.assertIn(expected, names)

        # 占比已归一化，且列表按能量排序。
        self.assertAlmostEqual(sum(g["energy_share"] for g in groups), 1.0, places=6)
        energies = [g["energy_pj"] for g in groups]
        self.assertEqual(energies, sorted(energies, reverse=True))

        # 每层恰好被计入一次，且截断被遵守。
        self.assertEqual(sum(g["layer_count"] for g in groups), 25)
        for group in groups:
            self.assertLessEqual(len(group["top_layers"]), 2)
            self.assertGreaterEqual(group["layer_count"], 1)


if __name__ == "__main__":
    unittest.main()
