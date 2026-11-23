"""可加载工作负载层（analytic 模式下的真实模型）的测试。"""

import json
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
        # 回归锚点固定用 SI 镜像：规范单位的 energy_mj 是该值的 1e-9 倍，
        # 用 SI 值可以钉住原始物理量而不受展示单位影响。
        # 锚点随架构搜索空间改为 2 的幂网格而更新（基线 16x16x16 @4 GHz / 512 KB）。
        # 旧锚点 1.1556e8 pJ，新锚点 9.8991e7 pJ：同一量级、同向可解释
        # （波长 16→16 不变、频率 5→4 GHz 与缓存 485→512 KB 共同作用）。
        # 回归锚点钉在**计算能耗**上：它是 SimPhony 的物理输出，不随"是否把
        # 存储能耗计入目标"这一建模选择而变。总能耗另由下面的分量关系约束。
        self.assertAlmostEqual(
            report["si_metrics"]["energy_compute_pj"], 98990574.31838855, places=3
        )
        # 总能耗必须等于计算 + 存储（口径自洽，且存储分量必须被真正计入）。
        si = report["si_metrics"]
        self.assertAlmostEqual(
            si["energy_pj"], si["energy_compute_pj"] + si["energy_memory_pj"], places=3
        )
        self.assertGreater(si["energy_memory_pj"], 0.0)

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
        ea = evaluate(narrow, ROOT, workload_name="vit_base")["metrics"]["energy_mj"]
        eb = evaluate(wide, ROOT, workload_name="vit_base")["metrics"]["energy_mj"]
        self.assertNotAlmostEqual(ea, eb, places=3)

    def test_area_is_workload_independent(self):
        """面积是硬件属性；只有能量/延迟会变化。"""
        tiny = evaluate(self.hardware, ROOT, workload_name="vit_tiny")
        small = evaluate(self.hardware, ROOT, workload_name="vit_small")
        self.assertAlmostEqual(
            tiny["metrics"]["area_mm2"], small["metrics"]["area_mm2"], places=6
        )

    def test_per_layer_energy_sums_to_total(self):
        """原生逐层分解必须无损。"""
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        by_layer = sub_arch(report["energy"]["by_layer"])
        self.assertEqual(
            set(by_layer), set(sub_arch(report["workload"]["partition_cycles"]))
        )
        # by_layer 只覆盖**计算**能耗（存储能耗按层级报告在 memory.energy），
        # 因此应与 si_metrics 的计算分量比较，而不是含存储的总量。
        self.assertAlmostEqual(
            sum(by_layer.values()), report["si_metrics"]["energy_compute_pj"], places=3
        )

    def test_per_layer_energy_is_not_a_cycles_estimate(self):
        """P1：原生能量必须携带周期占比所不能承载的信息。"""
        report = evaluate(self.hardware, ROOT, workload_name="vit_base")
        by_layer = sub_arch(report["energy"]["by_layer"])
        cycles = sub_arch(report["workload"]["partition_cycles"])
        switching = {name: c[0] * c[1] * c[2] for name, c in cycles.items()}
        total_switching = sum(switching.values())
        total_energy = report["metrics"]["energy_mj"]

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
            # 取全部层：单位一致性断言要求逐层求和，截断到 top-5 会漏掉一部分。
            ranks = AnalysisToolbox(search_space={}).run(
                "summarize_layer_ranks", context, {"top_layers": 10**6}
            ).payload["layers"]
            # 在临时目录销毁前读取报告原始值，供下面的单位一致性断言使用。
            report = json.loads(
                Path(evaluation.detail_files["report"]).read_text(encoding="utf-8")
            )
            native_pj = sum(
                sum(v.values()) if isinstance(v, dict) else float(v)
                for v in report["energy"]["by_layer"].values()
            )
            total_mj = evaluation.metrics["energy_mj"]

        self.assertTrue(ranks)
        for layer in ranks:
            self.assertEqual(layer["energy_source"], "simulated")
            self.assertGreater(layer["energy_mj"], 0.0)
        energies = [layer["energy_mj"] for layer in ranks]
        self.assertEqual(energies, sorted(energies, reverse=True))

        # 单位一致性回归：``ranks`` 混合了两个来源——SimPhony 原生的
        # ``energy.by_layer``（pJ）与规范单位的目标指标（mJ）。早期版本把 pJ
        # 直接塞进 ``energy_mj``，使排序在 1e9 倍误差下错乱，瓶颈定位全错。
        # 不变量 1：逐层能耗必须等于**报告原生 by_layer 的 pJ 值 ÷ 1e9**。
        # 这是独立重算：它直接检查换算有没有发生，而不是复述被测代码。
        self.assertGreater(native_pj, 0.0)
        self.assertAlmostEqual(sum(energies), native_pj * 1e-9,
                               delta=max(native_pj * 1e-9 * 1e-6, 1e-12))
        # 不变量 2：逐层之和虽只是总能耗的子集（不含存储/静态），但必须在同一
        # 数量级内——把 pJ 直接当 mJ 用时这里会差 9 个数量级。
        self.assertGreater(sum(energies), total_mj * 1e-3)
        self.assertLess(sum(energies), total_mj * 2.0)

        # 逐层延迟必须是 ms 量级（cycles/freq 得到 ns，需乘 1e-6）。
        for layer in ranks:
            self.assertLess(layer["latency_ms"], 1e3, layer)
            self.assertGreater(layer["latency_ms"], 0.0, layer)

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
        energies = [g["energy_mj"] for g in groups]
        self.assertEqual(energies, sorted(energies, reverse=True))

        # 每层恰好被计入一次，且截断被遵守。
        self.assertEqual(sum(g["layer_count"] for g in groups), 25)
        for group in groups:
            self.assertLessEqual(len(group["top_layers"]), 2)
            self.assertGreaterEqual(group["layer_count"], 1)


if __name__ == "__main__":
    unittest.main()


class TestAccountingConsistency(unittest.TestCase):
    """口径自洽：能耗（计算+存储）与面积（逐器件+节点边界框）必须能对上账。

    这组断言针对的是**跨模块接口**：单看任一模块都自洽，错配只在汇总处显现。
    历史上有两处真实缺陷正是这样漏掉的：
      * ``energy`` 只计计算能耗，而 ``latency`` 含存储延时（存储占 75–90%）；
      * 厂商 ``simu_area`` 的 ``if "node" not in v`` 判断的是字典内部键而非键名，
        导致节点面积被重复计入（真实模式多算 43%）。
    """

    def _report(self):
        import tempfile

        from photon_tuner.simulators.simphony import SimphonyAdapter

        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root="photon_tuner/simulators/simphony",
                output_root=Path(tmp),
                mode="analytic",
                workload_name="vit_tiny",
            )
            evaluation = adapter.evaluate(default_hardware(), 0)
            report = json.loads(
                Path(evaluation.detail_files["report"]).read_text(encoding="utf-8")
            )
        return evaluation.metrics, report

    def test_energy_splits_into_compute_and_memory(self):
        metrics, report = self._report()
        self.assertIn("energy_compute_mj", metrics)
        self.assertIn("energy_memory_mj", metrics)
        total = metrics["energy_mj"]
        self.assertAlmostEqual(
            metrics["energy_compute_mj"] + metrics["energy_memory_mj"], total,
            delta=max(total * 1e-9, 1e-12),
        )
        # 存储能耗必须被真正计入，而不是被静默丢弃。
        self.assertGreater(metrics["energy_memory_mj"], 0.0)
        # SI 镜像与规范单位必须一致。
        si = report["si_metrics"]
        self.assertAlmostEqual(metrics["energy_mj"], si["energy_pj"] * 1e-9,
                               delta=abs(metrics["energy_mj"]) * 1e-9)

    def test_area_total_equals_device_plus_node_bbox(self):
        metrics, report = self._report()
        arch = list(report["area"]["total"])[0]
        composition = report["area"]["composition"][arch]
        self.assertAlmostEqual(
            composition["device_area_um2"] + composition["node_bbox_um2"],
            composition["total_um2"],
            delta=max(composition["total_um2"] * 1e-9, 1e-6),
        )
        self.assertAlmostEqual(
            metrics["area_mm2"], composition["total_um2"] * 1e-6,
            delta=abs(metrics["area_mm2"]) * 1e-9,
        )

    def test_chip_breakdown_sums_to_total(self):
        """按 chip 求和必须等于总量——厂商版本会静默漏掉节点边界框。"""
        _, report = self._report()
        arch = list(report["area"]["by_chip_type"])[0]
        chip_sum = sum(v["total_area"] for v in report["area"]["by_chip_type"][arch].values())
        total = report["area"]["total"][arch]
        self.assertAlmostEqual(chip_sum, total, delta=max(abs(total) * 1e-6, 1e-6))

    def test_latency_splits_into_computation_and_memory(self):
        _, report = self._report()
        arch = list(report["latency"]["total"])[0]

        def leaves(value):
            if isinstance(value, dict):
                return sum(leaves(v) for v in value.values())
            return float(value)

        parts = leaves(report["latency"]["computation"][arch]) + leaves(
            report["latency"]["memory"][arch]
        )
        self.assertAlmostEqual(
            leaves(report["latency"]["total"][arch]), parts,
            delta=max(abs(parts) * 1e-6, 1e-12),
        )


if __name__ == "__main__":
    unittest.main()
