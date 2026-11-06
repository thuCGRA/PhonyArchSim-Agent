"""校验 real 模式的硬件注入桥。

``_inject_hardware`` 用搜索到的 ``hardware`` 字典替换 ``ONNArchSimulator`` 解析出的
固定 YAML 子架构。这些测试用仿真器结构的最小 mock（``sub_arch_dictionary`` /
``sub_arch_to_layer_mapping`` / ``arch_blocks`` / ``layer_to_sub_arch_mapping``）来
验证该逻辑，无需导入真实 SimPhony 栈。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._subarch import sub_arch
from photon_tuner.core.search_space import default_hardware
from photon_tuner.simulators.simphony import _inject_hardware

ROOT = Path(__file__).resolve().parents[1] / "photon_tuner" / "simulators" / "simphony"


class MockSimulator:
    """``_inject_hardware`` 所访问属性的最小替身。"""

    def __init__(self):
        self.sub_arch_dictionary = {
            "LT": {"tiles": 1, "cores_per_tile": 1, "core": {"height": 8, "width": 8}},
        }
        self.sub_arch_to_layer_mapping = {"LT": ["layer1", "layer2"]}
        self.arch_blocks = {"layer1": [1, 1, 8, 8], "layer2": [1, 1, 8, 8]}
        self.layer_to_sub_arch_mapping = {"layer1": None, "layer2": None}


class TestRealHardwareInjection(unittest.TestCase):
    def test_inject_hardware_updates_all_structures(self):
        sim = MockSimulator()
        _inject_hardware(sim, default_hardware(), str(ROOT))

        sub = sub_arch(sim.sub_arch_dictionary)
        self.assertEqual(sub["tiles"], 4)
        self.assertEqual(sub["cores_per_tile"], 2)
        self.assertEqual(sub["core"]["height"], 12)
        self.assertEqual(sub["core"]["width"], 12)

        # 逐层 miniblock 遵循注入的几何（[tiles, cores_per_tile, height, width]）。
        self.assertEqual(sim.arch_blocks["layer1"], [4, 2, 12, 12])
        self.assertEqual(sim.arch_blocks["layer2"], [4, 2, 12, 12])

        # layer→sub-arch 映射现在指向注入的配置，使内存流程读取注入的 dataflow /
        # temporal factor。
        self.assertEqual(sim.layer_to_sub_arch_mapping["layer1"]["tiles"], 4)

        # L4 topology 覆盖已落到网表上。
        netlist = sub["core"]["netlist"]
        self.assertIn("scaling_rules", netlist)
        self.assertEqual(netlist["temporal_accum_factor"]["duration"], 3)

    def test_inject_hardware_applies_l4_override(self):
        sim = MockSimulator()
        hw = default_hardware()
        hw["arch"]["topology"]["sharing_policy"] = "no_sharing"
        hw["arch"]["topology"]["laser_distribution"] = "H*W*N"
        _inject_hardware(sim, hw, str(ROOT))

        netlist = sub_arch(sim.sub_arch_dictionary)["core"]["netlist"]
        self.assertEqual(netlist["scaling_rules"]["i2"], "H*W*R*C")
        self.assertEqual(netlist["laser_power_distribution"], "H*W*N")
        self.assertEqual(netlist["laser_power_distribution_factor"], "H*W*N")

    def test_inject_hardware_mesh_uses_matrix_dimension_for_miniblock(self):
        # mesh core 的 miniblock 必须是 n_modes × n_modes 的酉尺寸，而非 MZI 数量
        # 包围盒（height=n_mzi, width=1），否则真实仿真器会把工作负载铺到退化的
        # n_mzi × 1 阵列上。
        sim = MockSimulator()
        hw = default_hardware()
        hw["arch"]["topology"]["core_kind"] = "mzi_mesh_clements"
        hw["arch"]["topology"]["node_kind"] = "tunable_mzi"
        hw["arch"]["topology"]["mesh_rank"] = 4  # n_modes（酉秩）
        hw["arch"]["optical_core"]["height"] = 4
        hw["arch"]["optical_core"]["width"] = 4
        _inject_hardware(sim, hw, str(ROOT))

        sub = sub_arch(sim.sub_arch_dictionary)
        self.assertEqual(sub["core"]["type"], "MZI_mesh")
        # N=4 Clements 时 n_mzi = 6；miniblock 必须保持 4x4，而非 6x1。
        self.assertEqual(sub["core"]["height"], 6)
        self.assertEqual(sub["core"]["width"], 1)
        self.assertEqual(sub["core"]["matrix_height"], 4)
        self.assertEqual(sub["core"]["matrix_width"], 4)
        for layer in ("layer1", "layer2"):
            self.assertEqual(sim.arch_blocks[layer], [4, 2, 4, 4], layer)


if __name__ == "__main__":
    unittest.main()
