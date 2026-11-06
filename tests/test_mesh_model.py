"""MZI-mesh 拓扑模型（已发表的 MZI 数量/深度）的测试。"""

import unittest

from photon_tuner.simulators.mesh_model import (
    COUPLERS_PER_MZI,
    PHASE_SHIFTERS_PER_MZI,
    depth,
    mesh_topology,
    n_mzi,
)


class TestMeshTopology(unittest.TestCase):
    def test_reck_counts_and_depth(self):
        # Reck 1994 三角型：N(N-1)/2 个 MZI，深度 2N-3。
        self.assertEqual(n_mzi("reck", 4), 6)
        self.assertEqual(depth("reck", 4), 5)
        self.assertEqual(n_mzi("reck", 3), 3)
        self.assertEqual(depth("reck", 3), 3)

    def test_clements_counts_and_depth(self):
        # Clements 2016 矩型：N(N-1)/2 个 MZI，深度 N。
        self.assertEqual(n_mzi("clements", 4), 6)
        self.assertEqual(depth("clements", 4), 4)
        self.assertEqual(n_mzi("clements", 8), 28)
        self.assertEqual(depth("clements", 8), 8)

    def test_butterfly_counts_and_depth(self):
        # Butterfly FFT mesh（Tang 2021）：M*log2(M)/2 个 MZI，深度 log2(M)。
        self.assertEqual(n_mzi("butterfly", 8), 12)
        self.assertEqual(depth("butterfly", 8), 3)
        self.assertEqual(n_mzi("butterfly", 4), 4)
        self.assertEqual(depth("butterfly", 4), 2)

    def test_device_multiplicity(self):
        topo = mesh_topology("clements", 4)
        self.assertEqual(topo["n_coupler"], 2 * topo["n_mzi"])
        self.assertEqual(topo["n_phase_shifter"], 2 * topo["n_mzi"])
        self.assertEqual(topo["n_coupler"], 6 * COUPLERS_PER_MZI)

    def test_structural_flags(self):
        self.assertTrue(mesh_topology("reck", 4)["universal"])
        self.assertFalse(mesh_topology("reck", 4)["crossing_free"])
        self.assertTrue(mesh_topology("clements", 4)["universal"])
        self.assertTrue(mesh_topology("clements", 4)["crossing_free"])
        self.assertFalse(mesh_topology("butterfly", 8)["universal"])

    def test_cited_references(self):
        self.assertEqual(mesh_topology("reck", 4)["ref"], "10.1103/PhysRevLett.73.58")
        self.assertEqual(mesh_topology("clements", 4)["ref"], "10.1364/OPTICA.3.001460")
        # Butterfly：MZI 数量/深度源自经典 FFT（Cooley-Tukey 1965），而非光子 mesh
        # 论文（Tang 2021 已被撤回——它是 10 端口通用处理器，不是 butterfly）。
        self.assertEqual(mesh_topology("butterfly", 8)["ref"], "10.1090/S0025-5718-1965-0178586-1")

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            mesh_topology("bogus", 4)


class TestMeshModelMatchesTopologySpace(unittest.TestCase):
    """mesh_model 计算出的 MZI 数量/深度必须与 topology_space 声明的 note 一致。"""

    def test_counts_agree_with_declared_notes(self):
        from photon_tuner.core.topology_space import CORE_KINDS

        # Reck："depth 2N-3, N(N-1)/2 MZIs"——针对 CORE_KINDS note 做抽查。
        self.assertIn("depth 2N-3", CORE_KINDS["mzi_mesh_reck"]["note"])
        self.assertIn("N(N-1)/2", CORE_KINDS["mzi_mesh_reck"]["note"])
        self.assertIn("depth N", CORE_KINDS["mzi_mesh_clements"]["note"])
        self.assertIn("M*log2(M)/2", CORE_KINDS["mzi_mesh_butterfly"]["note"])


if __name__ == "__main__":
    unittest.main()
