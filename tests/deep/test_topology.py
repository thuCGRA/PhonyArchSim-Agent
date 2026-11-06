"""电路结构拓扑提供器与工具的测试。"""

import unittest

from photon_tuner.core.search_space import default_hardware
from photon_tuner.core.topology_spec import OPTICAL_PATH_ORDER, StaticTopologyProvider


class TestTopologyProvider(unittest.TestCase):
    def test_optical_path_order(self):
        provider = StaticTopologyProvider()
        graph = provider.topology(default_hardware())
        self.assertEqual(graph.source, "static")
        self.assertEqual(
            graph.optical_path,
            ["on_chip_laser", "laser_splitter", "waveguide", "mzm", "mrr_rerouter", "coupler", "photodetector"],
        )

    def test_path_nodes_flagged(self):
        provider = StaticTopologyProvider()
        graph = provider.topology(default_hardware())
        by_type = {n.device_type: n for n in graph.nodes}
        self.assertTrue(by_type["mzm"].on_optical_path)
        self.assertFalse(by_type["dac"].on_optical_path)
        self.assertFalse(by_type["adc"].on_optical_path)
        self.assertEqual(by_type["mzm"].device_name, "LT_MZM")

    def test_instance_switches_excluded(self):
        provider = StaticTopologyProvider()
        graph = provider.topology(default_hardware())
        types = {n.device_type for n in graph.nodes}
        self.assertNotIn("laser_source", types)
        self.assertNotIn("splitter_kind", types)


if __name__ == "__main__":
    unittest.main()
