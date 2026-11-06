"""MCP 主机侧明细文件物化的离线测试。

真实模式通过 MCP 在进程外运行 SimPhony；主机仍须物化 ``report.json`` 以及
各工作流的明细文件，使智能体的读取工具可用。本测试用返回带内 ``report`` 的
假 MCP 客户端驱动 ``SimphonyAdapter._evaluate_via_mcp``，并校验主机写出与
进程内路径相同的明细文件。
"""

import tempfile
import unittest
from pathlib import Path

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.simulators.simphony import SimphonyAdapter

ROOT = Path(__file__).resolve().parents[2]


class _FakeMCPClient:
    def __init__(self, report):
        self._report = report

    def evaluate(self, hardware, iteration):
        return {
            "metrics": {
                "latency_s": 1.0,
                "energy_pj": 2.0,
                "area_um2": 3.0,
                "total_macs": 100.0,
                "pJ_per_mac": 0.02,
                "core_insertion_loss": 4.0,
            },
            "report": self._report,
        }

    def close(self):
        pass


class TestMCPDetailFiles(unittest.TestCase):
    def test_evaluate_via_mcp_writes_detail_files(self):
        ss = read_json(ROOT / "configs" / "search_space.json")
        hardware = normalize_hardware_to_design_space(default_hardware(), ss)
        report = {
            "metrics": {"energy_pj": 2.0},
            "energy": {"by_device": {}},
            "area": {"by_device": {}},
            "insertion_loss": {"LT": {"core_insertion_loss": 4.0}},
            "workload": {"partition_cycles": {"LT": {}}},
            "memory": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            adapter = SimphonyAdapter(
                simphony_root=str(ROOT / "photon_tuner" / "simulators" / "simphony"),
                output_root=Path(tmp),
                search_space=ss,
                mode="analytic",
                transport="mcp",
                workload_name="vit_tiny",
            )
            adapter._mcp_client = _FakeMCPClient(report)
            result = adapter.evaluate(hardware, 0)
            self.assertIn("report", result.detail_files)
            self.assertTrue(result.detail_files["report"].exists())
            for name in ("energy", "area", "insertion_loss", "workload", "memory"):
                self.assertTrue(result.detail_files[name].exists(), name)
            adapter.close()


if __name__ == "__main__":
    unittest.main()
