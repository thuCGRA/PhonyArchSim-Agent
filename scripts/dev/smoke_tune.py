"""End-to-end deep-agents DSE smoke test (analytic + real Paratera, 1 iteration)."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from photon_tuner.common.io import read_json  # noqa: E402
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space  # noqa: E402
from photon_tuner.llm.config import LLMConfig  # noqa: E402
from photon_tuner.rag.vector_store import HistoryVectorStore  # noqa: E402
from photon_tuner.runtime.orchestrator import DeepAgentTuner  # noqa: E402
from photon_tuner.simulators.simphony import SimphonyAdapter  # noqa: E402


def main() -> int:
    search_space = read_json(ROOT / "configs" / "search_space.json")
    hardware = normalize_hardware_to_design_space(default_hardware(), search_space)

    with tempfile.TemporaryDirectory() as tmp:
        # 与 photon_tuner.cli 一致：real 模式 + MCP 桥（真实 ONNArchSimulator 跑在
        # Python-3.10 venv 中）。analytic 已废弃。
        adapter = SimphonyAdapter(
            simphony_root=ROOT / "photon_tuner" / "simulators" / "simphony",
            output_root=Path(tmp),
            search_space=search_space,
            mode="real",
            workload_name="vit_tiny",
            transport="mcp",
            mcp_env={"SIMPHONY_WORKLOAD": "vit_tiny", "SIMPHONY_MODEL_TYPE": "transformer"},
        )
        tuner = DeepAgentTuner(
            adapter=adapter,
            llm_config=LLMConfig(provider="paratera", temperature=0.2, max_tokens=16384, timeout=300),
            devicelib_root=str(ROOT / "photon_tuner" / "simulators" / "simphony" / "configs" / "devices"),
            output_root=Path(tmp) / "runs",
            search_space=search_space,
            store=HistoryVectorStore(Path(tmp) / "history.db"),
        )
        summary = tuner.tune(initial_hardware=hardware, max_iterations=1)

        print("=== SUMMARY ===")
        print("iterations:", summary["iterations"])
        print("best objective:", summary["best"]["objective"])
        print("history[0] (baseline) objective:", summary["history"][0]["metrics"]["objective"])
        print("run_dir:", summary["run_dir"])
        for f in sorted(Path(summary["run_dir"]).rglob("analysis/*.json")):
            print("  artifact:", f.relative_to(summary["run_dir"]))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
