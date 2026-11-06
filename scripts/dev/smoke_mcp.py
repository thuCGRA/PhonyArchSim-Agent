"""MCP roundtrip smoke: evaluate one hardware point via the out-of-process server.

Runs under .venv-agent (3.14) and drives the Python-3.10 SimPhony server over
stdio, exercising the full adapter -> MCP client -> server -> real SimPhony path.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.simulators.simphony import SimphonyAdapter


def main() -> int:
    ss = read_json(ROOT / "configs" / "search_space.json")
    hardware = normalize_hardware_to_design_space(default_hardware(), ss)

    adapter = SimphonyAdapter(
        simphony_root=str(ROOT / "photon_tuner" / "simulators" / "simphony"),
        output_root=ROOT / ".runs_mcp" / "host",
        search_space=ss,
        mode="real",
        transport="mcp",
        nn_model_factory="tests.factory:build_vit_tiny",
        workload_name="vit_tiny",
    )
    print("==> evaluating via MCP (vit_tiny) ...", flush=True)
    try:
        result = adapter.evaluate(hardware, 0)
        print("==> metrics:", json.dumps(result.metrics, indent=2), flush=True)
        print("==> run_dir:", result.run_dir, flush=True)
        print("==> raw_files:", {k: v.name for k, v in result.raw_files.items()}, flush=True)
        print("SMOKE_MCP_OK")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
