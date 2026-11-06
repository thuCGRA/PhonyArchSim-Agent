"""Probe: does real SimPhony evaluate one hardware point in .venv-simulator?

Run with the Python-3.10 venv:
    .venv-simulator/bin/python scripts/dev/probe_real.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIMPHONY_ROOT = ROOT / "photon_tuner" / "simulators" / "simphony"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIMPHONY_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "mplconfig"))

from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import default_hardware, normalize_hardware_to_design_space
from photon_tuner.simulators.simphony import SimphonyAdapter


def main() -> int:
    ss = read_json(ROOT / "configs" / "search_space.json")
    hardware = normalize_hardware_to_design_space(default_hardware(), ss)

    adapter = SimphonyAdapter(
        simphony_root=str(SIMPHONY_ROOT),
        output_root=ROOT / ".runs_probe" / "real",
        search_space=ss,
        mode="real",
        nn_model_factory="tests.factory:build_vit_tiny",
        workload_name="vit_tiny",
    )
    print("==> evaluating real SimPhony (vit_tiny) ...", flush=True)
    result = adapter.evaluate(hardware, 0)
    print("==> metrics:", json.dumps(result.metrics, indent=2), flush=True)
    print("==> detail_files:", sorted(p.name for p in result.detail_files.values()), flush=True)
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
