"""Smoke test: run the real SimPhony ONNArchSimulator end-to-end on CPU.

Runnable from anywhere (it chdir's into the writable SimPhony checkout so that
``memory.py`` resolves ``./cacti`` and ``./configs/memory`` correctly)::

    source scripts/env/activate_real.sh
    python tests/smoke_real.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMPHONY_ROOT = PROJECT_ROOT / "photon_tuner" / "simulators" / "simphony"
TESTS_DIR = Path(__file__).resolve().parent

# memory.py / CACTI resolve relative to the SimPhony checkout root.
os.chdir(SIMPHONY_ROOT)

sys.path.insert(0, str(SIMPHONY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))  # for ``factory``

from factory import build_model  # noqa: E402
from onnarchsim.simulator import ONNArchSimulator  # noqa: E402


def main():
    model = build_model()
    print(f"[1] model: {type(model).__name__} on {next(model.parameters()).device}")

    sim = ONNArchSimulator(
        nn_model=model,
        onn_conversion_cfg=None,
        nn_conversion_cfg="configs/nn_mapping/simple_cnn.yml",
        onn_model=None,
        model2arch_map_cfg="configs/architecture_mapping/simple_lt_cnn.yml",
        devicelib_root="configs/devices",
        device_cfg_files=["*/*.yml"],
        arch_cfg_file="configs/design/architectures/LT_hetero.yml",
        arch_version="v1",
        input_shape=(2, 3, 224, 224),
        log_path=str(PROJECT_ROOT / "runs" / "legacy" / "smoke_real_log.txt"),
    )
    print("[2] simulator initialized (device DB + arch config + NN->ONN done)")

    partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)
    print(f"[3] partition_cycles: {partition_cycles}")

    insertion_loss = sim.simu_insertion_loss()
    print(f"[4] insertion_loss keys: {list(insertion_loss)[:5]}...")

    energy_breakdown, total_energy, latency = sim.simu_energy(
        partition_cycles, insertion_loss
    )
    print(f"[5] total_energy={total_energy}  latency={latency}")

    area_breakdown, total_area = sim.simu_area()
    print(f"[6] total_area={total_area}")

    mem_latency, mem_energy, mem_spec = sim.simu_memory_cost(partition_cycles)
    print(f"[7] memory: latency={mem_latency} energy={mem_energy} spec={mem_spec}")

    print("SMOKE TEST PASSED")
    print(f"METRICS: energy_pj={total_energy} area_um2={total_area} latency_s={latency}")


if __name__ == "__main__":
    main()
