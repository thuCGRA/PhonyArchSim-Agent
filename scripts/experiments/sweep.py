"""Sequential real-mode DSE sweep over all workloads x N runs.

Each run is one full DSE invocation (baseline eval -> model/layer/optical/solution
agents -> proposal eval) against the real SimPhony simulator and the real LLM.

Layout::

    runs/sweep/<workload>/run_<i>/{iter_000,iter_001,...}
    runs/sweep/<workload>/run_<i>/run.log
    runs/sweep/sweep_summary.jsonl   (one record per run, appended)

Usage::

    source scripts/env/activate_real.sh
    python scripts/experiments/sweep.py [--runs 5] [--iterations 1] [--workloads vit_tiny,vit_small,vit_base,mobilevit_xxs,mobilevit_xs,mobilevit,cnn_tiny,cnn_small,cnn_base]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = [
    "vit_tiny", "vit_small", "vit_base",
    "mobilevit_xxs", "mobilevit_xs", "mobilevit",
    "cnn_tiny", "cnn_small", "cnn_base",
]
LLM_PROVIDER = "paratera"


def model_type_for(workload: str) -> str:
    """Map a workload to the SimPhony model type (and thus NN/arch mapping).

    Pure-CNN workloads use the ``simple_cnn`` mapping; everything else uses the
    ``simple_transformer`` mapping.
    """
    return "cnn" if workload.startswith("cnn_") else "transformer"


def run_one(workload: str, run_idx: int, iterations: int) -> dict:
    out = ROOT / "runs" / "sweep" / workload / f"run_{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--mode", "real",
        "--simphony-root", str(ROOT / "photon_tuner" / "simulators" / "simphony"),
        "--provider", LLM_PROVIDER,
        "--nn-model-factory", f"tests.factory:build_{workload}",
        "--model-type", model_type_for(workload),
        "--iterations", str(iterations),
        "--output-root", str(out),
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    record = {
        "workload": workload,
        "run": run_idx,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Best metrics live in the last successful iter_N/metrics.json (or summary).
    metrics = None
    summary = None
    for iterdir in sorted(out.glob("iter_*")):
        mf = iterdir / "metrics.json"
        if mf.exists():
            metrics = json.loads(mf.read_text(encoding="utf-8"))
    if metrics:
        record["objective"] = metrics.get("objective")
        record["energy_pj"] = metrics.get("energy_pj")
        record["area_um2"] = metrics.get("area_um2")
        record["latency_s"] = metrics.get("latency_s")

    # Distinguish "proposal changed the design" from "proposal was a no-op".
    hws = [json.loads((d / "hardware.json").read_text(encoding="utf-8"))
           for d in sorted(out.glob("iter_*")) if (d / "hardware.json").exists()]
    record["distinct_hardware"] = len({json.dumps(h, sort_keys=True) for h in hws})

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        record["error"] = tail.strip().splitlines()[-1] if tail.strip() else "unknown"

    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--skip-complete", action="store_true",
                    help="skip runs whose trajectory.json already exists")
    args = ap.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    summary_path = ROOT / "runs" / "sweep" / "sweep_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for wl in workloads:
        for i in range(args.runs):
            out_dir = ROOT / "runs" / "sweep" / wl / f"run_{i}"
            if args.skip_complete and (out_dir / "trajectory.json").exists():
                print(f"skip {wl} run_{i} (already complete)", flush=True)
                continue
            rec = run_one(wl, i, args.iterations)
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
