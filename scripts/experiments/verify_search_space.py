#!/usr/bin/env python3
"""Search-space validity verification CLI.

Runs the deterministic ablation harness: baseline + single-variable perturbation
for every (parameter, candidate value, workload), records injection readback
and metric deltas into ``data/evidence.db``.

Example (real mode, P3 representative subset):
    source scripts/env/activate_real.sh
    python scripts/experiments/verify_search_space.py \
        --simphony-root <project>/photon_tuner/simulators/simphony \
        --workloads vit_tiny,vit_small,vit_base,mobilevit \
        --params work_freq,core_height,core_width,num_wavelength,dataflow,tiles,precision_in_bit,hbm_bandwidth,sharing_policy,laser_distribution \
        --devices mzm,coupler,photodetector \
        --mode real --model-type transformer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photon_tuner.core.design_space import DesignSpace
from photon_tuner.common.io import read_json
from photon_tuner.core.search_space import (
    default_hardware,
    discover_device_catalog,
    discover_device_variants,
)
from photon_tuner.evidence import EvidenceStore, SearchSpaceProbe
from photon_tuner.simulators.simphony import SimphonyAdapter

# P3 representative subset (confirmed with the user).
DEFAULT_ARCH_PARAMS = [
    "work_freq", "core_height", "core_width", "num_wavelength", "dataflow",
    "tiles", "precision_in_bit", "hbm_bandwidth", "sharing_policy",
    "laser_distribution",
]
DEFAULT_DEVICES = ["mzm", "coupler", "photodetector"]
DEFAULT_WORKLOADS = ["vit_tiny", "vit_small", "vit_base", "mobilevit"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simphony-root", required=True)
    p.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    p.add_argument("--params", default=",".join(DEFAULT_ARCH_PARAMS))
    p.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    p.add_argument("--search-space", default="configs/search_space.json")
    p.add_argument("--mode", default="real", choices=["real", "analytic", "auto"])
    p.add_argument("--model-type", default="transformer")
    p.add_argument("--output-dir", default="runs/evidence")
    p.add_argument("--db", default="data/evidence.db")
    p.add_argument("--k", type=int, default=None, help="max candidates per param")
    p.add_argument("--check-determinism", action="store_true")
    p.add_argument("--only", default=None, help="comma list of params/devices to run")
    return p.parse_args()


def _parse_csv(s: Optional[str]) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _device_level(device_type: str, baseline_hw: Dict[str, Any]) -> str:
    if device_type in baseline_hw.get("devices", {}).get("core", {}):
        return "core"
    if device_type in baseline_hw.get("devices", {}).get("node", {}):
        return "node"
    raise ValueError(f"device type {device_type!r} not found in baseline devices")


def _build_adapter(args: argparse.Namespace, workload: str) -> SimphonyAdapter:
    return SimphonyAdapter(
        simphony_root=args.simphony_root,
        output_root=str(Path(args.output_dir) / workload),
        search_space={},  # match the sweep's no-flag objective (energy*area*latency)
        mode=args.mode,
        model_type=args.model_type,
        nn_model_factory=f"tests.factory:build_{workload}",
        workload_name=workload,
    )


def _run_workload(
    args: argparse.Namespace,
    workload: str,
    baseline_hw: Dict[str, Any],
    design_space: DesignSpace,
    variants: Dict[str, List[str]],
    store: EvidenceStore,
    arch_params: List[str],
    devices: List[str],
) -> Dict[str, Any]:
    adapter = _build_adapter(args, workload)
    probe = SearchSpaceProbe(
        adapter=adapter,
        design_space=design_space,
        search_space={},
        store=store,
        output_root=Path(args.output_dir) / workload,
        device_variants=variants,
    )

    baseline_eval, features = probe.run_baseline(baseline_hw, workload)
    m = baseline_eval["metrics"]
    print(f"\n[{workload}] baseline objective={m['objective']:.6e} "
          f"energy={m['energy_pj']:.4f} area={m['area_um2']:.2f} "
          f"latency={m['latency_s']:.6g}")

    if args.check_determinism:
        again, _ = probe.run_baseline(baseline_hw, workload, tag="baseline_recheck")
        same = all(
            again["metrics"][k] == m[k]
            for k in ("objective", "energy_pj", "area_um2", "latency_s")
        )
        print(f"[{workload}] determinism recheck: {'PASS (bit-identical)' if same else 'FAIL'}")

    summary: Dict[str, Any] = {"workload": workload, "arch": [], "device": []}

    for param in arch_params:
        records = probe.probe_arch_param(
            param, baseline_hw, baseline_eval, features, workload, max_values=args.k
        )
        for r in records:
            d = r["metric_deltas"]
            print(
                f"  {param}: {r['old_value']}->{r['new_value']} "
                f"inj={bool(r['injection_confirmed'])} {r['constraint_result']} "
                f"null={bool(r['is_null'])} obj={d['objective_pct']:+.2f}% "
                f"energy={d['energy_pct']:+.2f}% area={d['area_pct']:+.2f}% "
                f"latency={d['latency_pct']:+.2f}%"
            )
            summary["arch"].append({
                "param": param,
                "new_value": r["new_value"],
                "objective_pct": d["objective_pct"],
                "injection_confirmed": bool(r["injection_confirmed"]),
                "is_null": bool(r["is_null"]),
            })

    for dev in devices:
        level = _device_level(dev, baseline_hw)
        records = probe.probe_device(
            level, dev, baseline_hw, baseline_eval, features, workload, max_values=args.k
        )
        for r in records:
            d = r["metric_deltas"]
            print(
                f"  devices.{level}.{dev}: {r['old_value']}->{r['new_value']} "
                f"inj={bool(r['injection_confirmed'])} null={bool(r['is_null'])} "
                f"obj={d['objective_pct']:+.2f}%"
            )
            summary["device"].append({
                "param": f"devices.{level}.{dev}",
                "new_value": r["new_value"],
                "objective_pct": d["objective_pct"],
                "injection_confirmed": bool(r["injection_confirmed"]),
                "is_null": bool(r["is_null"]),
            })

    return summary


def main() -> int:
    args = parse_args()

    arch_params = _parse_csv(args.params)
    devices = _parse_csv(args.devices)
    workloads = _parse_csv(args.workloads)
    only = set(_parse_csv(args.only))
    if only:
        arch_params = [p for p in arch_params if p in only]
        devices = [d for d in devices if d in only]

    search_space = read_json(args.search_space) if Path(args.search_space).exists() else {}
    devicelib_root = str(Path(args.simphony_root) / "configs" / "devices")
    catalog = discover_device_catalog(devicelib_root)
    variants = discover_device_variants(devicelib_root)
    design_space = DesignSpace(spec=search_space, device_catalog=catalog)

    baseline_hw = default_hardware()
    store = EvidenceStore(args.db)

    summaries = []
    for workload in workloads:
        summaries.append(
            _run_workload(
                args, workload, baseline_hw, design_space, variants, store,
                arch_params, devices,
            )
        )

    export_path = store.export(Path(args.output_dir) / "evidence.jsonl")
    print(f"\n=== evidence store: {store.count()} records -> {export_path} ===")
    store.close()

    summary_path = Path(args.output_dir) / "verify_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
