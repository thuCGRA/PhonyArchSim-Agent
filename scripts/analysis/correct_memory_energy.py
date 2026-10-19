"""Offline correction for the on-chip memory-energy over-count.

SimPhony's ``memory.py`` multiplies CACTI's "dynamic read/write energy **per
access**" (pJ per ``block_size``-byte access) by the byte count, so GLB2/GLB1/RF
energies are over-counted by ``block_size``. HBM energy is correct (pJ/Byte from
config). This is NOT in the objective scalar (which is computation energy), but
the reported ``report.memory.energy`` should be corrected for the paper.

Corrected total = HBM + (GLB2 + GLB1 + RF) / block_size.

Usage::

    python scripts/analysis/correct_memory_energy.py runs/sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def corrected_memory_energy(report: dict) -> dict:
    """Return {level: corrected_pJ, ...} plus the totals for one report."""
    spec = report.get("memory", {}).get("spec", {})
    energy = report.get("memory", {}).get("energy", {})

    # report.memory.energy is {"LT": {HBM:.., GLB2:.., GLB1:.., RF:..}}.
    per_sub = {}
    for sub, levels in energy.items():
        if not isinstance(levels, dict):
            continue
        corrected = {}
        for level, value in levels.items():
            value = float(value)
            if level == "HBM":
                corrected[level] = value  # already pJ/Byte * Bytes
            else:
                block_size = 8.0
                try:
                    block_size = float(spec.get(level, {}).get("block_size", 8))
                except (AttributeError, TypeError, ValueError):
                    pass
                corrected[level] = value / block_size  # pJ/access -> pJ/Byte
        per_sub[sub] = corrected
    return per_sub


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sweep")
    print(f"{'workload':10s} {'run':>4s} {'HBM_pJ':>12s} {'onchip_pJ':>12s} {'total_pJ':>12s} {'buggy_total_pJ':>12s}")
    for report_path in sorted(root.glob("*/run_*/iter_*/report.json")):
        rel = report_path.relative_to(root)
        wl = rel.parts[0]
        run = rel.parts[1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_sub = corrected_memory_energy(report)
        for sub, levels in per_sub.items():
            hbm = levels.get("HBM", 0.0)
            onchip = sum(v for k, v in levels.items() if k != "HBM")
            buggy = float(report["memory"]["energy"][sub].get("HBM", 0.0)) + sum(
                float(v) for k, v in report["memory"]["energy"][sub].items() if k != "HBM"
            )
            print(f"{wl:10s} {run:>4s} {hbm:12.3e} {onchip:12.3e} {hbm+onchip:12.3e} {buggy:12.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
