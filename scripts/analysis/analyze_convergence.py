"""Objective-vs-iteration trajectory for a multi-iteration convergence run.

Reads ``<output_root>/iter_*/metrics.json`` and prints the per-iteration
objective, the running best, and the cumulative improvement. Also prints the
hardware delta between consecutive iterations (what changed).

Usage::

    python scripts/analysis/analyze_convergence.py runs/convergence_vit_base
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/convergence_vit_base")
    iters = sorted(root.glob("iter_*"), key=lambda p: int(p.name.split("_")[1]))

    prev_hw = None
    print(f"{'iter':>6} {'objective':>14} {'best':>14} {'cumul%':>7}  delta")
    best = None
    base = None
    for it in iters:
        mf = it / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        obj = m.get("objective")
        hw = None
        hwf = it / "hardware.json"
        if hwf.exists():
            hw = json.loads(hwf.read_text(encoding="utf-8"))
        if base is None:
            base = obj
        if best is None or obj < best:
            best = obj
        cumul = (1 - best / base) * 100 if base else 0.0
        delta = ""
        if prev_hw is not None and hw is not None:
            delta = _hw_delta(prev_hw, hw)
        print(f"{it.name:>6} {obj:14.6g} {best:14.6g} {cumul:7.2f}%  {delta}")
        prev_hw = hw
    return 0


def _hw_delta(a: dict, b: dict) -> str:
    parts = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        else:
            parts.append(f"{path}:{x}->{y}")

    walk(a, b, "")
    return "; ".join(parts)[:200]


if __name__ == "__main__":
    raise SystemExit(main())
