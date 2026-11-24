"""Audit completeness and atomicity of the paper baseline campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence


METHODS = ("rs", "sa", "bo", "ga", "sweep")
WORKLOADS = (
    "vit_tiny", "vit_small", "vit_base",
    "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
    "cnn_tiny", "cnn_small", "cnn_base",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args(argv)
    issues = []
    rows = []
    for workload in WORKLOADS:
        for method in METHODS:
            path = args.root / method / workload / "seed_0" / "trajectory.json"
            if not path.is_file():
                issues.append(f"missing {method}/{workload}")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            history = data.get("history") or []
            curve = data.get("objective_curve") or []
            if data.get("budget") != args.budget:
                issues.append(f"budget {method}/{workload}={data.get('budget')}")
            expected = args.budget + 1
            if len(history) != expected:
                issues.append(f"history {method}/{workload}={len(history)}")
            if len(curve) != expected:
                issues.append(f"curve {method}/{workload}={len(curve)}")
            initial = float(data["initial_metrics"]["objective"])
            best = float(data["best"]["objective"])
            if not (math.isfinite(initial) and math.isfinite(best)):
                issues.append(f"nonfinite {method}/{workload}")
            non_atomic = []
            for item in history[1:]:
                action = item.get("action") or {}
                edits = action.get("edits")
                if isinstance(edits, list) and len(edits) > 1:
                    non_atomic.append(item.get("iteration"))
            if non_atomic:
                issues.append(f"non-atomic {method}/{workload}:{non_atomic[:5]}")
            rows.append((workload, method, initial, best,
                         (initial - best) / initial * 100.0, len(history)))

    print(f"runs {len(rows)} expected {len(WORKLOADS) * len(METHODS)}")
    print(f"issues {len(issues)}")
    for issue in issues:
        print(f"ISSUE {issue}")
    print("workload_counts")
    for workload in WORKLOADS:
        print(workload, sum(1 for row in rows if row[0] == workload))
    print("best_by_workload")
    for workload in WORKLOADS:
        candidates = [row for row in rows if row[0] == workload]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row[3])
        print(workload, best[1], f"improvement={best[4]:.3f}%", f"best={best[3]:.6g}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
