"""Matched-quality sample-efficiency analysis for agent/baseline trajectories.

Round zero is the shared initial design.  Rounds 1..N are proposal/evaluation
rounds, so the checkpoint at round 10 is ``objective_curve[10]`` rather than
the tenth array element.  The primary matched-quality target for each workload
is fixed by the best traditional method at round 50; it is never redefined from
the agent result.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from baselines.compare import AGENT_LABEL, discover_agent_runs, discover_baseline_runs


DEFAULT_CHECKPOINTS: Tuple[int, ...] = (10, 20, 30, 40, 50)
DEFAULT_IMPROVEMENT_TARGETS: Tuple[float, ...] = (3.0, 5.0, 10.0)


def _normalized_curve(run: Dict[str, Any], max_round: int) -> List[float]:
    curve = [float(x) for x in (run.get("curve") or [])]
    initial = float((run.get("initial_metrics") or {}).get("objective", curve[0] if curve else 0.0))
    if not curve or initial <= 0:
        return []
    # Include round zero plus max_round proposal rounds.
    return [value / initial for value in curve[: max_round + 1]]


def _median_curve(runs: Sequence[Dict[str, Any]], max_round: int) -> List[float]:
    curves = [curve for curve in (_normalized_curve(run, max_round) for run in runs) if curve]
    if not curves:
        return []
    common = min(max_round + 1, min(len(curve) for curve in curves))
    return [float(median(curve[i] for curve in curves)) for i in range(common)]


def first_hit(curve: Sequence[float], target: float, max_round: int = 50) -> Optional[int]:
    """Return the first proposal round whose best-so-far value reaches target."""
    for round_index, value in enumerate(curve[: max_round + 1]):
        if float(value) <= float(target) + 1e-12:
            return round_index
    return None


def _fastest_method(
    curves: Dict[str, Sequence[float]], target: float, max_round: int,
) -> Tuple[Optional[str], Optional[int]]:
    hits = [
        (hit, method)
        for method, curve in curves.items()
        if (hit := first_hit(curve, target, max_round)) is not None
    ]
    if not hits:
        return None, None
    hit, method = min(hits, key=lambda item: (item[0], item[1]))
    return method, hit


def analyse(
    runs: Iterable[Dict[str, Any]],
    *,
    max_round: int = 50,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    improvement_targets: Sequence[float] = DEFAULT_IMPROVEMENT_TARGETS,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        workload = str(run.get("workload") or "unknown")
        grouped[workload][str(run.get("method") or "unknown")].append(run)

    matched_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    for workload, methods in sorted(grouped.items()):
        curves = {
            method: _median_curve(method_runs, max_round)
            for method, method_runs in methods.items()
        }
        curves = {method: curve for method, curve in curves.items() if curve}
        baseline_curves = {method: curve for method, curve in curves.items() if method != AGENT_LABEL}
        agent_curve = curves.get(AGENT_LABEL, [])

        for method, curve in sorted(curves.items()):
            for checkpoint in checkpoints:
                value = curve[checkpoint] if checkpoint < len(curve) else None
                checkpoint_rows.append({
                    "workload": workload,
                    "method": method,
                    "round": checkpoint,
                    "normalized_objective": value,
                    "improvement_pct": None if value is None else (1.0 - value) * 100.0,
                    "reached": value is not None,
                })

        targets: List[Tuple[str, Optional[float]]] = []
        complete_baselines = {
            method: curve for method, curve in baseline_curves.items() if len(curve) > max_round
        }
        if complete_baselines:
            matched_target = min(curve[max_round] for curve in complete_baselines.values())
            targets.append(("best_traditional_at_50", matched_target))
        else:
            targets.append(("best_traditional_at_50", None))
        targets.extend((f"fixed_{pct:g}pct", 1.0 - pct / 100.0) for pct in improvement_targets)

        for target_name, target in targets:
            traditional_method = traditional_round = agent_round = None
            if target is not None:
                traditional_method, traditional_round = _fastest_method(
                    baseline_curves, target, max_round,
                )
                agent_round = first_hit(agent_curve, target, max_round) if agent_curve else None
            saved = (
                traditional_round - agent_round
                if traditional_round is not None and agent_round is not None
                else None
            )
            saved_pct = (
                saved / traditional_round * 100.0
                if saved is not None and traditional_round not in (None, 0)
                else None
            )
            matched_rows.append({
                "workload": workload,
                "target_name": target_name,
                "target_normalized_objective": target,
                "traditional_method": traditional_method,
                "traditional_round": traditional_round,
                "agent_round": agent_round,
                "agent_reached": agent_round is not None,
                "iterations_saved": saved,
                "iterations_saved_pct": saved_pct,
            })

    return {"matched_quality": matched_rows, "checkpoints": checkpoint_rows}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "matched_quality.csv", result["matched_quality"])
    _write_csv(out_dir / "checkpoints.csv", result["checkpoints"])
    (out_dir / "sample_efficiency.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    lines = [
        "# Matched-quality sample efficiency",
        "",
        "Round 0 is the shared initial design; rounds 1–50 are proposal/evaluation rounds.",
        "NR means that the method did not reach the predeclared target within 50 rounds.",
        "",
        "| workload | target | target norm. obj. | fastest traditional | traditional round | agent round | saved rounds | saved |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["matched_quality"]:
        def show(value: Any, fmt: str = "{}") -> str:
            return "NR" if value is None else fmt.format(value)
        lines.append(
            f"| {row['workload']} | {row['target_name']} | "
            f"{show(row['target_normalized_objective'], '{:.6f}')} | "
            f"{show(row['traditional_method'])} | {show(row['traditional_round'])} | "
            f"{show(row['agent_round'])} | {show(row['iterations_saved'])} | "
            f"{show(row['iterations_saved_pct'], '{:.1f}%')} |"
        )
    (out_dir / "sample_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute matched-quality iteration savings.")
    parser.add_argument("--agent-runs", action="append", default=None)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--max-round", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs: List[Dict[str, Any]] = []
    for pattern in args.agent_runs or []:
        runs.extend(discover_agent_runs([pattern]))
    runs.extend(discover_baseline_runs(args.baseline_root))
    result = analyse(runs, max_round=args.max_round)
    write_report(result, args.out)
    print(f"[sample-efficiency] wrote {len(result['matched_quality'])} target rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
