"""Run official DeepSeek Flash agent trajectories for paper workloads.

The API key is inherited from ``OPENAI_API_KEY`` and is never written to a
command, log, status file, or experiment artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "paper_experiment_workloads.json"


def _initial_hardware_path(item: Dict[str, Any]) -> Path:
    raw = item.get("initial_hardware")
    if not raw:
        raise SystemExit(f"workload {item.get('name')!r} has no initial_hardware")
    path = Path(str(raw))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"initial hardware file does not exist: {path}")
    return path


def _write_status(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def _complete_trajectory(run_root: Path, iterations: int) -> Optional[Path]:
    for path in sorted(run_root.glob("run_iteration_*/trajectory.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if len(data.get("history") or []) == iterations + 1:
            return path
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real official-DeepSeek agent campaign.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--il-budget", type=float, default=10.0)
    parser.add_argument("--workloads", default=None)
    parser.add_argument("--provider", default="deepseek-official")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required but was not provided")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workloads = list(manifest.get("workloads") or [])
    if args.workloads:
        selected = {value.strip() for value in args.workloads.split(",") if value.strip()}
        workloads = [item for item in workloads if item.get("name") in selected]
        missing = selected - {str(item.get("name")) for item in workloads}
        if missing:
            raise SystemExit(f"unknown workloads in selection: {sorted(missing)}")

    output_root = args.output_root.resolve()
    log_root = output_root.parent / f"{output_root.name}_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    status_suffix = ""
    if args.workloads:
        status_suffix = "_" + "_".join(
            value.strip() for value in args.workloads.split(",") if value.strip()
        )
    status_path = output_root / f"campaign_status{status_suffix}.json"
    rows: List[Dict[str, Any]] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    for item in workloads:
        name = str(item["name"])
        run_root = output_root / name / "seed_0"
        initial_hardware = _initial_hardware_path(item)
        existing = _complete_trajectory(run_root, args.iterations)
        if existing is not None:
            rows.append({"workload": name, "status": "skipped_complete",
                         "trajectory": str(existing)})
            _write_status(status_path, rows)
            print(f"[paper-agents] {name} skipped (complete)")
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        history_db = run_root / "history.db"
        if history_db.exists():
            raise SystemExit(
                f"refusing to reuse RAG history for fresh workload {name}: {history_db}; "
                "choose a new output root"
            )
        log_path = log_root / f"{name}.log"
        command = [
            sys.executable, str(ROOT / "main.py"),
            "--provider", args.provider,
            "--model", args.model,
            "--base-url", args.base_url,
            "--iterations", str(args.iterations),
            "--workload", name,
            "--model-type", str(item["model_type"]),
            "--nn-model-factory", str(item["nn_model_factory"]),
            "--initial-hardware", str(initial_hardware),
            "--il-budget", str(args.il_budget),
            "--output-root", str(run_root),
            "--state-strategy", "best",
            "--search-strategy", "joint",
            "--max-edits", "1",
            "--schedule-strategy", "rag",
            "--effective-improvement-pct", "5",
            "--warm-precedent-target", "3",
            "--minimum-mature-layer-coverage", "3",
            "--convergence-window", "10",
            "--convergence-delta", "0.10",
            "--max-consecutive-focus", "0",
            "--max-pipeline-evidence-age", "2",
        ]
        row: Dict[str, Any] = {
            "workload": name,
            "family": item.get("family"),
            "model_type": item["model_type"],
            "nn_model_factory": item["nn_model_factory"],
            "initial_hardware": str(initial_hardware),
            "iterations": args.iterations,
            "provider": args.provider,
            "model": args.model,
            "max_edits": 1,
            "scheduler": {
                "effective_improvement_pct": 5.0,
                "warm_precedent_target": 3,
                "minimum_mature_layer_coverage": 3,
                "convergence_window": 10,
                "convergence_delta": 0.10,
                "focus_exit_delta": None,
                "max_consecutive_focus": 0,
                "max_pipeline_evidence_age": 2,
                "require_pipeline_before_focus": True,
                "phase_trend": ["serial", "pipeline", "focus"],
            },
            "log": str(log_path),
            "rag_history_db": str(history_db),
            "rag_start_case_count": 0,
            "started_at_unix": time.time(),
            "status": "running",
        }
        rows.append(row)
        _write_status(status_path, rows)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        trajectory = _complete_trajectory(run_root, args.iterations)
        row["returncode"] = completed.returncode
        row["finished_at_unix"] = time.time()
        row["trajectory"] = str(trajectory) if trajectory else None
        row["status"] = "complete" if completed.returncode == 0 and trajectory else "failed"
        _write_status(status_path, rows)
        if row["status"] != "complete":
            print(f"[paper-agents] {name} failed; see {log_path}", file=sys.stderr)
            return completed.returncode or 1
        print(f"[paper-agents] {name} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
