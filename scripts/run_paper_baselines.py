"""Run the paper's traditional baselines from the canonical workload manifest.

Each workload is executed in its own subprocess because the real SimPhony MCP
bridge receives one model factory and one mapping type per process.  Completed
trajectories are skipped by ``baselines.runner``, so the campaign is resumable.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real SimPhony baselines for paper workloads.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--algorithms", default="rs,sa,bo,ga,sweep")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--il-budget", type=float, default=10.0)
    parser.add_argument("--workloads", default=None,
                        help="Optional comma-separated subset; default runs all manifest entries.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workloads = list(manifest.get("workloads") or [])
    if args.workloads:
        selected = {item.strip() for item in args.workloads.split(",") if item.strip()}
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
        initial_hardware = _initial_hardware_path(item)
        cmd = [
            sys.executable, "-m", "baselines.runner",
            "--algo", args.algorithms,
            "--workload", name,
            "--budget", str(args.budget),
            "--seeds", args.seeds,
            "--model-type", str(item["model_type"]),
            "--nn-model-factory", str(item["nn_model_factory"]),
            "--initial-hardware", str(initial_hardware),
            "--il-budget", str(args.il_budget),
            "--output-root", str(output_root),
        ]
        log_path = log_root / f"{name}.log"
        row: Dict[str, Any] = {
            "workload": name,
            "family": item.get("family"),
            "model_type": item["model_type"],
            "nn_model_factory": item["nn_model_factory"],
            "initial_hardware": str(initial_hardware),
            "budget": args.budget,
            "seeds": args.seeds,
            "algorithms": args.algorithms,
            "log": str(log_path),
            "started_at_unix": time.time(),
            "status": "running",
        }
        rows.append(row)
        _write_status(status_path, rows)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        row["returncode"] = completed.returncode
        row["finished_at_unix"] = time.time()
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        _write_status(status_path, rows)
        if completed.returncode != 0:
            print(f"[paper-baselines] {name} failed; see {log_path}", file=sys.stderr)
            return completed.returncode
        print(f"[paper-baselines] {name} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
