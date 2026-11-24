"""Verify a fresh installation by running one ViT-Tiny search in its own directory."""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def check_run(run, mode):
    if read_json(run / "summary.json")["iterations"] != 1:
        raise ValueError("The search did not complete one iteration.")
    for step in ("iter_000", "iter_001"):
        objective = read_json(run / step / "metrics.json")["objective"]
        if not math.isfinite(objective):
            raise ValueError(f"{step} has a non-finite objective.")
        print(f"{step} objective = {objective:.6g}")

    analysis = run / "iter_001" / "analysis"
    if not read_json(analysis / "candidate_proposal.json") or not read_json(run / "iter_001/hardware.json"):
        raise ValueError("The candidate proposal or evaluated hardware is empty.")
    if mode == "model":
        with (analysis / "llm_trace.jsonl").open(encoding="utf-8") as trace:
            calls = sum(json.loads(line).get("event") == "llm_end" for line in trace if line.strip())
        if not calls:
            raise ValueError("No completed model call was recorded.")
        print(f"Completed model calls = {calls}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("simulator", "model"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/verification")
    args = parser.parse_args()
    label = args.mode
    try:
        if label == "model":
            config = ROOT / "configs/llm.json"
            settings = read_json(config)
            for key in ("provider", "model", "base_url"):
                if not isinstance(settings.get(key), str) or not settings[key].strip():
                    raise ValueError(f"Set {key} in configs/llm.json.")
            if settings["provider"].strip().lower() == "mock":
                raise ValueError("Select an online model in configs/llm.json.")
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                raise ValueError("Set OPENAI_API_KEY for the configured endpoint.")
            options = ["--llm-config", str(config)]
        else:
            options = ["--llm-config", "configs/llm.example.json", "--provider", "mock"]

        args.output_root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=args.output_root)).resolve()
        print(f"Verification output: {output}", flush=True)
        subprocess.run(
            [sys.executable, "main.py", *options, "--workload", "vit_tiny",
             "--nn-model-factory", "tests.factory:build_vit_tiny", "--iterations", "1",
             "--output-root", str(output), "--quiet"],
            cwd=ROOT, check=True,
        )
        check_run(output / "run_iteration_001", label)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"FAIL {label}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
