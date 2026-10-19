"""Command-line entry point for the photonic DSE framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import read_json
from photon_tuner.common.paths import SIMPHONY_ROOT
from photon_tuner.common.progress import ProgressReporter
from photon_tuner.llm.config import LLMConfig
from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.workload import load_workload_configs
from photon_tuner.llm.clients import create_llm_client
from photon_tuner.runtime.tuner import MultiAgentTuner
from photon_tuner.simulators.simphony import SimphonyAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photon-tuner",
        description="Multi-agent photonic accelerator design-space exploration on SimPhony.",
    )
    parser.add_argument("--initial-hardware", type=Path, default=None,
                        help="JSON file with the initial photonic hardware configuration.")
    parser.add_argument("--search-space", type=Path, default=None,
                        help="JSON file overriding the arch/device search-space candidates.")
    parser.add_argument("--simphony-root", type=Path, default=SIMPHONY_ROOT,
                        help="Root of the copied SimPhony project.")
    parser.add_argument("--devicelib-root", type=Path, default=None,
                        help="Device library root (defaults to <simphony-root>/configs/devices).")
    parser.add_argument("--output-root", type=Path, default=Path("runs"),
                        help="Directory for per-iteration runs, artifacts, and the history DB.")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Number of tuning iterations to run.")
    parser.add_argument("--mode", choices=["auto", "analytic", "real"], default="auto",
                        help="Simulator execution mode.")
    parser.add_argument("--llm-config", type=Path, default=Path("configs/llm.json"),
                        help="JSON LLM configuration (provider/model/api_key/base_url).")
    parser.add_argument("--provider", default=None,
                        help="LLM provider override (mock / paratera / deepseek / openai-compatible).")
    parser.add_argument("--model", default=None, help="LLM model name override.")
    parser.add_argument("--api-key", default=None, help="API key override for openai-compatible providers.")
    parser.add_argument("--base-url", default=None, help="Base URL override for openai-compatible providers.")
    parser.add_argument("--model-type", choices=["transformer", "cnn"], default="transformer",
                        help="Model family for real-mode mapping configs (BERT/MobileViT -> transformer).")
    parser.add_argument("--nn-model-factory", default=None,
                        help="Real-mode 'module:function' that builds the ONN/nn model.")
    parser.add_argument("--workload", default=None,
                        help="Analytic-mode workload name from configs/workloads.json "
                             "(e.g. synthetic / vit_tiny / vit_small / vit_base). "
                             "Defaults to the config's 'default' entry.")
    parser.add_argument("--list-workloads", action="store_true",
                        help="List the registered analytic workloads and exit.")
    parser.add_argument("--il-budget", type=float, default=None,
                        help="Optical insertion-loss budget (dB) for the link-budget "
                             "penalty; designs whose longest optical path exceeds it are "
                             "softly penalized.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable the LLM request/response trace under <output-root>/trace.")
    return parser


def _llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    config = LLMConfig(provider="paratera")
    if args.llm_config is not None and Path(args.llm_config).exists():
        payload = read_json(args.llm_config)
        config = LLMConfig(**{key: payload[key] for key in payload if key in LLMConfig.__dataclass_fields__})
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key
    if args.base_url:
        config.base_url = args.base_url
    return config


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    simphony_root = Path(args.simphony_root).resolve()
    devicelib_root = Path(args.devicelib_root).resolve() if args.devicelib_root else simphony_root / "configs" / "devices"
    if not devicelib_root.exists():
        print(f"error: device library not found at {devicelib_root}", file=sys.stderr)
        return 2

    if args.list_workloads:
        workload_config = load_workload_configs(simphony_root=str(simphony_root))
        default_name = workload_config.get("default")
        for name in sorted((workload_config.get("workloads") or {})):
            marker = " (default)" if name == default_name else ""
            print(f"{name}{marker}")
        return 0

    initial_hardware = materialize_hardware(
        read_json(args.initial_hardware) if args.initial_hardware and Path(args.initial_hardware).exists() else None
    )
    search_space: Dict[str, Any] = {}
    if args.search_space and Path(args.search_space).exists():
        search_space = read_json(args.search_space)

    llm_client = create_llm_client(_llm_config_from_args(args))
    adapter = SimphonyAdapter(
        simphony_root=simphony_root,
        output_root=args.output_root,
        search_space=search_space,
        mode=args.mode,
        model_type=args.model_type,
        nn_model_factory=args.nn_model_factory,
        workload_name=args.workload,
    )

    progress = None if args.quiet else ProgressReporter(enabled=True)
    tuner = MultiAgentTuner(
        adapter=adapter,
        llm_client=llm_client,
        devicelib_root=str(devicelib_root),
        output_root=args.output_root,
        search_space=search_space,
        trace_enabled=not args.no_trace,
    )

    summary = tuner.tune(
        initial_hardware=initial_hardware,
        max_iterations=args.iterations,
        trajectory_logging=True,
        progress=progress,
        il_budget=args.il_budget,
    )

    print("\n=== Photonic DSE summary ===")
    print(json.dumps(
        {
            "iterations": summary["iterations"],
            "best": summary["best"],
            "best_hardware": summary["best_hardware"],
        },
        indent=2,
        ensure_ascii=False,
    ))
    if summary.get("trace_dir"):
        print(f"\n[trace] every LLM request/response is under {summary['trace_dir']}/  "
              f"(raw_json/, view_md/, prompt_calls.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
