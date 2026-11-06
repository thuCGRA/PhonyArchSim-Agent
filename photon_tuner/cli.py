"""基于 LangGraph 的光子 DSE 框架的命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import read_json
from photon_tuner.common.logging import configure_log_level
from photon_tuner.common.paths import SIMPHONY_ROOT
from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.workload import load_workload_configs
from photon_tuner.llm.config import LLMConfig
from photon_tuner.llm.model import build_chat_model
from photon_tuner.rag.vector_store import HistoryVectorStore
from photon_tuner.runtime.orchestrator import DeepAgentTuner
from photon_tuner.simulators.simphony import SimphonyAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photon-tuner",
        description="Deep-agents photonic accelerator design-space exploration on SimPhony.",
    )
    parser.add_argument("--initial-hardware", type=Path, default=None,
                        help="JSON file with the initial photonic hardware configuration.")
    parser.add_argument("--search-space", type=Path, default=None,
                        help="JSON file overriding the arch/device search-space candidates.")
    parser.add_argument("--simphony-root", type=Path, default=SIMPHONY_ROOT,
                        help="Root of the copied SimPhony project.")
    parser.add_argument("--devicelib-root", type=Path, default=None,
                        help="Device library root (defaults to <simphony-root>/configs/devices).")
    parser.add_argument("--output-root", type=Path, default=Path("runs/run_agent"),
                        help="Directory for per-iteration runs, artifacts, and the history DB.")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Number of tuning iterations to run.")
    parser.add_argument("--llm-config", type=Path, default=Path("configs/llm.json"),
                        help="JSON LLM configuration (provider/model/base_url). The API key is "
                             "not configured here: it comes from the OPENAI_API_KEY environment "
                             "variable (see configs/api_keys.py).")
    parser.add_argument("--provider", default=None,
                        help="LLM provider override (mock / paratera / deepseek / openai-compatible).")
    parser.add_argument("--model", default=None, help="LLM model name override.")
    parser.add_argument("--base-url", default=None, help="Base URL override.")
    parser.add_argument("--model-type", choices=["transformer", "cnn"], default="transformer",
                        help="Model family for real-mode mapping configs (BERT/MobileViT -> transformer).")
    parser.add_argument("--nn-model-factory", default=None,
                        help="'module:function' that builds the ONN/nn model for the simulator.")
    parser.add_argument("--workload", default=None,
                        help="Registered workload name from configs/workloads.json "
                             "(e.g. vit_tiny / vit_small / vit_base). "
                             "Defaults to the config's 'default' entry.")
    parser.add_argument("--list-workloads", action="store_true",
                        help="List the registered workloads and exit.")
    parser.add_argument("--il-budget", type=float, default=None,
                        help="Optical insertion-loss budget (dB) for the link-budget "
                             "penalty; designs whose longest optical path exceeds it are "
                             "softly penalized.")
    parser.add_argument("--state-strategy", choices=["best", "next"], default="best",
                        help="Which evaluated point to carry into the next iteration: "
                             "the best so far, or the latest proposal.")
    parser.add_argument("--search-strategy", choices=["joint", "sequential"], default="joint",
                        help="Co-search mode: joint (all layers every iteration) or "
                             "sequential (cycle architecture -> topology -> device).")
    parser.add_argument("--topology-sweep", action="store_true",
                        help="Evaluate every evaluable circuit template (core_kind) "
                             "up front and archive them on the Pareto frontier.")
    parser.add_argument("--no-gate", action="store_true",
                        help="Disable the evidence-evaluation gate.")
    parser.add_argument("--gate-retries", type=int, default=2,
                        help="Max re-propose attempts when the gate rejects a solution.")
    parser.add_argument("--no-trajectory", action="store_true",
                        help="Do not write trajectory.json.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging across the runtime "
                             "(equivalent to PHOTON_TUNER_LOG_LEVEL=DEBUG).")
    parser.add_argument("--serve", action="store_true",
                        help="Run the FastAPI HTTP server (POST /tune, GET /health) "
                             "instead of a one-shot tuning run.")
    parser.add_argument("--serve-host", default="0.0.0.0",
                        help="Host to bind when --serve is set.")
    parser.add_argument("--serve-port", type=int, default=8000,
                        help="Port to bind when --serve is set.")
    return parser


def _llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    config = LLMConfig(provider="paratera")
    if args.llm_config is not None and Path(args.llm_config).exists():
        payload = read_json(args.llm_config)
        if "api_key" in payload:
            print(f"warning: {args.llm_config} still carries an 'api_key' entry; it is ignored "
                  "now. Set OPENAI_API_KEY in the environment instead "
                  "(configs/api_keys.py).", file=sys.stderr)
        config = LLMConfig(**{key: payload[key] for key in payload if key in LLMConfig.__dataclass_fields__})
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    return config


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        configure_log_level("DEBUG")

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

    # 仿真固定走 real 模式：真实 ONNArchSimulator 在 Python-3.10 虚拟环境中运行，
    # 智能体侧（本进程，Python-3.14）通过 MCP 桥跨进程调用。analytic 镜像不再作为
    # 可选项——它只保留在模拟器库内部供离线实验与单元测试直接调用。
    mcp_env: Dict[str, str] = {}
    if args.workload:
        mcp_env["SIMPHONY_WORKLOAD"] = args.workload
    if args.nn_model_factory:
        mcp_env["SIMPHONY_NN_FACTORY"] = args.nn_model_factory
    mcp_env["SIMPHONY_MODEL_TYPE"] = args.model_type

    llm_config = _llm_config_from_args(args)
    if llm_config.provider == "mock":
        # 离线冒烟路径：确定性的工具调用模型，无网络。
        from photon_tuner.testing.fake_model import FakeToolCallingModel

        model = FakeToolCallingModel()
    else:
        model = build_chat_model(llm_config)
    adapter = SimphonyAdapter(
        simphony_root=simphony_root,
        output_root=args.output_root,
        search_space=search_space,
        mode="real",
        model_type=args.model_type,
        nn_model_factory=args.nn_model_factory,
        workload_name=args.workload,
        transport="mcp",
        mcp_env=mcp_env,
    )

    output_root = Path(args.output_root).resolve()
    store = HistoryVectorStore(output_root / "history.db")
    tuner = DeepAgentTuner(
        adapter=adapter,
        llm_config=llm_config,
        devicelib_root=str(devicelib_root),
        output_root=output_root,
        search_space=search_space,
        store=store,
        model=model,
        enable_gate=not args.no_gate,
        max_gate_retries=args.gate_retries,
    )

    if args.serve:
        from photon_tuner.server import create_app

        try:
            import uvicorn  # noqa: PLC0415
        except ImportError as exc:
            print("error: --serve requires uvicorn; run `pip install fastapi uvicorn`",
                  file=sys.stderr)
            return 2
        app = create_app(tuner, adapter=adapter, store=store)
        print(f"serving photonic-tuner on http://{args.serve_host}:{args.serve_port} "
              "(POST /tune, GET /health)", file=sys.stderr)
        uvicorn.run(app, host=args.serve_host, port=args.serve_port)
        return 0

    try:
        summary = tuner.tune(
            initial_hardware=initial_hardware,
            max_iterations=args.iterations,
            state_strategy=args.state_strategy,
            trajectory_logging=not args.no_trajectory,
            il_budget=args.il_budget,
            search_strategy=args.search_strategy,
            topology_sweep=args.topology_sweep,
        )
    finally:
        adapter.close()
        store.close()

    if not args.quiet:
        print("\n=== Photonic DSE summary ===")
        print(json.dumps(
            {
                "iterations": summary["iterations"],
                "best": summary["best"],
                "best_hardware": summary["best_hardware"],
                "run_dir": summary["run_dir"],
            },
            indent=2,
            ensure_ascii=False,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
