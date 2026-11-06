"""传统算法基线的命令行入口。

仿真固定走 **real** 模式（真实 ONNArchSimulator，经 MCP 桥在 Python-3.10 venv 中运行），
与 ``photon_tuner.cli`` 的 agent 侧完全一致——两者必须同引擎，对比才有意义。

用法示例：:

    # 单个算法、单个 workload、50 次提案（总 51 次求值）
    .venv-agent/bin/python -m baselines.runner \\
        --algo bo --workload vit_tiny --budget 50 --seeds 0 \\
        --output-root runs/run_baseline

    # 四个主基线 x 三个 seed，一次跑完
    .venv-agent/bin/python -m baselines.runner \\
        --algo rs,sa,bo,ga --workload vit_tiny,vit_base \\
        --budget 50 --seeds 0,1,2

产物目录（与 ``photon_tuner`` 的 ``trajectory.json`` 兼容、字段为其超集）::

    <output-root>/<algo>/<workload>/seed_<N>/{trajectory.json, summary.json, iter_*/}

算法参数用 ``--algo-option rs:key=value,...`` 透传（见各算法模块 docstring）。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photon_tuner.common.io import read_json
from photon_tuner.common.paths import SIMPHONY_ROOT
from photon_tuner.core.search_space import (
    build_design_space,
    discover_device_catalog,
    materialize_hardware,
)

from baselines.action_space import ActionSpace
from baselines.algorithms import PRIMARY_ALGORITHMS, REGISTRY, build_optimizer
from baselines.evaluator import SimphonyBaselineEvaluator
from baselines.harness import run_search


def parse_algo_options(specs: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """解析 ``rs:key=value,rs:key2=v2,bo:xi=0.02`` 为按算法分组的选项。"""
    out: Dict[str, Dict[str, Any]] = {}
    for spec in (specs or []):
        if ":" not in spec:
            continue
        algo, kv = spec.split(":", 1)
        if algo not in REGISTRY or "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        out.setdefault(algo, {})[key] = _coerce(value)
    return out


def _coerce(value: str) -> Any:
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baselines.runner", description="Run traditional DSE baselines against SimPhony.")
    p.add_argument("--algo", default="rs,sa,bo,ga", help="comma-separated algorithm names: %s (plus sweep)" % ",".join(sorted(REGISTRY)))
    p.add_argument("--workload", default="vit_tiny", help="comma-separated workload names (configs/workloads.json); default matches photon_tuner DEFAULT_WORKLOAD")
    p.add_argument("--budget", type=int, default=50, help="proposal budget per run (total evals = budget + 1)")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated integer seeds")
    p.add_argument("--simphony-root", type=Path, default=SIMPHONY_ROOT)
    p.add_argument("--initial-hardware", type=Path, default=None)
    p.add_argument("--search-space", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("runs/run_baseline"))
    p.add_argument("--model-type", choices=["transformer", "cnn"], default="transformer")
    p.add_argument("--nn-model-factory", default=None, help="ONN model factory (module:function)")
    p.add_argument("--state-strategy", choices=["best", "latest"], default="best")
    p.add_argument("--il-budget", type=float, default=None)
    p.add_argument("--algo-option", action="append", default=None, help="algo:key=value passthrough (repeatable)")
    p.add_argument("--force", action="store_true", help="re-run even if trajectory.json already exists")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    algos = [a.strip() for a in args.algo.split(",") if a.strip()]
    workloads = [w.strip() for w in args.workload.split(",") if w.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    algo_options = parse_algo_options(args.algo_option)

    search_space: Dict[str, Any] = read_json(args.search_space) if args.search_space else {}
    initial_hw = read_json(args.initial_hardware) if args.initial_hardware else None

    root = args.output_root
    manifest: List[Dict[str, Any]] = []

    for algo in algos:
        for workload in workloads:
            for seed in seeds:
                run_dir = root / algo / workload / f"seed_{seed}"
                if (run_dir / "trajectory.json").exists() and not args.force:
                    if not args.quiet:
                        print(f"== {algo} / {workload} / seed {seed} -> skip (exists; use --force to re-run)")
                    continue
                if not args.quiet:
                    print(f"== {algo} / {workload} / seed {seed} -> {run_dir}")

                device_catalog = discover_device_catalog(f"{args.simphony_root}/configs/devices")
                space = build_design_space(search_space=search_space, device_catalog=device_catalog)
                if args.il_budget is not None:
                    search_space = dict(search_space)
                    search_space["il_budget"] = float(args.il_budget)

                # MCP 桥的环境变量：让 Python-3.10 侧的 simphony server 用同一个
                # 工作负载/模型工厂，与 agent 侧（photon_tuner.cli）完全一致。
                mcp_env: Dict[str, str] = {}
                if workload:
                    mcp_env["SIMPHONY_WORKLOAD"] = workload
                if args.nn_model_factory:
                    mcp_env["SIMPHONY_NN_FACTORY"] = args.nn_model_factory
                mcp_env["SIMPHONY_MODEL_TYPE"] = args.model_type

                evaluator = SimphonyBaselineEvaluator(
                    simphony_root=args.simphony_root,
                    output_root=run_dir,
                    mode="real",
                    transport="mcp",
                    mcp_env=mcp_env,
                    workload_name=workload,
                    search_space=search_space,
                    design_space=space,
                    model_type=args.model_type,
                    nn_model_factory=args.nn_model_factory,
                )
                try:
                    action_space = ActionSpace(space, device_catalog, mode="real")
                    rng = random.Random(seed)
                    optimizer = build_optimizer(
                        algo, action_space, rng,
                        seed=seed,
                        **(algo_options.get(algo, {})),
                    )
                    init = materialize_hardware(initial_hw)
                    result = run_search(
                        method=algo,
                        optimizer=optimizer,
                        initial_hardware=init,
                        budget=args.budget,
                        evaluator=evaluator,
                        run_dir=run_dir,
                        state_strategy=args.state_strategy,
                        options={"seed": seed, "budget": args.budget},
                        verbose=not args.quiet,
                    )
                    manifest.append({
                        "algo": algo, "workload": workload, "seed": seed,
                        "initial_objective": result.initial_objective,
                        "best_objective": result.best_objective,
                        "improvement": result.improvement,
                        "n_calls": result.n_calls, "n_unique": result.n_unique,
                        "wall_s": result.wall_s,
                    })
                finally:
                    evaluator.close()

    if manifest:
        from photon_tuner.common.io import write_json
        write_json(root / "manifest.json", manifest)
    if not args.quiet:
        print("\n=== baselines summary ===")
        for row in manifest:
            print(f"{row['algo']:6s} {row['workload']:12s} seed={row['seed']}  "
                  f"best={row['best_objective']:.4e}  improvement={row['improvement']*100:.1f}%  "
                  f"evals={row['n_unique']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())