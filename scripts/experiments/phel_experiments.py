"""PHEL-Agent 的两组对比实验：等预算对比 与 同目标收敛对比。

两组实验各自在**一个文件夹内落两个文件**（数据 + 大模型运行记录），便于论文
直接引用与复现审计：

::

    runs/exp1_equal_budget/
        experiment_data.json      # 指标、改进幅度、收敛曲线、逐方法明细
        llm_run_records.json      # LLM 调用级 provenance（tokens/耗时/阶段/调度）

    runs/exp2_convergence/
        experiment_data.json
        llm_run_records.json

## 为什么是这两个实验

**实验一（等预算）**回答"同样的 SimPhony 求值次数下，谁把指标改得更多"。
横轴统一为求值次数：agent 的 ``--iterations N`` 与基线的 ``--budget N`` 都是
"1 次初始求值 + N 次提案"，引擎、搜索空间、目标函数、动作空间、keep-best
规则全部相同（见 ``baselines/README.md`` 的公平性契约）。

**实验二（同目标收敛）**回答"达到同一优化目标，谁需要的求值更少"。这里必须
固定一个**外生目标**再数求值，否则用 agent 自己的最好点做目标会循环论证。
默认目标为"相对初始点改善 ``--target``"（默认 30%），也可用 ``--target-objective``
直接给绝对值。

## 用法

::

    # 两组实验都跑：5 个基线 x 3 seed，agent 10 轮 x 3 seed
    .venv-agent/bin/python scripts/experiments/phel_experiments.py all \\
        --workload mobilevit --nn-model-factory tests.factory:build_mobilevit \\
        --iterations 10 --seeds 0,1,2

    # 只汇总已有产物（不重跑）——实验中途被打断后用它重建两个文件
    .venv-agent/bin/python scripts/experiments/phel_experiments.py all --collect-only

    # 只跑基线（agent 侧需要 LLM 凭据）
    .venv-agent/bin/python scripts/experiments/phel_experiments.py all --skip-agent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from photon_tuner.common.io import read_json, write_json  # noqa: E402

BASELINE_ALGOS = ["rs", "sa", "bo", "ga", "sweep"]


# --------------------------------------------------------------------------- #
# 运行层：把 agent / 基线分别跑起来
# --------------------------------------------------------------------------- #


def _run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return proc.returncode


def run_agent(args, out_dir: Path) -> None:
    """跑 agent 侧：每个 seed 一次独立的 tune 运行。"""
    for seed in args.seed_list:
        run_dir = out_dir / "agent" / f"seed_{seed}"
        if (run_dir / "summary.json").is_file() and not args.force:
            print(f"[agent] seed {seed}: 已有产物，跳过（--force 可重跑）")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "main.py"),
            "--simphony-root", str(ROOT / "photon_tuner" / "simulators" / "simphony"),
            "--workload", args.workload,
            "--model-type", args.model_type,
            "--iterations", str(args.iterations),
            "--output-root", str(run_dir),
            "--schedule-strategy", args.schedule_strategy,
        ]
        if args.nn_model_factory:
            cmd += ["--nn-model-factory", args.nn_model_factory]
        if args.il_budget is not None:
            cmd += ["--il-budget", str(args.il_budget)]
        if args.llm_provider:
            cmd += ["--provider", args.llm_provider]
        if args.llm_model:
            cmd += ["--model", args.llm_model]
        if args.llm_base_url:
            # configs/llm.json 里的 base_url 优先级高于 provider 预设；不显式覆盖，
            # 官方 key 会被发到代理端点并得到 401。
            cmd += ["--base-url", args.llm_base_url]
        print(f"[agent] seed {seed} -> {' '.join(cmd)}", flush=True)
        code = _run(cmd, run_dir / "run.log")
        if code != 0:
            print(f"[agent] seed {seed} 失败（exit {code}），见 {run_dir/'run.log'}", flush=True)


def run_baselines(args, out_dir: Path) -> None:
    """跑传统基线：一次调用覆盖所有算法/seed（runner 自己按目录跳过已完成项）。"""
    cmd = [
        sys.executable, "-m", "baselines.runner",
        "--algo", args.algos,
        "--workload", args.workload,
        "--budget", str(args.iterations),
        "--seeds", ",".join(str(s) for s in args.seed_list),
        "--model-type", args.model_type,
        "--output-root", str(out_dir / "baselines"),
    ]
    if args.nn_model_factory:
        cmd += ["--nn-model-factory", args.nn_model_factory]
    if args.il_budget is not None:
        cmd += ["--il-budget", str(args.il_budget)]
    if args.force:
        cmd += ["--force"]
    print(f"[baselines] {' '.join(cmd)}", flush=True)
    code = _run(cmd, out_dir / "baselines.log")
    if code != 0:
        print(f"[baselines] 失败（exit {code}），见 {out_dir/'baselines.log'}", flush=True)


# --------------------------------------------------------------------------- #
# 汇总层：从产物重建两个文件
# --------------------------------------------------------------------------- #


def _resolve_agent_run(seed_dir: Path) -> Optional[Path]:
    """定位 agent 一次运行的产物目录。

    产物在 ``seed_N/run_iteration_XXX/`` 下（多一层运行序号），因此不能直接
    在 seed 目录里找 ``trajectory.json``。多次运行取最新的一份。
    """
    candidates = sorted(seed_dir.glob("run_iteration_*/trajectory.json"))
    if not candidates:
        direct = seed_dir / "trajectory.json"
        return seed_dir if direct.is_file() else None
    return candidates[-1].parent


def _agent_curve(seed_dir: Path) -> Optional[Dict[str, Any]]:
    """从一次 agent 运行重建 (初始点, best-so-far 曲线, 明细)。"""
    run_dir = _resolve_agent_run(seed_dir)
    if run_dir is None:
        return None
    traj_path = run_dir / "trajectory.json"
    summary_path = run_dir / "summary.json"
    if not traj_path.is_file():
        return None
    traj = read_json(traj_path)
    summary = read_json(summary_path) if summary_path.is_file() else {}
    history = traj.get("history") or []
    if not history:
        return None
    initial = history[0]["metrics"]
    curve: List[float] = []
    best = float("inf")
    for item in history:
        value = float(item["metrics"]["objective"])
        best = min(best, value)
        curve.append(best)
    # history[0] 是初始基线求值；提案从 index 1 开始。
    return {
        "method": "agent",
        "initial_objective": float(initial["objective"]),
        "best_objective": best,
        "objective_curve": curve,
        "evaluations": len(history),
        "proposals": max(len(history) - 1, 0),
        "best_metrics": {
            key: traj.get("best", {}).get(key) for key in
            ("objective", "latency_ms", "energy_mj", "area_mm2", "fJ_per_mac",
             "core_insertion_loss")
        },
        "schedule_summary": summary.get("schedule_summary"),
        "schedule_history": summary.get("schedule_history"),
        "rework_total": summary.get("rework_total"),
        "run_dir": str(run_dir),
    }


def _baseline_curves(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*/seed_*/trajectory.json")):
        traj = read_json(path)
        curve = traj.get("objective_curve") or []
        if not curve:
            continue
        rows.append({
            "method": traj.get("method", path.parents[2].name),
            "seed": traj.get("seed"),
            "workload": traj.get("workload"),
            "initial_objective": float(traj["initial_metrics"]["objective"]),
            "best_objective": float(min(curve)),
            "objective_curve": [float(v) for v in curve],
            "evaluations": len(curve),
            "proposals": max(len(curve) - 1, 0),
            "wall_s": traj.get("wall_s"),
            "n_calls": traj.get("n_calls"),
            "best_metrics": {
                key: (traj.get("best") or {}).get(key) for key in
                ("objective", "latency_ms", "energy_mj", "area_mm2", "fJ_per_mac",
                 "core_insertion_loss")
            },
            "run_dir": str(path.parent),
        })
    return rows


def _llm_records(agent_root: Path) -> Dict[str, Any]:
    """聚合 LLM 调用级记录（来自各轮 ``llm_trace.jsonl`` 的 llm_end 事件）。"""
    per_seed: List[Dict[str, Any]] = []
    for seed_dir in sorted(p for p in agent_root.glob("seed_*") if p.is_dir()):
        stage_totals: Dict[str, Dict[str, float]] = {}
        calls: List[Dict[str, Any]] = []
        for trace in sorted(seed_dir.rglob("iter_*/analysis/llm_trace.jsonl")):
            stage = trace.parent.parent.name  # iter_XXX
            for line in trace.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "llm_end":
                    continue
                usage = record.get("token_usage") or {}
                entry = {
                    "iteration": stage,
                    "duration_s": record.get("duration_s"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    "model": ((usage.get("quota") or {}).get("model")),
                }
                calls.append(entry)
                bucket = stage_totals.setdefault(
                    stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0, "duration_s": 0.0}
                )
                bucket["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    bucket[key] += int(entry.get(key) or 0)
                bucket["duration_s"] += float(entry.get("duration_s") or 0.0)

        resolved = _resolve_agent_run(seed_dir) or seed_dir
        summary_path = resolved / "summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        totals = {
            "calls": len(calls),
            "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
            "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in calls),
            "total_tokens": sum(int(c.get("total_tokens") or 0) for c in calls),
            "reasoning_tokens": sum(int(c.get("reasoning_tokens") or 0) for c in calls),
            "llm_duration_s": round(sum(float(c.get("duration_s") or 0.0) for c in calls), 3),
        }
        per_seed.append({
            "seed": seed_dir.name.replace("seed_", ""),
            "run_dir": str(resolved),
            "totals": totals,
            "per_iteration": stage_totals,
            "stage_totals": (summary.get("metrics") or {}).get("stages"),
            "run_totals": (summary.get("metrics") or {}).get("totals"),
            "schedule_summary": summary.get("schedule_summary"),
            "schedule_history": summary.get("schedule_history"),
            "scheduler_config": summary.get("scheduler_config"),
            "search_strategy": summary.get("search_strategy"),
            "stall_s": None,
            "calls": calls,
        })
    return {
        "note": (
            "每条记录对应一次 LLM 往返（langchain llm_end 事件），粒度比 summary.json "
            "的 `calls` 更细：后者统计的是角色阶段调用次数，而一个 ReAct 阶段内部会有多次"
            "模型往返（每次工具调用一次）。因此本文件的 calls 总是大于等于阶段调用数。"
            "prompt_tokens 含工具结果的完整上下文；reasoning_tokens 为推理模型内部的思考 "
            "token（计入 completion_tokens）。"
        ),
        "seeds": per_seed,
    }


def _mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按方法聚合多个 seed：均值/标准差/最好值，以及平均 best-so-far 曲线。"""
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    table: List[Dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        improvements = [
            (item["initial_objective"] - item["best_objective"]) / item["initial_objective"] * 100
            for item in items
            if item["initial_objective"] > 0
        ]
        # 曲线按公共前缀长度对齐后逐点取均值。
        shortest = min((len(item["objective_curve"]) for item in items), default=0)
        mean_curve = [
            sum(item["objective_curve"][i] for item in items) / len(items)
            for i in range(shortest)
        ] if shortest else []
        modes = [item.get("schedule_summary") or {} for item in items]
        table.append({
            "method": method,
            "runs": len(items),
            "schedule_mode_counts": (
                modes[0].get("mode_counts") if method == "agent" and modes else None
            ),
            "initial_objective": _mean([item["initial_objective"] for item in items]),
            "best_objective_mean": _mean([item["best_objective"] for item in items]),
            "best_objective_best": min(item["best_objective"] for item in items),
            "improvement_pct_mean": _mean(improvements),
            "improvement_pct_min": min(improvements) if improvements else None,
            "improvement_pct_max": max(improvements) if improvements else None,
            "evaluations": _mean([item["evaluations"] for item in items]),
            "mean_curve": mean_curve,
            "best_metrics_mean": {
                key: _mean([item["best_metrics"].get(key) for item in items])
                for key in ("latency_ms", "energy_mj", "area_mm2", "fJ_per_mac",
                            "core_insertion_loss")
            },
        })
    return table


def _evals_to_target(curve: List[float], initial: float, target_pct: float,
                     target_objective: Optional[float]) -> Optional[int]:
    """返回达到目标所需的求值序号（1-based）；未达到返回 ``None``。"""
    target = target_objective if target_objective is not None else initial * (1 - target_pct)
    for index, value in enumerate(curve, start=1):
        if value <= target:
            return index
    return None


def _convergence_table(rows: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """多目标梯度下的"达标所需评估次数"。

    单一目标（默认 30%）在修正后的目标函数上**所有方法都达不到**——10 次提案的最好
    改进约 14%。用一个全员不达标的目标做收敛对比没有信息量，因此按梯度统计
    （默认 3% / 5% / 10%），并额外给出"各方法自身最好成绩出现在第几次评估"。
    """
    targets = list(args.targets) if getattr(args, "targets", None) else [0.03, 0.05, 0.10]
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    table: List[Dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        modes = [item.get("schedule_summary") or {} for item in items]
        entry: Dict[str, Any] = {
            "method": method,
            "runs": len(items),
            "schedule_mode_counts": (
                modes[0].get("mode_counts") if method == "agent" and modes else None
            ),
        }
        for target in targets:
            reached = [
                _evals_to_target(item["objective_curve"], item["initial_objective"],
                                 target, args.target_objective)
                for item in items
            ]
            hits = [r for r in reached if r is not None]
            key = f"evals_to_{target * 100:g}pct"
            entry[f"{key}_mean"] = _mean(hits) if hits else None
            entry[f"{key}_worst"] = max(hits) if hits else None
            entry[f"{key}_reached_runs"] = len(hits)
            entry[f"{key}_values"] = reached
        best_indices = []
        for item in items:
            curve = item["objective_curve"]
            if not curve:
                continue
            best_indices.append(curve.index(min(curve)) + 1)
        entry["evals_to_own_best_mean"] = _mean(best_indices) if best_indices else None
        table.append(entry)
    return table


def _relative_reduction(
    table: List[Dict[str, Any]],
    reference: str = "agent",
    key: str = "evals_to_10pct_mean",
) -> List[Dict[str, Any]]:
    """以 ``reference`` 为基准，算其他方法相对它多用/少用多少次求值。

    ``key`` 指定在哪个达标准入线上比较；正数 = 智能体更省。
    """
    base = next((row for row in table if row["method"] == reference), None)
    base_evals = (base or {}).get(key)
    rows: List[Dict[str, Any]] = []
    for row in table:
        if row["method"] == reference:
            continue
        other = row.get(key)
        if not base_evals or not other:
            rows.append({"method": row["method"], "vs_agent_reduction_pct": None})
            continue
        rows.append({
            "method": row["method"],
            "vs_agent_reduction_pct": (other - base_evals) / other * 100,
            "agent_evals": base_evals,
            "method_evals": other,
            "criterion": key,
        })
    return rows


def _markdown_report(data: Dict[str, Any], kind: str) -> str:
    """把实验结果渲染成可直接贴进论文/周报的 Markdown。"""
    meta = data["meta"]
    lines = [
        f"# {'实验一：等预算对比' if kind == 'equal-budget' else '实验二：同目标收敛对比'}",
        "",
        f"- 工作负载：`{meta['workload']}`",
        f"- 预算：{meta['iterations']} 次提案（1 次初始求值 + N 次提案，所有方法一致）",
        f"- seed：{meta['seeds']}",
        f"- 目标：`latency_ms × fJ_per_mac × area_mm2`（含 IL 预算软惩罚，越小越好）",
        f"- 生成时间：{meta['generated_at']}",
        "",
    ]
    if kind == "equal-budget":
        lines += [
            "各方法的 SimPhony 求值次数可能略有差异：agent 若某一轮产出不合法的结构化输出，"
            "该轮按设计被作废（不消耗仿真），因此可能少于预算。表格专门列出实际评估次数，"
            "不把它藏起来。",
            "",
            "| 方法 | 运行数 | 评估次数 | 初始目标 | 最优目标(均值) | 改进(均值) | 改进范围 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in data["methods"]:
            if row.get("improvement_pct_mean") is None:
                continue
            rng = f"{row['improvement_pct_min']:.1f}% ~ {row['improvement_pct_max']:.1f}%"
            lines.append(
                f"| {row['method']} | {row['runs']} | {row['evaluations']:.0f} | "
                f"{row['initial_objective']:.1f} | "
                f"{row['best_objective_mean']:.1f} | {row['improvement_pct_mean']:.1f}% | {rng} |"
            )
    else:
        targets = meta.get("convergence_targets") or []
        lines.append(
            "达标线（相对初始点的改进幅度）按梯度统计；单一高目标在预算内无人达到，"
            "因此分档报告。"
        )
        lines.append("")
        header = "| 方法 | 运行数 | " + " | ".join(
            f"达到 {t * 100:g}% 的评估数(均值/最差)" for t in targets
        ) + " | 自身最优出现的评估数 |"
        lines.append(header)
        lines.append("|---|" + "---:|" * (len(targets) + 2))
        for row in data.get("convergence", []):
            cells = []
            for t in targets:
                key = f"evals_to_{t * 100:g}pct"
                mean = row.get(f"{key}_mean")
                worst = row.get(f"{key}_worst")
                cells.append("—" if mean is None else f"{mean:.1f} / {worst}")
            own = row.get("evals_to_own_best_mean")
            lines.append(
                f"| {row['method']} | {row['runs']} | " + " | ".join(cells)
                + f" | {'—' if own is None else f'{own:.1f}'} |"
            )
        rel = [r for r in data.get("relative_to_agent", [])
               if r.get("vs_agent_reduction_pct") is not None]
        if rel:
            lines += ["", f"相对智能体的求值节省（口径：{rel[0].get('criterion')}，正 = 智能体更省）：", ""]
            lines += ["| 对比方法 | 智能体求值 | 该方法求值 | 节省 |", "|---|---:|---:|---:|"]
            for row in rel:
                lines.append(
                    f"| {row['method']} | {row['agent_evals']:.1f} | {row['method_evals']:.1f} | "
                    f"{row['vs_agent_reduction_pct']:+.1f}% |"
                )
    return "\n".join(lines) + "\n"


def _meta(args, kind: str) -> Dict[str, Any]:
    return {
        "experiment": kind,
        "workload": args.workload,
        "iterations": args.iterations,
        "seeds": args.seed_list,
        "algos": args.algos,
        "nn_model_factory": args.nn_model_factory,
        "model_type": args.model_type,
        "il_budget": args.il_budget,
        "schedule_strategy": args.schedule_strategy,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "target_improvement_pct": args.target,
        "convergence_targets": list(args.targets),
        "target_objective": args.target_objective,
        "artifact_source": str(_artifact_source(args, kind)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": str(ROOT),
    }


def _artifact_source(args, key: str) -> Path:
    """实验二复用实验一的运行产物（同一批 run，两种读法），避免重复烧算力。"""
    if key == "convergence" and not args.recompute_convergence:
        candidate = args.out_root / FOLDERS["equal-budget"]
        if (candidate / "agent").is_dir() or (candidate / "baselines").is_dir():
            return candidate
    return args.out_root / FOLDERS[key]


def collect(args, out_dir: Path, kind: str, source: Optional[Path] = None) -> Dict[str, Any]:
    """重建 ``experiment_data.json`` 与 ``llm_run_records.json``。

    ``source`` 是运行产物的实际位置；它可能与输出目录不同（实验二复用实验一的
    run），此时产物仍是同一批，只是换一种口径统计。
    """
    source = source or out_dir
    agent_rows = []
    for seed_dir in sorted(p for p in (source / "agent").glob("seed_*") if p.is_dir()):
        row = _agent_curve(seed_dir)
        if row is not None:
            agent_rows.append(row)
    baseline_rows = _baseline_curves(source / "baselines") if (source / "baselines").is_dir() else []
    rows = agent_rows + baseline_rows

    table = _aggregate(rows)
    data: Dict[str, Any] = {
        "meta": _meta(args, kind),
        "fairness_contract": {
            "same_engine": "SimPhony real (ONNArchSimulator) via MCP",
            "same_search_space": "configs/search_space.json + configs/initial_hardware.json",
            "same_objective": "latency_ms * fJ_per_mac * area_mm2 (x IL budget penalty)",
            "budget_axis": "SimPhony evaluation count (1 initial + N proposals)",
        },
        "methods": table,
        "runs": rows,
    }
    if kind == "convergence":
        convergence = _convergence_table(rows, args)
        data["convergence"] = convergence
        data["relative_to_agent"] = _relative_reduction(convergence)
    write_json(out_dir / "experiment_data.json", data)

    llm = _llm_records(source / "agent")
    llm["meta"] = _meta(args, kind)
    write_json(out_dir / "llm_run_records.json", llm)
    # 人类可读摘要与两个机器可读文件放在同一目录，便于引用时三者一致。
    (out_dir / "summary.md").write_text(_markdown_report(data, kind), encoding="utf-8")
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["equal-budget", "convergence", "all"],
                        help="Which experiment to run. 'all' runs both and writes each "
                             "into its own folder.")
    parser.add_argument("--workload", default="mobilevit")
    parser.add_argument("--nn-model-factory", default="tests.factory:build_mobilevit")
    parser.add_argument("--model-type", choices=["transformer", "cnn"], default="transformer")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Agent iterations; baselines get the same proposal budget.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--algos", default=",".join(BASELINE_ALGOS))
    parser.add_argument("--il-budget", type=float, default=10.0)
    parser.add_argument("--schedule-strategy", choices=["rag", "fixed"], default="rag")
    parser.add_argument("--llm-provider", default="deepseek-official",
                        help="Agent-side LLM provider preset (see photon_tuner/llm/model.py).")
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com/v1",
                        help="Overrides configs/llm.json's base_url (which outranks "
                             "provider presets).")
    parser.add_argument("--target", type=float, default=0.30,
                        help="Legacy single convergence target (kept for the meta block).")
    parser.add_argument("--targets", default="0.03,0.05,0.10",
                        help="Comma-separated convergence targets as fractions of the "
                             "initial objective. A single ambitious target is useless when "
                             "no method reaches it within budget.")
    parser.add_argument("--target-objective", type=float, default=None,
                        help="Absolute target objective (overrides --target).")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--skip-agent", action="store_true", help="Only run/collect baselines.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--collect-only", action="store_true",
                        help="Do not execute anything; rebuild the two files from artifacts.")
    parser.add_argument("--force", action="store_true", help="Re-run existing results.")
    parser.add_argument("--recompute-convergence", action="store_true",
                        help="Run a separate campaign for the convergence experiment "
                             "instead of re-reading the equal-budget run artifacts.")
    args = parser.parse_args(argv)
    args.seed_list = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    args.targets = [float(t) for t in str(args.targets).split(",") if t.strip()]
    args.algos = ",".join(a.strip() for a in str(args.algos).split(",") if a.strip())
    return args


FOLDERS = {
    "equal-budget": "exp1_equal_budget",
    "convergence": "exp2_convergence",
}


def _run_one(args, key: str) -> None:
    out_dir = args.out_root / FOLDERS[key]
    out_dir.mkdir(parents=True, exist_ok=True)
    # 收敛实验默认复用实验一的运行产物（同一批 run，只换统计口径），
    # 因此不再重复烧一遍算力。
    reusing = _artifact_source(args, key) != out_dir
    if not args.collect_only and not reusing:
        if not args.skip_baselines:
            run_baselines(args, out_dir)
        if not args.skip_agent:
            run_agent(args, out_dir)
    data = collect(args, out_dir, key, source=_artifact_source(args, key))
    methods = ", ".join(
        f"{row['method']}={row['improvement_pct_mean']:.1f}%"
        for row in data["methods"] if row.get("improvement_pct_mean") is not None
    )
    print(f"[{key}] {out_dir/'experiment_data.json'}")
    print(f"[{key}] improvement: {methods or '(no data yet)'}")
    if key == "convergence" and data.get("relative_to_agent"):
        print(f"[{key}] evals-to-target vs agent: {data['relative_to_agent']}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    keys = ["equal-budget", "convergence"] if args.experiment == "all" else [args.experiment]
    for key in keys:
        _run_one(args, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
