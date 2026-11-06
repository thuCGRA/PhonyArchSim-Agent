"""对比报告：LLM agent（``photon_tuner`` 运行产物）vs 传统算法基线产物。

输入：

* agent 运行目录：``runs/run_agent/run_iteration_*``（各含 ``trajectory.json``）；
* 基线运行目录：``runs/run_baseline/<algo>/<workload>/seed_*/trajectory.json``。

输出（``--out`` 目录）：

* ``compare.md`` / ``compare.csv`` —— 每个 (workload, method) 的最佳目标、相对初始
  点的归一化目标、改进幅度、收敛所需的求值次数、以及最佳点的 latency/energy/area；
* ``curves.svg`` —— best-so-far 目标 vs 求值次数的收敛曲线（按 workload 分组）；
* ``bars.svg`` —— 最终归一化目标的柱状图（每 workload 一组）。

公平口径：横轴统一为 **SimPhony 求值次数**。传统基线 1 初始 + budget 次提案 =
``budget+1`` 次求值；agent 的 ``N`` 次迭代对应 ``history`` 长度（同样 1 初始 + N 提案）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from photon_tuner.common.io import read_json

from baselines.plotting import bar_chart_svg, line_chart_svg

AGENT_LABEL = "agent"
UNKNOWN_WORKLOAD = "unknown"


def _safe_slug(name: str) -> str:
    """把 workload 名转成可用作文件名的 slug（避免 ``<unknown>`` 这类字符）。"""
    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(name))
    return slug.strip("_") or UNKNOWN_WORKLOAD


def _read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return read_json(path)
    except Exception:
        return None


def _infer_workload(run_dir: Path, traj: Dict[str, Any]) -> Optional[str]:
    wl = traj.get("workload")
    if wl:
        return str(wl)
    # agent 运行：读 iter_000/workload.json。
    wl_path = run_dir / "iter_000" / "workload.json"
    data = _read_json_optional(wl_path)
    if data:
        name = data.get("name")
        if name is None and isinstance(data.get("workload"), dict):
            name = data["workload"].get("name")
        if name is not None:
            return str(name)
    return None


def _best_so_far(objectives: List[float]) -> List[float]:
    out: List[float] = []
    best = float("inf")
    for v in objectives:
        best = min(best, v)
        out.append(best)
    return out


def load_run(traj_path: Path) -> Optional[Dict[str, Any]]:
    """加载一次完整的搜索运行，统一 agent 与基线两种产物格式。"""
    traj = _read_json_optional(traj_path)
    if not traj:
        return None
    run_dir = traj_path.parent
    method = traj.get("method") or (AGENT_LABEL if "run_iteration" in run_dir.name else "?")
    workload = _infer_workload(run_dir, traj)

    history = traj.get("history") or []
    if history:
        metrics_list = [h.get("metrics", {}) for h in history]
        objectives = [float(m.get("objective", float("inf"))) for m in metrics_list]
        curve = _best_so_far(objectives)
        initial_metrics = metrics_list[0]
        n_evals = len(history)
    else:
        curve = list(traj.get("objective_curve") or [])
        objectives = list(curve)
        metrics_list = []
        initial_metrics = traj.get("initial_metrics") or {}
        n_evals = traj.get("n_calls") or len(curve)

    best = traj.get("best") or {}
    best_hardware = traj.get("best_hardware") or {}

    seed = traj.get("seed")
    return {
        "traj_path": str(traj_path),
        "run_dir": str(run_dir),
        "method": method,
        "workload": workload,
        "seed": seed if seed is not None else (None if method == AGENT_LABEL else 0),
        "best": best,
        "best_hardware": best_hardware,
        "objectives": objectives,
        "metrics_list": metrics_list,
        "initial_metrics": initial_metrics,
        "curve": curve,
        "n_evals": n_evals,
        "n_unique": traj.get("n_unique"),
    }


def _record_from_iterdirs(run_dir: Path, method: str = AGENT_LABEL) -> Optional[Dict[str, Any]]:
    """从 ``iter_*/metrics.json`` 重建一次运行（用于被中途截断、没有
    ``trajectory.json`` 的 agent 运行）。"""
    evals: List[tuple] = []
    for d in sorted(run_dir.glob("iter_*")):
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            index = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data = _read_json_optional(metrics_path)
        if data:
            evals.append((index, data))
    if not evals:
        return None
    evals.sort(key=lambda item: item[0])

    objectives = [float(m.get("objective", float("inf"))) for _, m in evals]
    curve = _best_so_far(objectives)
    best_pos = min(range(len(objectives)), key=lambda i: objectives[i])
    return {
        "traj_path": "",
        "run_dir": str(run_dir),
        "method": method,
        "workload": _infer_workload(run_dir, {}),
        "seed": None,
        "best": evals[best_pos][1],
        "best_hardware": {},
        "objectives": objectives,
        "metrics_list": [m for _, m in evals],
        "initial_metrics": evals[0][1],
        "curve": curve,
        "n_evals": len(evals),
        "n_unique": len(evals),
    }


def discover_agent_runs(patterns: List[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for pattern in patterns:
        for run_dir in sorted(glob.glob(str(pattern))):
            path = Path(run_dir)
            if not path.is_dir():
                continue
            traj_path = path / "trajectory.json"
            if traj_path.exists():
                rec = load_run(traj_path)
            else:
                rec = _record_from_iterdirs(path)
            if rec and rec["method"] == AGENT_LABEL:
                runs.append(rec)
    return runs


def discover_baseline_runs(root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for tpath in sorted(root.rglob("trajectory.json")):
        rec = load_run(tpath)
        if rec and rec["method"] != AGENT_LABEL:
            runs.append(rec)
    return runs


def _metric(metrics: Dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _truncate(rec: Dict[str, Any], at_budget: int) -> Dict[str, Any]:
    """把一次运行截断到前 ``at_budget`` 次求值（同预算公平对比）。"""
    objectives = rec.get("objectives") or []
    metrics_list = rec.get("metrics_list") or []
    n = min(at_budget, len(objectives)) if objectives else 0
    if n <= 0:
        return rec
    objectives = objectives[:n]
    curve = _best_so_far(objectives)
    best_pos = min(range(n), key=lambda i: objectives[i])
    best = metrics_list[best_pos] if best_pos < len(metrics_list) else rec.get("best", {})
    out = dict(rec)
    out.update({
        "objectives": objectives,
        "metrics_list": metrics_list[:n],
        "curve": curve,
        "best": best,
        "n_evals": n,
        "n_unique": min(n, int(rec.get("n_unique") or n)),
    })
    return out


def write_report(
    runs: List[Dict[str, Any]],
    out_dir: Path,
    at_budget: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按 workload 分组的聚合。
    by_workload: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in runs:
        wl = rec["workload"] or UNKNOWN_WORKLOAD
        by_workload[wl][rec["method"]].append(rec)

    rows: List[Dict[str, Any]] = []
    curves_by_workload: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    bars: Dict[str, Dict[str, Any]] = {}

    for wl, methods in sorted(by_workload.items()):
        bars[wl] = {"categories": [], "series": []}
        for method in sorted(methods):
            recs = methods[method]
            # 同预算截断：只比较前 at_budget 次求值。
            if at_budget:
                recs = [_truncate(rec, at_budget) for rec in recs]
            # 以中位数种子为准（稳健），曲线取各 seed 的平均 best-so-far。
            recs_sorted = sorted(recs, key=lambda r: float(r["best"].get("objective", float("inf"))))
            median = recs_sorted[len(recs_sorted) // 2]
            init_obj = float(median["initial_metrics"].get("objective", float("inf")))
            best_obj = float(median["best"].get("objective", float("inf")))
            improvement = (init_obj - best_obj) / init_obj if init_obj and init_obj != float("inf") else 0.0

            rows.append({
                "workload": wl,
                "method": method,
                "n_runs": len(recs),
                "best_objective": best_obj,
                "normalized_objective": best_obj / init_obj if init_obj else float("inf"),
                "improvement_pct": improvement * 100.0,
                "initial_objective": init_obj,
                "n_evals": median.get("n_evals"),
                "n_unique": median.get("n_unique"),
                "latency_s": median["best"].get("latency_s", ""),
                "energy_pj": median["best"].get("energy_pj", ""),
                "area_um2": median["best"].get("area_um2", ""),
                "pJ_per_mac": median["best"].get("pJ_per_mac", ""),
            })

            # 收敛曲线（各 seed 平均，按求值序号对齐）。
            max_len = max(len(r["curve"]) for r in recs) if recs else 0
            mean_curve: List[float] = []
            for i in range(max_len):
                vals = [r["curve"][min(i, len(r["curve"]) - 1)] for r in recs if r["curve"]]
                if vals:
                    mean_curve.append(sum(vals) / len(vals))
            curves_by_workload[wl].append({
                "name": method,
                "x": list(range(len(mean_curve))),
                "y": mean_curve,
            })

            bars[wl]["series"].append({
                "name": method,
                "values": [best_obj / init_obj if init_obj else float("inf")],
            })

        bars[wl]["categories"] = ["normalized objective (lower is better)"]

    # 写 CSV。
    fieldnames = ["workload", "method", "n_runs", "best_objective", "normalized_objective",
                  "improvement_pct", "initial_objective", "n_evals", "n_unique",
                  "latency_s", "energy_pj", "area_um2", "pJ_per_mac"]
    with open(out_dir / "compare.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 写 Markdown 表。
    lines = ["# 传统算法基线 vs LLM 多智能体 agent 对比", ""]
    lines.append("> 归一化目标 = best_objective / initial_objective（<1 表示相对初始点改进，越低越好）。")
    lines.append("> 横轴统一为 SimPhony 求值次数；``n_runs`` 为随机种子数（agent 为运行目录数）。")
    lines.append("")
    lines.append("| workload | method | runs | initial obj | best obj | normalized | improvement | evals | latency_s | energy_pj | area_um2 | pJ/MAC |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['method']} | {row['n_runs']} | "
            f"{row['initial_objective']:.3g} | {row['best_objective']:.3g} | "
            f"{row['normalized_objective']:.3f} | {row['improvement_pct']:.1f}% | "
            f"{row['n_evals']} | {row['latency_s']} | {row['energy_pj']} | {row['area_um2']} | {row['pJ_per_mac']} |"
        )
    lines.append("")
    out_md = "\n".join(lines)
    (out_dir / "compare.md").write_text(out_md, encoding="utf-8")

    # 折线图（每 workload 一幅）。
    svg_paths: List[str] = []
    for wl, series in sorted(curves_by_workload.items()):
        p = line_chart_svg(
            series,
            str(out_dir / f"curves_{_safe_slug(wl)}.svg"),
            title=f"{wl}: best-so-far objective vs SimPhony evaluations",
            xlabel="SimPhony evaluations",
            ylabel="objective (lower is better)",
            log_y=True,
        )
        svg_paths.append(p)
    # 柱状图（每 workload 一幅）。
    for wl, data in sorted(bars.items()):
        p = bar_chart_svg(
            categories=data["categories"],
            series=data["series"],
            out_path=str(out_dir / f"bars_{_safe_slug(wl)}.svg"),
            title=f"{wl}: normalized objective (initial = 1.0)",
            ylabel="normalized objective",
        )
        svg_paths.append(p)

    (out_dir / "compare.json").write_text(
        json.dumps({"rows": rows, "svg": svg_paths}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"rows": rows, "md": str(out_dir / "compare.md"), "svg": svg_paths}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baselines.compare", description="Compare agent vs traditional baselines.")
    p.add_argument("--agent-runs", action="append", default=None,
                   help="glob pattern(s) of agent run dirs (e.g. 'runs/run_agent/run_iteration_*'); repeatable")
    p.add_argument("--baseline-root", type=Path, default=Path("runs/run_baseline"))
    p.add_argument("--at-budget", type=int, default=None,
                   help="truncate every run to its first N evaluations (like-for-like budget)")
    p.add_argument("--out", type=Path, default=Path("reports/compare"))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    agent_patterns = args.agent_runs or ["runs/run_iteration_*"]
    runs: List[Dict[str, Any]] = []
    runs.extend(discover_agent_runs(agent_patterns))
    if args.baseline_root.exists():
        runs.extend(discover_baseline_runs(args.baseline_root))

    if not runs:
        print(f"[compare] no runs found (agent patterns={agent_patterns}, baseline root={args.baseline_root})")
        return 1

    result = write_report(runs, args.out, at_budget=args.at_budget)
    print(f"[compare] wrote {len(result['rows'])} rows -> {result['md']}")
    if args.at_budget:
        print(f"[compare] truncated to first {args.at_budget} evaluations per run")
    for s in result["svg"]:
        print(f"[compare]   {s}")
    # 打印摘要表。
    print("\n=== comparison table ===")
    for row in result["rows"]:
        print(f"{row['workload']:14s} {row['method']:8s}  best={row['best_objective']:.4e}  "
              f"norm={row['normalized_objective']:.3f}  improvement={row['improvement_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())