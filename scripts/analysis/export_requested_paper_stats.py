#!/usr/bin/env python3
"""Export traceable CSV/JSON statistics requested for the paper figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/fyq/epic")
MAIN = ROOT / "runs/paper_experiments_20261116/01_main_budget_atomic_formal_eff5_focus15"
ABL = ROOT / "runs/paper_experiments_20261117/04_ablations_deepseek_official/mobilevit_s"
MODELS = ROOT / "runs/paper_experiments_20261117/05_model_comparison/mobilevit_s"
OUT = ROOT / "paper_statistics_20261117"
WORKLOADS = [
    "vit_tiny", "vit_small", "vit_base",
    "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
    "cnn_tiny", "cnn_small", "cnn_base",
]
ALGOS = ["rs", "sa", "bo", "ga", "sweep"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_curve(path: Path, budget: int = 50) -> list[float]:
    history = load(path)["history"]
    values = {int(row["iteration"]): float(row["metrics"]["objective"]) for row in history}
    best = values[min(values)]
    curve = []
    for iteration in range(0, budget + 1):
        if iteration in values:
            best = min(best, values[iteration])
        curve.append(best)
    return curve


def trajectory_raw(path: Path, budget: int = 50) -> dict[int, float]:
    return {
        int(row["iteration"]): float(row["metrics"]["objective"])
        for row in load(path)["history"]
        if int(row["iteration"]) <= budget
    }


def metric_curve(root: Path, budget: int) -> list[float]:
    values: dict[int, float] = {}
    for metric in root.glob("iter_*/metrics.json"):
        iteration = int(metric.parent.name.split("_")[-1])
        if iteration <= budget:
            values[iteration] = float(load(metric)["objective"])
    if 0 not in values:
        raise RuntimeError(f"missing baseline metrics in {root}")
    curve = []
    best = values[0]
    for iteration in range(0, budget + 1):
        if iteration in values:
            best = min(best, values[iteration])
        curve.append(best)
    return curve


def first_hit(curve: list[float], threshold: float) -> int | None:
    for iteration, value in enumerate(curve[1:], start=1):
        if value <= threshold:
            return iteration
    return None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_stats() -> dict:
    curves: dict[str, dict[str, list[float]]] = {}
    final_rows = []
    threshold_rows = []
    matched_rows = []
    per_round_rows = []
    multi_threshold_rows = []
    threshold_50_wide_rows = []
    curve_rows = []
    for workload in WORKLOADS:
        agent_path = MAIN / "agent_mature3_w10_d10" / workload / "seed_0/run_iteration_001/trajectory.json"
        method_paths = {"agent": agent_path}
        methods = {"agent": trajectory_curve(agent_path)}
        for algo in ALGOS:
            method_paths[algo] = MAIN / "baselines" / algo / workload / "seed_0/trajectory.json"
            methods[algo] = trajectory_curve(method_paths[algo])
        curves[workload] = methods
        for method, curve in methods.items():
            initial = curve[0]
            raw = trajectory_raw(method_paths[method])
            for iteration, best_value in enumerate(curve):
                per_round_rows.append({
                    "workload": workload,
                    "method": method,
                    "iteration": iteration,
                    "initial_objective": initial,
                    "evaluated_objective": raw.get(iteration),
                    "best_objective": best_value,
                    "normalized_best_objective": best_value / initial,
                    "improvement_pct": 100.0 * (initial - best_value) / initial,
                })
            for improvement_threshold in [10, 20, 30, 40, 50, 60, 70]:
                objective_threshold = initial * (1.0 - improvement_threshold / 100.0)
                multi_threshold_rows.append({
                    "workload": workload,
                    "method": method,
                    "improvement_threshold_pct": improvement_threshold,
                    "objective_threshold": objective_threshold,
                    "first_hit_iteration": first_hit(curve, objective_threshold),
                })

        threshold_50_wide_rows.append({
            "workload": workload,
            "improvement_threshold_pct": 50,
            "objective_threshold": methods["agent"][0] * 0.5,
            **{
                f"{method}_first_hit_iteration": first_hit(methods[method], methods[method][0] * 0.5)
                for method in ["agent", *ALGOS]
            },
        })
        final = {"workload": workload, **{name: curve[50] for name, curve in methods.items()}}
        best_algo = min(ALGOS, key=lambda name: methods[name][50])
        final["best_traditional_method"] = best_algo
        final["best_traditional_objective"] = methods[best_algo][50]
        final_rows.append(final)

        threshold = 0.5 * methods["agent"][0]
        agent_hit = first_hit(methods["agent"], threshold)
        trad_hits = {name: first_hit(methods[name], threshold) for name in ALGOS}
        reached = {name: hit for name, hit in trad_hits.items() if hit is not None}
        best_trad_method = min(reached, key=reached.get) if reached else None
        best_trad_hit = reached.get(best_trad_method) if best_trad_method else None
        saving = None
        if agent_hit is not None and best_trad_hit is not None:
            saving = 100.0 * (best_trad_hit - agent_hit) / best_trad_hit
        threshold_rows.append({
            "workload": workload,
            "absolute_threshold": threshold,
            "normalized_threshold": 0.5,
            "agent_first_hit_iteration": agent_hit,
            "best_traditional_method": best_trad_method,
            "best_traditional_first_hit_iteration": best_trad_hit,
            "relative_iteration_reduction_pct": saving,
        })

        # A second, endpoint-aware target: the better quality that both the
        # agent and the best final traditional run can actually attain within
        # 50 evaluations.  This is useful for a matched-quality speedup bar.
        matched_threshold = max(methods["agent"][50], methods[best_algo][50])
        matched_agent_hit = first_hit(methods["agent"], matched_threshold)
        matched_trad_hit = first_hit(methods[best_algo], matched_threshold)
        matched_saving = 100.0 * (matched_trad_hit - matched_agent_hit) / matched_trad_hit
        matched_rows.append({
            "workload": workload,
            "matched_quality_threshold": matched_threshold,
            "agent_first_hit_iteration": matched_agent_hit,
            "best_traditional_method": best_algo,
            "best_traditional_first_hit_iteration": matched_trad_hit,
            "signed_agent_iteration_reduction_pct": matched_saving,
        })

    for iteration in range(0, 51):
        row = {"iteration": iteration}
        for method in ["agent", *ALGOS]:
            normalized = [curves[w][method][iteration] / curves[w][method][0] for w in WORKLOADS]
            row[f"{method}_mean_normalized_best"] = sum(normalized) / len(normalized)
        curve_rows.append(row)

    write_csv(
        OUT / "01_main/main_final_objective_at_50.csv", final_rows,
        ["workload", "agent", *ALGOS, "best_traditional_method", "best_traditional_objective"],
    )
    write_csv(
        OUT / "01_main/main_all_workloads_all_methods_per_round.csv", per_round_rows,
        ["workload", "method", "iteration", "initial_objective", "evaluated_objective",
         "best_objective", "normalized_best_objective", "improvement_pct"],
    )
    write_csv(
        OUT / "01_main/main_first_hit_iterations_by_improvement_threshold.csv", multi_threshold_rows,
        ["workload", "method", "improvement_threshold_pct", "objective_threshold",
         "first_hit_iteration"],
    )
    write_csv(
        OUT / "01_main/main_first_hit_iterations_at_50pct_wide.csv", threshold_50_wide_rows,
        ["workload", "improvement_threshold_pct", "objective_threshold",
         *[f"{method}_first_hit_iteration" for method in ["agent", *ALGOS]]],
    )
    write_csv(
        OUT / "01_main/main_mean_normalized_curve_0_50.csv", curve_rows,
        ["iteration", *[f"{m}_mean_normalized_best" for m in ["agent", *ALGOS]]],
    )
    write_csv(
        OUT / "01_main/main_threshold_50pct_iteration_reduction.csv", threshold_rows,
        ["workload", "absolute_threshold", "normalized_threshold", "agent_first_hit_iteration",
         "best_traditional_method", "best_traditional_first_hit_iteration",
         "relative_iteration_reduction_pct"],
    )
    write_csv(
        OUT / "01_main/main_matched_quality_iteration_reduction.csv", matched_rows,
        ["workload", "matched_quality_threshold", "agent_first_hit_iteration",
         "best_traditional_method", "best_traditional_first_hit_iteration",
         "signed_agent_iteration_reduction_pct"],
    )
    return {"final": final_rows, "fixed_threshold": threshold_rows, "matched_quality": matched_rows}


def ablation_stats() -> list[dict]:
    curves = {
        "full_system": trajectory_curve(
            MAIN / "agent_mature3_w10_d10/mobilevit_s/seed_0/run_iteration_001/trajectory.json"
        ),
        "no_rag": metric_curve(ABL / "no_rag/run_iteration_001", 33),
        "single_agent": metric_curve(ABL / "single_agent/run_iteration_001", 27),
    }
    budgets = {"full_system": 50, "no_rag": 33, "single_agent": 27}
    rows = []
    for method, curve in curves.items():
        hit = first_hit(curve, 200000.0)
        rows.append({
            "method": method,
            "effective_evaluations": budgets[method],
            "threshold": 200000.0,
            "first_hit_iteration": hit,
            "best_objective_within_budget": min(curve),
        })
    full_hit = rows[0]["first_hit_iteration"]
    for row in rows:
        hit = row["first_hit_iteration"]
        row["full_system_iteration_reduction_vs_method_pct"] = (
            0.0 if row["method"] == "full_system" else 100.0 * (hit - full_hit) / hit
        )
    write_csv(
        OUT / "02_ablation/mobilevit_s_threshold_200000.csv", rows,
        ["method", "effective_evaluations", "threshold", "first_hit_iteration",
         "best_objective_within_budget", "full_system_iteration_reduction_vs_method_pct"],
    )
    return rows


def model_stats() -> list[dict]:
    curves = {
        "deepseek_official_flash": trajectory_curve(
            MAIN / "agent_mature3_w10_d10/mobilevit_s/seed_0/run_iteration_001/trajectory.json", 10
        ),
        "cloudmist_glm_5.3_flash": metric_curve(
            MODELS / "cloudmist_glm-5.3-flash/run_iteration_001", 10
        ),
        "cloudmist_gpt_6.1_sol": metric_curve(
            MODELS / "cloudmist_gpt-6.1-sol/run_iteration_001", 10
        ),
    }
    rows = []
    for iteration in range(0, 11):
        for model, curve in curves.items():
            rows.append({
                "model": model,
                "iteration": iteration,
                "best_objective": curve[iteration],
                "normalized_best_objective": curve[iteration] / curve[0],
            })
    write_csv(
        OUT / "03_model_comparison/mobilevit_s_first_10_evaluations.csv", rows,
        ["model", "iteration", "best_objective", "normalized_best_objective"],
    )
    summary = []
    for model, curve in curves.items():
        summary.append({
            "model": model,
            "evaluation_budget": 10,
            "initial_objective": curve[0],
            "best_objective_at_10": curve[10],
            "improvement_pct": 100.0 * (curve[0] - curve[10]) / curve[0],
        })
    write_csv(
        OUT / "03_model_comparison/mobilevit_s_summary_at_10.csv", summary,
        ["model", "evaluation_budget", "initial_objective", "best_objective_at_10", "improvement_pct"],
    )
    return summary


def draw_main_figures() -> None:
    import matplotlib.pyplot as plt

    curve_rows = list(csv.DictReader(
        (OUT / "01_main/main_mean_normalized_curve_0_50.csv").open(encoding="utf-8-sig")
    ))
    labels = {
        "agent": "OPEarch-Agent", "rs": "Random Search", "sa": "Simulated Annealing",
        "bo": "Bayesian Optimization", "ga": "Genetic Algorithm", "sweep": "Parameter Sweep",
    }
    colors = {
        "agent": "#c62828", "rs": "#1565c0", "sa": "#2e7d32",
        "bo": "#ef6c00", "ga": "#6a1b9a", "sweep": "#546e7a",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = [int(row["iteration"]) for row in curve_rows]
    for method in ["agent", "rs", "sa", "bo", "ga", "sweep"]:
        y = [float(row[f"{method}_mean_normalized_best"]) for row in curve_rows]
        ax.plot(x, y, label=labels[method], color=colors[method],
                linewidth=4 if method == "agent" else 2)
    ax.set_xlabel("Simulation evaluations")
    ax.set_ylabel("Mean normalized best objective (lower is better)")
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "01_main/fig_main_convergence_50.png", dpi=300)
    fig.savefig(OUT / "01_main/fig_main_convergence_50.pdf")
    plt.close(fig)

    matched = list(csv.DictReader(
        (OUT / "01_main/main_matched_quality_iteration_reduction.csv").open(encoding="utf-8-sig")
    ))
    names = [row["workload"].replace("_", "-") for row in matched]
    savings = [float(row["signed_agent_iteration_reduction_pct"]) for row in matched]
    bar_colors = ["#2e7d32" if value >= 0 else "#c62828" for value in savings]
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    bars = ax.bar(names, savings, color=bar_colors, width=0.72)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Agent evaluation reduction at matched quality (%)")
    ax.set_xlabel("Workload")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (1.2 if value >= 0 else -1.2),
                f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "01_main/fig_main_matched_quality_iteration_reduction.png", dpi=300)
    fig.savefig(OUT / "01_main/fig_main_matched_quality_iteration_reduction.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "main_experiment": main_stats(),
        "official_deepseek_ablation": ablation_stats(),
        "model_comparison_first_10": model_stats(),
        "notes": {
            "main_baseline_seed": 0,
            "main_threshold_definition": "objective <= 0.5 * workload-specific initial objective",
            "ablation_threshold": 200000.0,
            "model_comparison_budget": 10,
        },
    }
    (OUT / "all_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_main_figures()
    print(OUT)


if __name__ == "__main__":
    main()
