#!/usr/bin/env python3
"""Build the grouped main bar chart and the three-curve ablation chart."""

from __future__ import annotations

import html
import json
from itertools import accumulate
from math import floor, isfinite
from pathlib import Path
from statistics import mean


ROOT = Path("/home/fyq/epic")
MAIN_ROOT = ROOT / "runs/paper_experiments_20261116/01_main_budget_atomic_formal_eff5_focus15"
MAIN_BUDGET = 50
OUT_MAIN = ROOT / ".scratch/main_experiment_figure2.drawio"
OUT_ABL = ROOT / ".scratch/ablation_line_chart.drawio"
MAIN_TRAJ = MAIN_ROOT / "agent_mature3_w10_d10/mobilevit_s/seed_0/run_iteration_001/trajectory.json"
ABL_ROOT = ROOT / "runs/paper_experiments_20261117/04_ablations_deepseek_official/mobilevit_s"

METHODS = [
    ("agent", "OPEarch-Agent", "#C62828"),
    ("rs", "Random Search", "#1565C0"),
    ("sa", "Simulated Annealing", "#2E7D32"),
    ("bo", "Bayesian Optimization", "#EF6C00"),
    ("ga", "Genetic Algorithm", "#6A1B9A"),
    ("sweep", "Parameter Sweep", "#546E7A"),
]
WORKLOAD_FAMILIES = {
    "ViT": ("vit_tiny", "vit_small", "vit_base"),
    "MobileViT": ("mobilevit_xxs", "mobilevit_xs", "mobilevit_s"),
    "CNN": ("cnn_tiny", "cnn_small", "cnn_base"),
}


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def vertex(cell_id: str, value: str, x: float, y: float, w: float, h: float, style: str) -> str:
    return (
        f'<mxCell id="{esc(cell_id)}" value="{esc(value)}" vertex="1" parent="1" '
        f'style="{esc(style)}"><mxGeometry x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" as="geometry" /></mxCell>'
    )


def edge(cell_id: str, x1: float, y1: float, x2: float, y2: float, style: str) -> str:
    return (
        f'<mxCell id="{esc(cell_id)}" edge="1" parent="1" style="{esc(style)}" value="">'
        f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.2f}" y="{y1:.2f}" '
        f'as="sourcePoint" /><mxPoint x="{x2:.2f}" y="{y2:.2f}" '
        f'as="targetPoint" /></mxGeometry></mxCell>'
    )


def document(cells: list[str], width: int, height: int, diagram_id: str) -> str:
    return (
        f'<mxfile host="app.diagrams.net"><diagram name="第 1 页" id="{diagram_id}">'
        f'<mxGraphModel dx="{width}" dy="{height}" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{width}" pageHeight="{height}" math="0" shadow="0"><root>'
        + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>\n'
    )


def main_first_hits() -> tuple[int, dict[tuple[str, str], int]]:
    family_curves = {}
    for family, workloads in WORKLOAD_FAMILIES.items():
        workload_curves = {method: [] for method, _, _ in METHODS}
        for workload in workloads:
            initials = set()
            for method, _, _ in METHODS:
                if method == "agent":
                    path = MAIN_ROOT / "agent_mature3_w10_d10" / workload
                    path /= "seed_0/run_iteration_001/trajectory.json"
                else:
                    path = MAIN_ROOT / "baselines" / method / workload / "seed_0/trajectory.json"
                history = json.loads(path.read_text(encoding="utf-8"))["history"]
                if len(history) < MAIN_BUDGET + 1 or int(history[0]["iteration"]) != 0:
                    raise ValueError(f"Incomplete trajectory within comparison budget: {path}")

                # History order tracks search proposals. Cache hits can reuse earlier iteration IDs.
                values = [float(row["metrics"]["objective"]) for row in history[:MAIN_BUDGET + 1]]
                initial = values[0]
                if not isfinite(initial) or initial <= 0:
                    raise ValueError(f"Invalid initial objective: {path}")
                initials.add(initial)
                # Failed simulations carry +inf, which leaves the best-so-far unchanged.
                improvement = [100 * (initial - best) / initial for best in accumulate(values, min)]
                workload_curves[method].append(improvement)
            if len(initials) != 1:
                raise ValueError(f"Methods used different initial objectives for {workload}")
        for method, curves in workload_curves.items():
            family_curves[family, method] = [mean(round_values) for round_values in zip(*curves)]

    # Every monotonic family curve can reach this whole-percentage threshold.
    threshold = floor(min(curve[-1] for curve in family_curves.values()))
    if threshold < 1:
        raise ValueError("No positive whole-percentage threshold was reached by every family curve")
    hits = {
        key: next(iteration for iteration, improvement in enumerate(curve) if improvement >= threshold)
        for key, curve in family_curves.items()
    }
    return threshold, hits


def build_main_bar() -> None:
    threshold, hits = main_first_hits()
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    text = "text;html=1;strokeColor=none;fillColor=none;fontColor=#222222;"
    page_w, page_h = 1050, 470
    x0, y0, plot_w, plot_h = 90, 112, 920, 272
    y_bottom = y0 + plot_h
    y_max = ((max(hits.values()) + 4) // 5) * 5

    cells.append(vertex("title", f"首次达到 {threshold}% 平均改善所需的迭代次数",
                        0, 15, page_w, 32,
                        text + "align=center;verticalAlign=middle;fontSize=20;fontStyle=1;"))
    for idx, (method, label, color) in enumerate(METHODS):
        x = 46 + idx * 164
        cells.append(vertex(f"legend_{method}", "", x, 62, 14, 14,
                            f"rounded=0;html=1;fillColor={color};strokeColor={color};"))
        cells.append(vertex(f"legend_text_{method}", label, x + 19, 57, 144, 24,
                            text + "align=left;verticalAlign=middle;fontSize=12;"))

    for value in range(0, y_max + 1, 5):
        y = y_bottom - plot_h * value / y_max
        if value:
            cells.append(edge(f"grid_{value}", x0, y, x0 + plot_w, y,
                              "endArrow=none;html=1;strokeColor=#D9DDE3;strokeWidth=0.7;dashed=1;"))
        cells.append(edge(f"ytick_{value}", x0 - 5, y, x0, y,
                          "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1;"))
        cells.append(vertex(f"ylabel_{value}", str(value), x0 - 42, y - 9, 32, 18,
                            text + "align=right;verticalAlign=middle;fontSize=11;"))
    cells.append(edge("xaxis", x0, y_bottom, x0 + plot_w, y_bottom,
                      "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1.5;"))
    cells.append(edge("yaxis", x0, y0, x0, y_bottom,
                      "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1.5;"))
    cells.append(vertex("ycaption", "迭代次数", 0, y0 + plot_h / 2 - 13, 62, 26,
                        text + "align=center;verticalAlign=middle;fontSize=13;rotation=270;"))

    group_w = plot_w / len(WORKLOAD_FAMILIES)
    bar_w, gap = 30, 7
    bars_total = len(METHODS) * bar_w + (len(METHODS) - 1) * gap
    for idx, family in enumerate(WORKLOAD_FAMILIES):
        center = x0 + group_w * (idx + 0.5)
        start = center - bars_total / 2
        for mi, (method, _, color) in enumerate(METHODS):
            value = hits[family, method]
            bx = start + mi * (bar_w + gap)
            bh = plot_h * value / y_max
            cells.append(vertex(f"bar_{family}_{method}", "", bx, y_bottom - bh, bar_w, bh,
                                f"rounded=0;html=1;fillColor={color};strokeColor={color};"))
            cells.append(vertex(f"value_{family}_{method}", str(value), bx,
                                y_bottom - bh - 20, bar_w, 18,
                                text + "align=center;verticalAlign=middle;fontSize=13;fontStyle=1;"))
        cells.append(vertex(f"family_{family}", family, center - 80, y_bottom + 10, 160, 25,
                            text + "align=center;verticalAlign=middle;fontSize=15;fontStyle=1;"))

    note = "每类三种规模，按各自初始目标归一化后逐轮等权平均。迭代次数按提案计。"
    cells.append(vertex("normalization_note", note, 90, 430, 920, 20,
                        text + "align=center;verticalAlign=middle;fontSize=12;fontColor=#4B5563;"))
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAIN.write_text(document(cells, page_w, page_h, "main-bar-family"), encoding="utf-8")


def metric_values(root: Path, count: int) -> list[float]:
    return [float(json.loads((root / f"iter_{i:03d}/metrics.json").read_text())["objective"])
            for i in range(count + 1)]


def continued_values(method: str, prefix_rounds: int, initial: float) -> list[float]:
    prefix = metric_values(ABL_ROOT / method / "run_iteration_001", prefix_rounds)
    continuation = metric_values(
        ABL_ROOT / "continuations_to_50" / f"{method}_from{prefix_rounds}" / "run_iteration_001",
        MAIN_BUDGET - prefix_rounds,
    )
    if prefix[0] != initial or continuation[0] != min(prefix):
        raise ValueError(f"Initial or resumed baseline does not match for {method}")
    return prefix[1:] + continuation[1:]


def visible_segment(a: float, b: float, low: float, high: float) -> tuple[float, float] | None:
    """Return the fractions of a line segment that lie in a visible y-axis interval."""
    if max(a, b) < low or min(a, b) > high:
        return None
    if low <= a <= high and low <= b <= high:
        return 0.0, 1.0
    first, last = sorted(((low - a) / (b - a), (high - a) / (b - a)))
    return max(0.0, first), min(1.0, last)


def build_ablation_line() -> None:
    history = json.loads(MAIN_TRAJ.read_text(encoding="utf-8"))["history"]
    if [int(row["iteration"]) for row in history] != list(range(MAIN_BUDGET + 1)):
        raise ValueError(f"Expected rounds 0–{MAIN_BUDGET} in {MAIN_TRAJ}")
    initial = float(history[0]["metrics"]["objective"])
    values = {
        "full": [float(row["metrics"]["objective"]) for row in history[1:]],
        "norag": continued_values("no_rag", 33, initial),
        "single": continued_values("single_agent", 27, initial),
    }
    curves = {
        key: [100.0 * (initial - value) / initial for value in objectives]
        for key, objectives in values.items()
    }
    styles = [
        ("full", "完整系统", "#C62828", "ellipse", 9),
        ("norag", "无RAG", "#1565C0", "rectangle", 7),
        ("single", "单智能体", "#2E7D32", "rhombus", 7),
    ]
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    points = []
    text = "text;html=1;strokeColor=none;fillColor=none;fontColor=#222222;verticalAlign=middle;"
    axis = "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1.4;"
    page_w, page_h = 1100, 660
    x0, plot_w = 104, 940
    plot_panels = [
        (-175, 100, 113, 347, (-150, -100, -50, 0, 50, 100)),
        (-1700, -1600, 478, 76, (-1700, -1650, -1600)),
    ]
    y_bottom = plot_panels[-1][2] + plot_panels[-1][3]
    x_step = plot_w / MAIN_BUDGET
    cells.append(vertex("title", "MobileViT-S 消融实验逐轮对比", 0, 18, page_w, 32,
                        text + "align=center;fontSize=20;fontStyle=1;"))
    for idx, (key, label, color, shape, size) in enumerate(styles):
        x = 345 + idx * 148
        marker = f"shape={shape};html=1;fillColor={color};strokeColor=#FFFFFF;strokeWidth=0.6;"
        cells.append(edge(f"legend_line_{key}", x, 69, x + 35, 69,
                          f"endArrow=none;html=1;strokeColor={color};strokeWidth=1.7;"))
        cells.append(vertex(f"legend_point_{key}", "", x + (35 - size) / 2, 69 - size / 2,
                            size, size, marker))
        cells.append(vertex(f"legend_text_{key}", label, x + 42, 58, 95, 22,
                            text + "align=left;fontSize=13;"))
    cells.append(vertex("ycaption", "相对初始变化（%）", x0 - 57, 85, 180, 20,
                        text + "align=left;fontSize=13;"))

    for panel, (low, high, y_top, height, ticks) in enumerate(plot_panels):
        scale = height / (high - low)
        bottom = y_top + height
        for pct in ticks:
            y = bottom - (pct - low) * scale
            grid = "endArrow=none;html=1;strokeColor=#D9DDE3;strokeWidth=0.7;dashed=1;"
            if pct == 0:
                grid = "endArrow=none;html=1;strokeColor=#A6AFBB;strokeWidth=1.1;dashed=1;"
            cells.append(edge(f"grid_{pct}", x0, y, x0 + plot_w, y, grid))
            cells.append(edge(f"ytick_{pct}", x0 - 5, y, x0, y, axis))
            cells.append(vertex(f"ylabel_{pct}", str(pct), x0 - 58, y - 9, 47, 18,
                                text + "align=right;fontSize=12;"))
        cells.append(edge(f"yaxis_{panel}", x0, y_top, x0, bottom, axis))
        for key, _, color, shape, size in styles:
            line = f"endArrow=none;html=1;strokeColor={color};strokeWidth=1.7;"
            marker = f"shape={shape};html=1;fillColor={color};strokeColor=#FFFFFF;strokeWidth=0.6;"
            curve = curves[key]
            for iteration, (a, b) in enumerate(zip(curve, curve[1:]), start=1):
                interval = visible_segment(a, b, low, high)
                if interval is None:
                    continue
                start, end = interval
                cells.append(edge(
                    f"curve_{key}_{iteration}_{panel}",
                    x0 + x_step * (iteration + start), bottom - (a + (b - a) * start - low) * scale,
                    x0 + x_step * (iteration + end), bottom - (a + (b - a) * end - low) * scale,
                    line,
                ))
            for iteration, value in enumerate(curve, start=1):
                if low <= value <= high:
                    x = x0 + iteration * x_step
                    y = bottom - (value - low) * scale
                    points.append(vertex(f"point_{key}_{iteration}", "", x - size / 2, y - size / 2,
                                         size, size, marker))

    if len(points) != MAIN_BUDGET * len(styles):
        raise ValueError("Ablation measurements fall outside the displayed vertical-axis intervals")
    cells.append(edge("break_upper", x0 - 5, 468, x0 + 5, 461, axis))
    cells.append(edge("break_lower", x0 - 5, 477, x0 + 5, 470, axis))
    cells.append(edge("xaxis", x0, y_bottom, x0 + plot_w, y_bottom, axis))
    for iteration in range(0, MAIN_BUDGET + 1, 10):
        x = x0 + x_step * iteration
        cells.append(edge(f"xtick_{iteration}", x, y_bottom, x, y_bottom + 5,
                          axis))
        cells.append(vertex(f"xlabel_{iteration}", str(iteration), x - 16, y_bottom + 8, 32, 20,
                            text + "align=center;fontSize=12;"))
    cells.extend(points)
    cells.append(vertex("xcaption", "迭代次数", x0 + (plot_w - 120) / 2, 586, 120, 25,
                        text + "align=center;fontSize=13;"))
    note = "变化率 =（初始目标 − 当轮目标）/ 初始目标 × 100%。纵轴跳过 −1600% 至 −175% 的空区间。"
    cells.append(vertex("change_note", note, x0, 627, plot_w, 20,
                        text + "align=center;fontSize=12;fontColor=#4B5563;"))
    OUT_ABL.write_text(document(cells, page_w, page_h, "ablation-line-per-round"), encoding="utf-8")


def main() -> None:
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    build_main_bar()
    build_ablation_line()
    print(OUT_MAIN)
    print(OUT_ABL)


if __name__ == "__main__":
    main()
