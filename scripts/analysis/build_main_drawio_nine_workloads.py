#!/usr/bin/env python3
"""Build the editable 3x3 main-experiment convergence figure in Draw.io XML."""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/fyq/epic")
SOURCE = ROOT / "paper_statistics_20261117/01_main/main_all_workloads_all_methods_per_round.csv"
OUTPUT = ROOT / ".scratch/main_experiment_figure1.drawio"

WORKLOADS = [
    ("vit_tiny", "ViT-T"), ("vit_small", "ViT-S"), ("vit_base", "ViT-B"),
    ("mobilevit_xxs", "MobileViT-XXS"), ("mobilevit_xs", "MobileViT-XS"),
    ("mobilevit_s", "MobileViT-S"), ("cnn_tiny", "CNN-T"),
    ("cnn_small", "CNN-S"), ("cnn_base", "CNN-B"),
]
METHODS = [
    ("agent", "OPEarch-Agent", "#C62828", 2.2),
    ("rs", "Random Search", "#1565C0", 1.5),
    ("sa", "Simulated Annealing", "#2E7D32", 1.5),
    ("bo", "Bayesian Optimization", "#EF6C00", 1.5),
    ("ga", "Genetic Algorithm", "#6A1B9A", 1.5),
    ("sweep", "Parameter Sweep", "#546E7A", 1.5),
]


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


def main() -> None:
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            iteration = int(row["iteration"])
            if iteration in range(5, 51, 5):
                data[row["workload"]][row["method"]][iteration] = float(row["improvement_pct"])

    cells: list[str] = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    text_style = "text;html=1;strokeColor=none;fillColor=none;fontColor=#222222;"

    # Shared legend across the top.
    legend_x = 65
    for index, (method, label, color, width) in enumerate(METHODS):
        x = legend_x + index * 165
        cells.append(edge(f"legendLine_{method}", x, 24, x + 28, 24,
                          f"endArrow=none;html=1;strokeColor={color};strokeWidth={width};"))
        cells.append(vertex(f"legendPoint_{method}", "", x + 12, 22, 4, 4,
                            f"ellipse;html=1;aspect=fixed;fillColor={color};strokeColor={color};"))
        cells.append(vertex(f"legendText_{method}", label, x + 33, 13, 132, 21,
                            text_style + "align=left;verticalAlign=middle;fontSize=11;"))

    panel_w, panel_h = 350, 215
    start_x, start_y = 12, 43
    plot_w, plot_h = 272, 139
    left_pad, top_pad = 56, 27

    for panel_index, (workload, title) in enumerate(WORKLOADS):
        row, col = divmod(panel_index, 3)
        panel_x = start_x + col * panel_w
        panel_y = start_y + row * panel_h
        x0 = panel_x + left_pad
        y0 = panel_y + top_pad
        y_bottom = y0 + plot_h

        cells.append(vertex(f"title_{workload}", f"({chr(97 + panel_index)}) {title}",
                            panel_x + 60, panel_y, 232, 23,
                            text_style + "align=center;verticalAlign=middle;fontSize=13;fontStyle=1;"))

        # Horizontal grid lines, y ticks, and y labels from 0% to 100%.
        for pct in range(0, 101, 10):
            y = y_bottom - plot_h * pct / 100.0
            if pct not in (0, 100):
                cells.append(edge(f"grid_{workload}_{pct}", x0, y, x0 + plot_w, y,
                                  "endArrow=none;html=1;strokeColor=#E5E7EB;strokeWidth=0.6;dashed=1;"))
            cells.append(edge(f"ytick_{workload}_{pct}", x0 - 4, y, x0, y,
                              "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1;"))
            cells.append(vertex(f"ylabel_{workload}_{pct}", str(pct), x0 - 37, y - 8, 29, 16,
                                text_style + "align=right;verticalAlign=middle;fontSize=10;"))

        # Axes.
        cells.append(edge(f"yaxis_{workload}", x0, y0, x0, y_bottom,
                          "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1.3;"))
        cells.append(edge(f"xaxis_{workload}", x0, y_bottom, x0 + plot_w, y_bottom,
                          "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1.3;"))

        # X ticks and labels: 5, 10, ..., 50, exactly equally spaced.
        for iteration in range(5, 51, 5):
            x = x0 + plot_w * iteration / 50.0
            cells.append(edge(f"xtick_{workload}_{iteration}", x, y_bottom, x, y_bottom + 4,
                              "endArrow=none;html=1;strokeColor=#36393D;strokeWidth=1;"))
            cells.append(vertex(f"xlabel_{workload}_{iteration}", str(iteration), x - 11,
                                y_bottom + 4, 22, 16,
                                text_style + "align=center;verticalAlign=top;fontSize=10;"))

        # Axis captions.
        cells.append(vertex(f"xcaption_{workload}", "迭代次数", x0 + 84, y_bottom + 19, 104, 18,
                            text_style + "align=center;verticalAlign=middle;fontSize=11;"))
        cells.append(vertex(f"ycaption_{workload}", "Improvement (%)", panel_x - 2,
                            y0 + 29, 25, 88,
                            text_style + "align=center;verticalAlign=middle;fontSize=11;rotation=270;"))

        # Curves with a marker at every sampled evaluation.
        for method, _, color, width in METHODS:
            points: list[tuple[float, float, int]] = []
            for iteration in range(5, 51, 5):
                improvement = max(0.0, min(100.0, data[workload][method][iteration]))
                x = x0 + plot_w * iteration / 50.0
                y = y_bottom - plot_h * improvement / 100.0
                points.append((x, y, iteration))
            for seg, ((x1, y1, _), (x2, y2, _)) in enumerate(zip(points, points[1:])):
                cells.append(edge(f"curve_{workload}_{method}_{seg}", x1, y1, x2, y2,
                                  f"endArrow=none;html=1;strokeColor={color};strokeWidth={width};"))
            for x, y, iteration in points:
                size = 4.8 if method == "agent" else 4.0
                cells.append(vertex(f"point_{workload}_{method}_{iteration}", "",
                                    x - size / 2, y - size / 2, size, size,
                                    f"ellipse;html=1;aspect=fixed;fillColor={color};strokeColor=#FFFFFF;strokeWidth=0.5;"))

    xml = (
        '<mxfile host="app.diagrams.net"><diagram name="第 1 页" id="main-nine-workloads">'
        '<mxGraphModel dx="1070" dy="700" grid="0" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1070" '
        'pageHeight="700" math="0" shadow="0"><root>'
        + "".join(cells)
        + '</root></mxGraphModel></diagram></mxfile>\n'
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(xml, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
