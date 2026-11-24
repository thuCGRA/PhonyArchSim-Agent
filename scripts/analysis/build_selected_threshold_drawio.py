#!/usr/bin/env python3
"""Draw the selected workloads' first hits at a fixed 40% improvement."""

from build_iteration_savings_drawio import normalized_curve
from build_main_bar_and_ablation_drawio import MAIN_BUDGET, MAIN_ROOT, METHODS, ROOT, document, edge, vertex


WORKLOADS = {
    "vit_tiny": "ViT-T",
    "cnn_tiny": "CNN-T",
    "mobilevit_xxs": "MobileViT-XXS",
    "cnn_small": "CNN-S",
    "mobilevit_xs": "MobileViT-XS",
}
LABELS = {
    "agent": "OPEarch-Agent",
    "rs": "RS",
    "sa": "SA",
    "bo": "BO",
    "ga": "GA",
    "sweep": "Parameter sweep",
}
# Muted fills for quantitative marks, with darker companion colors for missing-value symbols.
PALETTE = {
    "agent": ("#5A86B5", "#446C96"),
    "rs": ("#ABBACC", "#647B95"),
    "sa": ("#9EBDB6", "#5D847C"),
    "bo": ("#D8BA9E", "#A0714D"),
    "ga": ("#B6AACB", "#7C6C99"),
    "sweep": ("#C3BCAD", "#847A65"),
}
THRESHOLD = 0.6
OUT = ROOT / ".scratch/selected_40pct_first_hits.drawio"
PAGE_ID = "selected-40pct-first-hits"


def first_hits(workload: str) -> dict[str, int | None]:
    agent_path = MAIN_ROOT / "agent_mature3_w10_d10" / workload / "seed_0/run_iteration_001/trajectory.json"
    initial, agent = normalized_curve(agent_path)
    curves = {"agent": agent}
    for method, _, _ in METHODS[1:]:
        path = MAIN_ROOT / "baselines" / method / workload / "seed_0/trajectory.json"
        baseline_initial, curves[method] = normalized_curve(path)
        if baseline_initial != initial:
            raise ValueError(f"Initial objective mismatch: {path}")
    return {
        method: next((i for i, v in enumerate(curve[1:], start=1) if v <= THRESHOLD), None)
        for method, curve in curves.items()
    }


def build() -> None:
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    width, height = 956, 424
    left, right, top, bottom = 70, 926, 118, 354
    plot_height = bottom - top
    text = "text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
    dark = text + "fontColor=#202938;"
    muted = text + "fontColor=#4E5A6B;"
    grid = "endArrow=none;html=1;strokeColor=#DDE5EE;strokeWidth=0.8;"
    axis = "endArrow=none;html=1;strokeColor=#526075;strokeWidth=1.1;"

    cells.append(vertex("title", "Iterations to reach 40% improvement", 26, 14, 900, 33,
                        dark + "align=left;fontSize=22;"))
    cells.append(vertex("subtitle", "Relative to each workload's initial objective (seed 0)", 26, 47, 900, 20,
                        muted + "align=left;fontSize=12;"))

    legend_layout = ((70, 138), (226, 39), (283, 39), (340, 39), (397, 39), (454, 129))
    for (method, _, _), (x, label_width) in zip(METHODS, legend_layout):
        color, _ = PALETTE[method]
        cells.append(vertex(f"legend_{method}", "", x, 80, 13, 10,
                            f"rounded=0;html=1;fillColor={color};strokeColor=none;"))
        cells.append(vertex(f"legend_text_{method}", LABELS[method], x + 18, 74, label_width, 22,
                            dark + "align=left;fontSize=13;"))
    cells.append(vertex("legend_missing", "×  Not reached within 50 iterations", 660, 74, 267, 22,
                        muted + "align=right;fontSize=13;"))
    cells.append(vertex("ycaption", "Iterations", 4, 90, 57, 17,
                        muted + "align=right;fontSize=11;"))

    for tick in range(0, MAIN_BUDGET + 1, 10):
        y = bottom - plot_height * tick / MAIN_BUDGET
        style = axis if tick == 0 else grid
        cells.append(edge(f"grid_{tick}", left, y, right, y, style))
        cells.append(edge(f"y_mark_{tick}", left - 4, y, left, y, axis))
        cells.append(vertex(f"tick_{tick}", str(tick), 25, y - 10, 34, 20,
                            muted + "align=right;fontSize=13;"))
    cells.append(edge("yaxis", left, top, left, bottom, axis))

    group_width = (right - left) / len(WORKLOADS)
    bar_width, gap = 18, 4
    span = bar_width * len(METHODS) + gap * (len(METHODS) - 1)
    for idx, (workload, label) in enumerate(WORKLOADS.items()):
        hits = first_hits(workload)
        print(label, " ".join(str(n) if n is not None else "NR" for n in hits.values()))
        center = left + (idx + 0.5) * group_width
        start = center - span / 2
        for col, (method, _, _) in enumerate(METHODS):
            x = start + col * (bar_width + gap)
            key = f"{workload}_{method}"
            hit = hits[method]
            color, missing_color = PALETTE[method]
            if hit is None:
                cells.append(vertex(f"missing_{key}", "×", x - 2, top - 20, bar_width + 4, 20,
                                    text + f"fontColor={missing_color};align=center;fontSize=16;fontStyle=1;"))
                continue
            height_px = plot_height * hit / MAIN_BUDGET
            y = bottom - height_px
            cells.append(vertex(f"bar_{key}", "", x, y, bar_width, height_px,
                                f"rounded=0;html=1;fillColor={color};strokeColor=none;"))
            cells.append(vertex(f"value_{key}", str(hit), x - 2, y - 19, bar_width + 4, 17,
                                dark + "align=center;fontSize=12;"))
        cells.append(edge(f"x_mark_{workload}", center, bottom, center, bottom + 4, axis))
        cells.append(vertex(f"group_{workload}", label, center - 80, bottom + 9, 160, 24,
                            dark + "align=center;fontSize=14;"))

    cells.append(vertex("footnote", "The target is reached when the best objective so far is at most 60% of its initial value.",
                        26, 394, 900, 20, muted + "align=left;fontSize=12;"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(document(cells, width, height, PAGE_ID), encoding="utf-8")


if __name__ == "__main__":
    build()
