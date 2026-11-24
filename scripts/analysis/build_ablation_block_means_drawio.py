#!/usr/bin/env python3
"""Plot five-round means of the measured MobileViT-S ablation changes."""

import json
from math import log2
from statistics import mean

from build_main_bar_and_ablation_drawio import MAIN_BUDGET, MAIN_TRAJ, ROOT, continued_values, document, edge, vertex


BLOCK_SIZE = 5
OUT = ROOT / ".scratch/ablation_block_means.drawio"
PAGE_ID = "ablation-five-round-means"
SERIES = (
    ("full", "完整系统", "#4D3B73", "ellipse", 9),
    ("no_rag", "无 RAG", "#AE6424", "rectangle", 7.5),
    ("single_agent", "单智能体", "#287579", "triangle", 9.5),
)


def block_improvements() -> dict[str, list[float]]:
    history = json.loads(MAIN_TRAJ.read_text(encoding="utf-8"))["history"]
    if [int(row["iteration"]) for row in history] != list(range(MAIN_BUDGET + 1)):
        raise ValueError("The full system trajectory must cover rounds 0–50")
    initial = float(history[0]["metrics"]["objective"])
    values = {
        "full": [float(row["metrics"]["objective"]) for row in history[1:]],
        "no_rag": continued_values("no_rag", 33, initial),
        "single_agent": continued_values("single_agent", 27, initial),
    }
    return {
        key: [mean(100 * (initial - v) / initial for v in objectives[i:i + BLOCK_SIZE])
              for i in range(0, MAIN_BUDGET, BLOCK_SIZE)]
        for key, objectives in values.items()
    }


def build() -> None:
    curves = block_improvements()
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    points = []
    last_points = []
    width, height = 950, 411
    left, first_x, last_x, grid_right = 83, 98, 824, 927
    top, bottom = 105, 320
    count = MAIN_BUDGET // BLOCK_SIZE
    x_step = (last_x - first_x) / (count - 1)
    text = "text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
    dark = text + "fontColor=#202938;"
    muted = text + "fontColor=#4E5A6B;"
    grid = "endArrow=none;html=1;strokeColor=#E1E4EB;strokeWidth=0.85;"
    reference = "endArrow=none;html=1;strokeColor=#A7AEBC;strokeWidth=0.9;dashed=1;dashPattern=4 4;"

    # The objective ratio for a block is 1 minus its mean improvement fraction.
    # Equally spaced ticks in log-ratio space retain their actual improvement labels.
    def y_for(pct: float) -> float:
        return top + (bottom - top) * (log2(1 - pct / 100) + 3) / 5

    cells.append(vertex("title", "MobileViT-S 消融实验", 26, 14, 890, 32,
                        dark + "align=left;fontSize=22;fontStyle=1;"))
    cells.append(vertex("subtitle", "每 5 轮的平均搜索改善", 26, 47, 375, 20,
                        muted + "align=left;fontSize=13;"))
    cells.append(vertex("ycaption", "平均改善", 12, 74, 61, 18,
                        muted + "align=right;fontSize=12;"))

    for idx, (key, label, color, shape, size) in enumerate(SERIES):
        x = 545 + idx * 127
        stroke = f"endArrow=none;html=1;strokeColor={color};strokeWidth=2.4;"
        marker = f"shape={shape};direction=north;html=1;fillColor={color};strokeColor=none;"
        cells.append(edge(f"legend_line_{key}", x, 57, x + 25, 57, stroke))
        cells.append(vertex(f"legend_point_{key}", "", x + 12.5 - size / 2, 57 - size / 2, size, size, marker))
        cells.append(vertex(f"legend_label_{key}", label, x + 30, 46, 87, 22,
                            dark + "align=left;fontSize=13;"))

    tick_labels = ((87.5, "87.5%"), (75, "75%"), (50, "50%"), (0, "0%"), (-100, "−100%"), (-300, "−300%"))
    for idx, (pct, label) in enumerate(tick_labels):
        y = y_for(pct)
        cells.append(edge(f"grid_{idx}", left, y, grid_right, y, reference if pct == 0 else grid))
        cells.append(vertex(f"tick_{idx}", label, 11, y - 10, 62, 20,
                            muted + "align=right;fontSize=12;"))
    for i in range(count):
        x = first_x + i * x_step
        cells.append(vertex(f"xblock_{i}", f"{i * BLOCK_SIZE + 1}–{(i + 1) * BLOCK_SIZE}",
                            x - 34, bottom + 9, 68, 20, muted + "align=center;fontSize=12;"))

    for key, _, color, shape, size in SERIES:
        stroke_width = 2.8 if key == "full" else 2.3
        line = f"endArrow=none;html=1;strokeColor={color};strokeWidth={stroke_width};"
        marker = f"shape={shape};direction=north;html=1;fillColor={color};strokeColor=#FFFFFF;strokeWidth=0.6;"
        curve = curves[key]
        if any(not -300 <= value <= 87.5 for value in curve):
            raise ValueError(f"An average lies outside the configured axis limits: {key}")
        print(key, " ".join(f"{v:.4f}" for v in curve))
        xy = [(first_x + i * x_step, y_for(v)) for i, v in enumerate(curve)]
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(xy, xy[1:])):
            cells.append(edge(f"curve_{key}_{i}", x1, y1, x2, y2, line))
        for i, (x, y) in enumerate(xy):
            points.append(vertex(f"point_{key}_{i}", "", x - size / 2, y - size / 2, size, size, marker))
        last_points.append((xy[-1][1], key, color, curve[-1]))
    cells.extend(points)

    labels = []
    for y, key, color, value in sorted(last_points):
        label_y = max(y, labels[-1][-1] + 18) if labels else y
        labels.append((y, key, color, value, label_y))
    shift = mean(label_y - y for y, _, _, _, label_y in labels)
    for y, key, color, value, label_y in labels:
        label_y -= shift
        leader = f"endArrow=none;html=1;strokeColor={color};strokeWidth=0.9;opacity=65;"
        cells.append(edge(f"end_leader_{key}", last_x + 6, y, last_x + 23, label_y, leader))
        cells.append(vertex(f"end_value_{key}", f"{value:.1f}%", last_x + 27, label_y - 10, 74, 20,
                            text + f"fontColor={color};align=left;fontSize=14;fontStyle=1;"))

    dip = curves["single_agent"][4]
    dip_color = SERIES[-1][2]
    cells.append(vertex("dip_label", f"{dip:.1f}%".replace("-", "−"),
                        first_x + 4 * x_step + 17, y_for(dip) - 16, 96, 20,
                        text + f"fontColor={dip_color};align=left;fontSize=13;fontStyle=1;"))
    cells.append(vertex("xcaption", "迭代轮次区间", 365, 350, 193, 20,
                        dark + "align=center;fontSize=13;"))
    note = "每点为 5 轮当轮改善率的算术平均。纵轴按平均目标比值的对数间距排列，刻度显示实际改善率。"
    cells.append(vertex("scale_note", note, 26, 377, 890, 20,
                        muted + "align=left;fontSize=11;"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(document(cells, width, height, PAGE_ID), encoding="utf-8")


if __name__ == "__main__":
    build()
