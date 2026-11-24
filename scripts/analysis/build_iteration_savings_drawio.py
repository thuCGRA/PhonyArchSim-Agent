#!/usr/bin/env python3
"""Compare seed-0 iteration savings at each baseline's 50-round quality."""

from __future__ import annotations

import json
from itertools import accumulate
from math import isfinite
from pathlib import Path
from statistics import mean

from build_main_bar_and_ablation_drawio import (
    MAIN_BUDGET,
    MAIN_ROOT,
    ROOT,
    WORKLOAD_FAMILIES,
    document,
    edge,
    vertex,
)


OUTPUT = ROOT / ".scratch/main_iteration_savings.drawio"
BASELINES = {
    "rs": ("RS", "随机搜索"),
    "sa": ("SA", "模拟退火"),
    "bo": ("BO", "贝叶斯优化"),
    "ga": ("GA", "遗传算法"),
    "sweep": ("Sweep", "参数扫描"),
}


def normalized_curve(path: Path) -> tuple[float, list[float]]:
    history = json.loads(path.read_text(encoding="utf-8"))["history"]
    if len(history) < MAIN_BUDGET + 1 or int(history[0]["iteration"]) != 0:
        raise ValueError(f"Incomplete 50-round trajectory: {path}")
    # Log order counts proposals, including cache hits that reuse an evaluation ID.
    objectives = [float(row["metrics"]["objective"]) for row in history[:MAIN_BUDGET + 1]]
    initial = objectives[0]
    if not isfinite(initial) or initial <= 0:
        raise ValueError(f"Invalid initial objective: {path}")
    return initial, [best / initial for best in accumulate(objectives, min)]


def family_matches(workloads: tuple[str, ...]) -> dict[str, tuple[float, int, int | None]]:
    agent_paths = [
        MAIN_ROOT / "agent_mature3_w10_d10" / workload / "seed_0/run_iteration_001/trajectory.json"
        for workload in workloads
    ]
    initials, agent_curves = zip(*(normalized_curve(path) for path in agent_paths))
    agent_average = list(map(mean, zip(*agent_curves)))
    matches = {}
    for method in BASELINES:
        baseline_curves = []
        for workload, initial in zip(workloads, initials):
            path = MAIN_ROOT / "baselines" / method / workload / "seed_0/trajectory.json"
            baseline_initial, curve = normalized_curve(path)
            if baseline_initial != initial:
                raise ValueError(f"Initial objective mismatch: {path}")
            baseline_curves.append(curve)
        baseline_average = list(map(mean, zip(*baseline_curves)))
        target = baseline_average[-1]
        baseline_hit = next(i for i, value in enumerate(baseline_average) if value <= target)
        agent_hit = next((i for i, value in enumerate(agent_average) if value <= target), None)
        matches[method] = target, baseline_hit, agent_hit
    return matches


def savings_color(value: float) -> str:
    if value < 0:
        return "#F6DEE0"
    light, dark = (246, 249, 255), (125, 169, 217)
    channels = [round(lo + (hi - lo) * value / 100) for lo, hi in zip(light, dark)]
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def build() -> None:
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    width, height = 916, 408
    x0, y0, cell_w, cell_h, gap = 143, 117, 143, 63, 7
    text = "text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
    dark = text + "fontColor=#202938;"
    muted = text + "fontColor=#425268;"
    grid = "endArrow=none;html=1;strokeColor=#DFE3EA;strokeWidth=0.75;"

    cells.append(vertex("title", "达到同等解质量时节省的迭代轮数", 27, 17, 860, 31,
                        dark + "align=left;fontSize=22;fontStyle=1;"))
    cells.append(vertex("subtitle", "首次达标轮次（传统基线 → OPEarch-Agent）", 27, 49, 840, 20,
                        muted + "align=left;fontSize=13;"))

    for col, (method, (abbr, label)) in enumerate(BASELINES.items()):
        x = x0 + col * (cell_w + gap)
        cells.append(vertex(f"heading_{method}", abbr, x, 76, cell_w, 21,
                            dark + "align=center;fontSize=15;fontStyle=1;"))
        cells.append(vertex(f"heading_caption_{method}", label, x, 97, cell_w, 17,
                            muted + "align=center;fontSize=11;"))

    for row, (family, workloads) in enumerate(WORKLOAD_FAMILIES.items()):
        y = y0 + row * (cell_h + gap)
        cells.append(vertex(f"family_{family}", family, 27, y, 101, cell_h,
                            dark + "align=right;fontSize=16;fontStyle=1;"))
        for col, (method, (target, baseline, agent)) in enumerate(family_matches(workloads).items()):
            x = x0 + col * (cell_w + gap)
            suffix = f"{family}_{method}"
            saving = 100 * (baseline - agent) / baseline if baseline and agent is not None else None
            label = f"{saving:.1f}%" if saving is not None else "—"
            agent_label = str(agent) if agent is not None else "未达标"
            shade = savings_color(saving) if saving is not None else "#EFF1F4"
            print(f"{family} {method}: target improvement {100 * (1 - target):.8f}%, "
                  f"{baseline} → {agent_label} rounds, savings {label}")
            cells.append(vertex(f"fill_{suffix}", "", x, y, cell_w, cell_h,
                                f"rounded=0;html=1;fillColor={shade};strokeColor=none;"))
            cells.append(vertex(f"reduction_{suffix}", label, x, y + 8, cell_w, 25,
                                dark + "align=center;fontSize=21;fontStyle=1;"))
            cells.append(vertex(f"rounds_{suffix}", f"{baseline} → {agent_label} 轮", x, y + 35, cell_w, 19,
                                muted + "align=center;fontSize=13;"))

    cells.append(vertex("scale_caption", "轮数节省", 649, 331, 64, 18,
                        muted + "align=right;fontSize=11;"))
    for idx, pct in enumerate(range(0, 101, 5)):
        shade = savings_color(pct)
        cells.append(vertex(f"scale_{pct}", "", 720 + 6 * idx, 334, 6, 11,
                            f"rounded=0;html=1;fillColor={shade};strokeColor=none;"))
    for pct, x in ((0, 704), (50, 765), (100, 829)):
        cells.append(vertex(f"scale_label_{pct}", f"{pct}%", x, 346, 36, 15,
                            muted + "align=center;fontSize=10;"))

    cells.append(edge("footer_rule", 27, 363, 886, 363, grid))
    cells.append(vertex("target_note", "每格的目标为该基线第 50 轮达到的归一化质量。三个规模等权平均。",
                        27, 368, 856, 17, muted + "align=left;fontSize=12;"))
    cells.append(vertex("sample_note", "所有方法均采用 seed 0。百分比以该基线的首次达标轮次为分母。",
                        27, 387, 856, 17, muted + "align=left;fontSize=12;"))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document(cells, width, height, "iteration-savings"), encoding="utf-8")


if __name__ == "__main__":
    build()
