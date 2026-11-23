"""调度阈值敏感性回放：用落盘的记忆成熟度重算模式序列，无需新仿真。

## 为什么需要

调度器是一个确定性阈值函数，但**阈值本身是超参**。论文里断言"模式由记忆成熟度
与剩余预算驱动"，就必须回答"结论对阈值敏感吗"。跑新的阈值扫描要重新烧评估预算；
而调度是纯函数，输入只有每轮的记忆成熟度与预算比例——两者都已落盘。因此可以离线
回放：把记录喂回 ``scheduler.decide``，在阈值网格上重算模式序列。

## 现在的调度器只有两个信号

早期版本用了五个（ρ, σ, π, κ, β）。离线回放显示它的输出与一条不看信号的三行固定
课程表**逐轮相同**，因此 σ/π/κ 与 ``single`` 模式一并移除，见 ``scheduler`` 模块
docstring。本脚本相应只扫 ρ 与 β 的两个阈值。

## 自检

回放结果必须与运行记录的 ``mode_sequence`` 逐轮一致，否则说明回放逻辑已与
``scheduler.decide`` 脱节，扫描结论不可信。

## 用法

    .venv-agent/bin/python scripts/analysis/scheduler_sensitivity.py \
        runs/exp1_equal_budget/agent/seed_0/run_iteration_001/trajectory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from photon_tuner.runtime.scheduler import (  # noqa: E402
    ScheduleSignals,
    SchedulerConfig,
    decide,
)


def load_maturity(trajectory: Path) -> List[float]:
    """读取逐轮记忆成熟度（每轮 [0,1]）。"""
    data = json.loads(Path(trajectory).read_text(encoding="utf-8"))
    return [float(v) for v in (data.get("memory_maturity") or [])]


def replay(maturity: List[float], cfg: SchedulerConfig) -> List[str]:
    """按 ``decide`` 重算模式序列。

    第 k 轮的决策只能看到前 k-1 轮已产生的成熟度——本轮结果尚未回填。差一轮会让
    回放与真实轨迹错位。
    """
    n = len(maturity)
    modes: List[str] = []
    for k in range(1, n + 1):
        signals = ScheduleSignals(
            iteration=k,
            total_iterations=n,
            recent_memory_maturity=maturity[: k - 1],
        )
        modes.append(decide(signals, cfg).mode)
    return modes


def _summarise(modes: List[str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for mode in modes:
        counts[mode] = counts.get(mode, 0) + 1
    return {"sequence": modes, "counts": counts}


def sweep(maturity: List[float], recorded: List[str]) -> Dict[str, Any]:
    base = SchedulerConfig()
    baseline = replay(maturity, base)

    one_d: Dict[str, List[Dict[str, Any]]] = {}
    grids = {
        "rag_cold_below": [0.2, 0.34, 0.5, 0.67],
        "budget_low_below": [0.1, 0.2, 0.3],
        "rag_window": [1, 2, 3, 4],
        "warm_precedent_target": [1, 2, 3],
    }
    for name, values in grids.items():
        entries = []
        for value in values:
            cfg = SchedulerConfig(**{**base.__dict__, name: value})
            if name == "warm_precedent_target":
                # 目标数改变的是成熟度本身的含义：先例数 = maturity * 原目标。
                adjusted = [
                    min(round(v * base.warm_precedent_target) / float(value), 1.0)
                    for v in maturity
                ]
                modes = replay(adjusted, cfg)
            else:
                modes = replay(maturity, cfg)
            entries.append({"value": value, "modes": _summarise(modes)})
        one_d[name] = entries

    return {
        "iterations": len(maturity),
        "baseline": _summarise(baseline),
        "reproduces_recorded": baseline == recorded,
        "recorded_sequence": list(recorded),
        "maturity": list(maturity),
        "one_dimensional": one_d,
    }


def to_markdown(result: Dict[str, Any], source: str) -> str:
    lines = [
        "# 调度阈值敏感性（离线回放）",
        "",
        f"- 数据来源：`{source}`",
        f"- 轮数：{result['iterations']}",
        f"- 自检（回放是否复现真实轨迹）：**{result['reproduces_recorded']}**",
        "",
        "调度器只有两个信号：ρ（记忆成熟度）与 β（剩余预算）。σ/π/κ 与 `single` 模式",
        "已因「从未改变任何决策」而移除。",
        "",
        "## 默认阈值下的轨迹",
        "",
        "```",
        " → ".join(result["baseline"]["sequence"]),
        "```",
        "构成：" + ", ".join(f"`{k}`×{v}" for k, v in result["baseline"]["counts"].items()),
        "",
        "## 单阈值扫描",
        "",
    ]
    for name, entries in result["one_dimensional"].items():
        lines += [f"**{name}**", "", "| 取值 | 模式构成 |", "|---:|---|"]
        for entry in entries:
            counts = ", ".join(f"{k}×{v}" for k, v in entry["modes"]["counts"].items())
            lines.append(f"| {entry['value']} | {counts} |")
        lines.append("")

    stable = all(
        entry["modes"]["sequence"] == result["baseline"]["sequence"]
        for entries in result["one_dimensional"].values()
        for entry in entries
    )
    lines += [
        "## 判读",
        "",
        ("在所有单阈值取值下，模式序列都不变——说明这套调度对阈值不敏感，"
         "也因此**不依赖精细调参**。" if stable else
         "存在阈值取值改变了模式序列——论文中必须给出所选阈值并说明理由。"),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=Path, help="Path to a run's trajectory.json.")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs" / "results" / "scheduler_sensitivity")
    args = parser.parse_args(argv)

    maturity = load_maturity(args.trajectory)
    if not maturity:
        print("no memory_maturity in trajectory; nothing to replay", file=sys.stderr)
        return 1
    data = json.loads(args.trajectory.read_text(encoding="utf-8"))
    recorded = [str(item.get("mode")) for item in (data.get("schedule_history") or [])]

    result = sweep(maturity, recorded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(
        to_markdown(result, str(args.trajectory)), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.md')}")
    print("baseline:", result["baseline"]["sequence"])
    print("reproduces recorded:", result["reproduces_recorded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
