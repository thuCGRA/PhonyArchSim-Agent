"""运行产物自洽性审计：把一次人工排查发现的不变量固化成可重复检查。

用法::

    # 审计一个运行目录（含 iter_* 的父目录）
    .venv-agent/bin/python scripts/analysis/audit_consistency.py runs/_diag_old_objective/agent/seed_0/run_iteration_001

    # 递归审计 runs/ 下所有运行
    .venv-agent/bin/python scripts/analysis/audit_consistency.py runs/ --all

退出码：0 = 全部通过（可能有警告），1 = 存在硬失败。

## 为什么需要这个脚本

本项目已经出现过四类"每个环节都自洽、只有接口处不自洽"的 bug，全部是人工排查
发现的，代价很高：

1. **单位错配**：逐层能耗来自原生 pJ 明细，却被当成 mJ 与 mJ 的总量比较排序，
   排序在 1e9 倍误差下错乱（`_rank_layers`）。
2. **口径不一致**：目标只计计算能耗，而延时却含存储延时；实测存储占总能耗
   75–90%，于是优化方向系统性偏移。
3. **字段连接错**：逐层 `memory_energy_mj` 用层名去查按存储层级组织的表，恒为 0，
   智能体据此以为存储能耗为零。
4. **协议污染**：仿真器的普通文本写进 MCP stdio 通道，整批实验随机中断。

这些检查都是廉价的纯读操作，没有理由靠人眼。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Audit:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.checks: List[str] = []

    def ok(self, message: str) -> None:
        self.checks.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require(self, condition: bool, message: str) -> bool:
        if condition:
            self.ok(message)
        else:
            self.fail(message)
        return condition


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sum_leaves(value: Any) -> float:
    if isinstance(value, dict):
        return sum(_sum_leaves(v) for v in value.values())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def audit_iteration(audit: Audit, iter_dir: Path) -> None:
    """审计单个迭代目录的数值自洽性。"""
    tag = iter_dir.name
    metrics = _read(iter_dir / "metrics.json")
    report = _read(iter_dir / "report.json")
    if metrics is None or report is None:
        audit.warn(f"{tag}: 缺少 metrics.json / report.json（可能被中断）")
        return

    canonical = ("latency_ms", "energy_mj", "area_mm2", "fJ_per_mac", "objective")
    missing = [k for k in canonical if k not in metrics]
    audit.require(not missing, f"{tag}: 规范指标键齐全 {missing or ''}".strip())

    # 0) metrics 必须全是数值：任何嵌套字典都会让下游 float() 与 Pareto 比较崩掉
    non_numeric = [k for k, v in metrics.items() if isinstance(v, (dict, list, str))]
    audit.require(not non_numeric, f"{tag}: metrics 全为数值（非数值键 {non_numeric or '无'}）")

    # 0b) report.metrics 与 metrics.json 的可比键必须一致（两条落盘路径同源）
    report_metrics = report.get("metrics") or {}
    shared = [k for k in report_metrics if k in metrics]
    mismatched = [
        k for k in shared
        if isinstance(report_metrics[k], (int, float)) and isinstance(metrics[k], (int, float))
        and abs(float(report_metrics[k]) - float(metrics[k])) > max(abs(float(metrics[k])) * 1e-6, 1e-9)
    ]
    audit.require(
        not mismatched,
        f"{tag}: report.metrics 与 metrics.json 一致（不一致键 {mismatched or '无'}）",
    )

    # 1) 能耗口径：总量必须等于计算 + 存储
    si = report.get("si_metrics") or {}
    if "energy_compute_pj" in si and "energy_memory_pj" in si:
        total = float(si.get("energy_pj", 0.0))
        parts = float(si["energy_compute_pj"]) + float(si["energy_memory_pj"])
        audit.require(
            abs(total - parts) <= max(abs(total) * 1e-9, 1e-6),
            f"{tag}: 总能耗 = 计算 + 存储（{total:.6e} vs {parts:.6e} pJ）",
        )
        audit.require(
            float(metrics.get("energy_mj", 0.0)) > 0,
            f"{tag}: energy_mj 为正",
        )
        # 存储能耗必须被真正计入（而非静默丢弃）
        audit.require(
            abs(float(metrics.get("energy_mj", 0.0)) * 1e9 - total) <= max(abs(total) * 1e-6, 1e-3),
            f"{tag}: metrics.energy_mj 与 si_metrics.energy_pj 一致",
        )
    else:
        audit.warn(f"{tag}: 报告缺少能耗分量字段（旧版产物，无法核对口径）")

    # 2) 面积：按 chip_type 求和必须等于总量，且总量 = 逐器件面积 + 节点边界框
    area = report.get("area", {})
    by_chip = area.get("by_chip_type") or {}
    total_area = _sum_leaves(area.get("total") or {})
    chip_sum = sum(_sum_leaves(v.get("total_area")) for arch in by_chip.values()
                   for v in (arch or {}).values() if isinstance(v, dict))
    if chip_sum > 0 and total_area > 0:
        audit.require(
            abs(chip_sum - total_area) <= max(total_area * 1e-6, 1e-6),
            f"{tag}: 面积 by_chip_type 求和 = total（{chip_sum:.1f} vs {total_area:.1f} µm²）",
        )
    composition = area.get("composition") or {}
    for arch, rec in composition.items():
        if not isinstance(rec, dict):
            continue
        device = float(rec.get("device_area_um2", 0.0))
        bbox = float(rec.get("node_bbox_um2", 0.0))
        arch_total = float((area.get("total") or {}).get(arch, 0.0))
        audit.require(
            abs(device + bbox - arch_total) <= max(abs(arch_total) * 1e-6, 1e-6),
            f"{tag}: 面积 total = 逐器件 + 节点边界框（{device:.0f}+{bbox:.0f} vs {arch_total:.0f} µm²）",
        )
    # 厂商口径若被原样采用，会多算节点面积；这里确保我们没有回退到它。
    vendor = area.get("vendor_total_um2") or {}
    if vendor and total_area:
        vendor_sum = _sum_leaves(vendor)
        if abs(vendor_sum - total_area) > max(total_area * 1e-6, 1e-6):
            audit.ok(
                f"{tag}: 已弃用厂商面积口径（vendor={vendor_sum:.0f} µm²，"
                f"采用 {total_area:.0f} µm²）"
            )

    # 3) 延时：总量 = 计算 + 存储
    latency = report.get("latency", {})
    if latency.get("total") and latency.get("computation") and latency.get("memory"):
        total_lat = _sum_leaves(latency["total"])
        parts = _sum_leaves(latency["computation"]) + _sum_leaves(latency["memory"])
        audit.require(
            abs(total_lat - parts) <= max(abs(total_lat) * 1e-6, 1e-12),
            f"{tag}: 延时 = 计算 + 存储（{total_lat:.6e} vs {parts:.6e} s）",
        )

    # 4) 量级合理性：规范单位下不应出现 pJ/µm²/s 量级的值
    for key, ceiling in (("latency_ms", 1e6), ("energy_mj", 1e6), ("area_mm2", 1e7)):
        value = abs(float(metrics.get(key, 0.0)))
        audit.require(
            value < ceiling,
            f"{tag}: {key}={value:.4g} 在规范单位下量级合理（<{ceiling:g}）",
        )

    # 5) 目标函数可复算
    try:
        recomputed = (
            float(metrics["latency_ms"]) * float(metrics["fJ_per_mac"]) * float(metrics["area_mm2"])
        )
        penalty = float(metrics.get("il_budget_penalty", 1.0))
        audit.require(
            abs(float(metrics["objective"]) - recomputed * penalty) <= max(recomputed * 1e-6, 1e-6),
            f"{tag}: objective = latency × fJ/MAC × area × IL 惩罚",
        )
    except (KeyError, TypeError, ValueError) as exc:
        audit.fail(f"{tag}: objective 无法复算（{exc}）")

    # 6) 逐层能耗与总量同尺度
    ranks = _read(iter_dir / "analysis" / "layer_ranks.json")
    if isinstance(ranks, dict):
        layers = ranks.get("layers") or []
        if layers:
            layer_sum = sum(float(row.get("energy_mj", 0.0)) for row in layers)
            audit.require(
                0 <= layer_sum <= float(metrics["energy_mj"]) * 1.01 + 1e-9,
                f"{tag}: 逐层能耗之和不超过总能耗（{layer_sum:.4f} ≤ {metrics['energy_mj']:.4f} mJ）",
            )


def audit_schedule(audit: Audit, iter_dir: Path) -> None:
    """审计调度决策与实际执行是否一致。"""
    tag = iter_dir.name
    decision = _read(iter_dir / "analysis" / "schedule.json")
    if not isinstance(decision, dict):
        if tag == "iter_000":
            # iter_000 是初始基线求值，没有调度决策是正常的。
            audit.ok(f"{tag}: 初始基线（无调度决策，符合预期）")
        else:
            audit.warn(f"{tag}: 缺少 schedule.json（固定模式或旧版产物）")
        return
    planned = set(decision.get("stages_to_run") or [])
    analysis = iter_dir / "analysis"
    interrupted = not (iter_dir / "metrics.json").is_file()
    for stage in ("architecture", "topology", "device"):
        ran = (analysis / f"{stage}_result.json").is_file()
        skipped = (analysis / f"{stage}_skipped.json").is_file()
        expected = stage in planned
        consistent = (ran and expected) or (skipped and not expected)
        if consistent:
            audit.ok(f"{tag}: {stage} 的执行与计划一致（计划={expected}, 实跑={ran}, 跳过={skipped}）")
        elif interrupted and not ran and not skipped:
            audit.warn(f"{tag}: {stage} 尚未执行（迭代被中断，属正常）")
        else:
            audit.fail(f"{tag}: {stage} 的执行与计划不一致（计划={expected}, 实跑={ran}, 跳过={skipped}）")
    # 证据偏移必须是单调不减的流水线结构
    offsets = decision.get("evidence_offsets") or {}
    if decision.get("mode") == "pipeline":
        audit.require(
            offsets == {"architecture": 0, "topology": 1, "device": 2},
            f"{tag}: pipeline 的证据偏移为 0/1/2（实际 {offsets}）",
        )


def audit_cases(audit: Audit, run_dir: Path) -> None:
    """审计 RAG 案例是否带结局标记。"""
    db = run_dir.parent.parent / "history.db"
    if not db.is_file():
        return
    import sqlite3

    try:
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT metrics_json FROM history").fetchall()
        conn.close()
    except sqlite3.Error as exc:
        audit.warn(f"无法读取 {db}: {exc}")
        return
    if not rows:
        audit.warn("历史库为空（首次运行属正常）")
        return
    bad = 0
    for (raw,) in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            bad += 1
            continue
        if not isinstance(payload, dict) or "before" not in payload or "after" not in payload:
            bad += 1
    audit.require(bad == 0, f"历史案例都带 before/after 指标（异常 {bad}/{len(rows)}）")


def audit_run(root: Path) -> Audit:
    audit = Audit()
    iterations = sorted(p for p in root.glob("iter_*") if p.is_dir())
    if not iterations:
        audit.warn(f"{root}: 没有 iter_* 目录")
        return audit
    for iter_dir in iterations:
        audit_iteration(audit, iter_dir)
        audit_schedule(audit, iter_dir)
    audit_cases(audit, root)
    return audit


def _find_runs(path: Path) -> List[Path]:
    if (path / "trajectory.json").is_file() or any(path.glob("iter_*")):
        return [path]
    return sorted({p.parent for p in path.rglob("iter_000") if p.is_dir()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Run directory, or a tree to search.")
    parser.add_argument("--all", action="store_true", help="Recurse and audit every run found.")
    parser.add_argument("--quiet", action="store_true", help="Only print failures/warnings.")
    args = parser.parse_args(argv)

    runs = _find_runs(args.path) if args.all or args.path.is_dir() else [args.path]
    total_fail = 0
    for run in runs:
        audit = audit_run(run)
        total_fail += len(audit.failures)
        if not args.quiet:
            for line in audit.checks:
                print(f"  PASS  {line}")
        for line in audit.warnings:
            print(f"  WARN  {line}")
        for line in audit.failures:
            print(f"  FAIL  {line}")
        print(f"[{run}] {len(audit.checks)} passed, "
              f"{len(audit.warnings)} warnings, {len(audit.failures)} failures\n")
    print(f"total failures: {total_fail}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
