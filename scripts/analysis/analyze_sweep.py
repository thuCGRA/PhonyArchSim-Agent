"""Analyze the real-mode sweep results (runs/sweep/).

Reads each completed run's iter_000 (baseline) and iter_001 (proposal) metrics,
computes per-run improvement, checks cross-run determinism of the baseline, and
lists any failed runs.  Prints a compact table + JSON summary.

Usage::

    source scripts/env/activate_real.sh
    python scripts/analysis/analyze_sweep.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "runs" / "sweep"
KEYS = ("latency_s", "energy_pj", "area_um2", "objective")


def load_metrics(run_dir: Path, it: str) -> dict:
    p = run_dir / it / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_best(run_dir: Path) -> dict:
    """The DSE's kept best point (trajectory.json 'best'), always >= baseline."""
    p = run_dir / "trajectory.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text(encoding="utf-8"))
    return t.get("best", {}) or {}


def load_hardware(run_dir: Path, it: str) -> dict:
    p = run_dir / it / "hardware.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def hw_delta(b: dict, p: dict) -> str:
    """Human-readable summary of what changed between two hardware dicts."""
    if not b or not p:
        return "?"
    diffs = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        else:
            diffs.append(f"{path}:{x}->{y}")

    walk(b, p, "")
    return "; ".join(diffs) if diffs else "(none)"


def classify_driven_by(run_dir: Path) -> str:
    """LLM vs fallback: did the LLM's successful edits explain the hardware diff?

    ``_repair_duplicate`` steps work_freq when a proposal is a duplicate of the
    baseline (i.e. when all LLM edits failed or were no-ops). If the diff is
    work_freq 5->10 and the log has no successful work_freq edit from the LLM,
    the improvement is rule-based ("fallback"); otherwise the LLM's edits
    explain the change ("LLM").
    """
    log = run_dir / "run.log"
    if not log.exists():
        return "?"
    text = log.read_text(encoding="utf-8", errors="replace")
    diff = hw_delta(load_hardware(run_dir, "iter_000"),
                    load_hardware(run_dir, "iter_001"))
    if diff in ("?", "(none)"):
        return "none" if diff == "(none)" else "?"
    # The fallback only ever produces work_freq 5->10 (first candidate step).
    if "work_freq:5->10" in diff:
        ok_lines = [ln for ln in text.splitlines() if "[EDIT][ok]" in ln]
        if any("step_hardware_parameter" in ln and "work_freq" in ln for ln in ok_lines):
            return "LLM"
        if any("modify_hardware_parameter" in ln and "work_freq" in ln for ln in ok_lines):
            return "LLM"
        return "fallback"
    return "LLM"


def main() -> int:
    rows = []
    baselines = defaultdict(list)  # workload -> list of baseline objective
    problems = []

    for wl_dir in sorted(SWEEP.iterdir()):
        if not wl_dir.is_dir() or wl_dir.name == "sweep_summary.jsonl":
            continue
        for run_dir in sorted(wl_dir.glob("run_*")):
            b = load_metrics(run_dir, "iter_000")
            p = load_metrics(run_dir, "iter_001")
            if not b:
                continue  # run not finished / no baseline yet
            base_obj = b.get("objective", float("nan"))
            prop_obj = p.get("objective", float("nan")) if p else float("nan")
            best_obj = load_best(run_dir).get("objective", float("nan"))
            imp = (base_obj - prop_obj) / base_obj * 100 if base_obj else float("nan")
            best_imp = (base_obj - best_obj) / base_obj * 100 if base_obj else float("nan")
            rows.append((wl_dir.name, run_dir.name, base_obj, prop_obj, imp, best_imp,
                         hw_delta(load_hardware(run_dir, "iter_000"),
                                 load_hardware(run_dir, "iter_001")),
                         classify_driven_by(run_dir)))
            baselines[wl_dir.name].append(base_obj)
            if not p:
                problems.append(f"{wl_dir.name}/{run_dir.name}: no iter_001 (proposal) found")

    print(f"{'workload':10s} {'run':6s} {'baseline':>13s} {'proposal':>13s} {'prop%':>7s} {'best%':>7s} {'driven':>8s}  delta")
    print("-" * 124)
    for wl, run, bo, po, imp, best_imp, delta, driven in sorted(rows):
        reg = "  ⚠回归" if imp < 0 else ""
        print(f"{wl:10s} {run:6s} {bo:13.5g} {po:13.5g} {imp:7.2f} {best_imp:7.2f} {driven:>8s}  {delta}{reg}")

    print()
    print("=== 确定性检查（各负载 5 次运行的 baseline objective 应一致）===")
    for wl, objs in sorted(baselines.items()):
        mn, mx = min(objs), max(objs)
        spread = (mx - mn) / mn * 100 if mn else float("nan")
        tag = "OK" if len(objs) >= 2 and spread < 1e-6 else ("PENDING" if len(objs) < 5 else "NON-DETERMINISTIC")
        print(f"{wl:10s} n={len(objs)}  min={mn:.10g} max={mx:.10g} spread={spread:.2e}%  [{tag}]")

    print()
    print("=== 各负载改进统计（5 次运行）===")
    from statistics import mean, pstdev
    for wl in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == wl]
        pimps = [r[4] for r in rr if r[4] == r[4]]  # 提案改进（可为负）
        bimps = [r[5] for r in rr if r[5] == r[5]]  # DSE 保留最佳改进（≥0）
        best = min((r[3] for r in rr if r[3] == r[3]), default=float("nan"))
        base = rr[0][2] if rr else float("nan")
        reg = sum(1 for x in pimps if x < 0)
        if bimps:
            print(f"{wl:10s} n={len(bimps)}  best_mean={mean(bimps):+.2f}%  std={pstdev(bimps):.2f}%  "
                  f"proposal_mean={mean(pimps):+.2f}%  baseline={base:.6g}  best={best:.6g}  回归次数={reg}")

    if problems:
        print()
        print("=== 问题 ===")
        for p in problems:
            print(" -", p)

    print()
    print("=== LLM vs 回退（改进来源）===")
    from collections import Counter
    for wl in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == wl and r[7] in ("LLM", "fallback")]
        c = Counter(r[7] for r in rr)
        n = len(rr)
        rate = c["LLM"] / n * 100 if n else 0.0
        print(f"{wl:10s} n={n}  LLM={c['LLM']}  fallback={c['fallback']}  LLM成功率={rate:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
