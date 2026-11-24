"""Audit attempted and successfully evaluated design-space coverage.

The audit deliberately separates hardware snapshots that were merely attempted
from snapshots with a completed ``metrics.json``.  Device-library coverage is
computed only for slots active under the snapshot's topology template; dormant
default subtrees in ``hardware.json`` therefore do not inflate coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from photon_tuner.common.paths import PROJECT_ROOT
from photon_tuner.core.search_space import (
    LAYERED_DESIGN_SPACE,
    DesignSpace,
    discover_device_catalog,
)
from photon_tuner.core.topology_space import (
    COUPLING_MATRIX,
    CORE_KINDS,
    PARAM_SCOPE,
    REQUIRED_DEVICE_SLOTS,
    TOPOLOGY_FAMILIES,
)


EXPERIMENT_ROOT = PROJECT_ROOT / "runs" / "paper_experiments_20261116"
FORMAL_NAME = "01_main_budget_atomic_formal_eff5_focus15"
DEVICE_ROOT = (
    PROJECT_ROOT
    / "photon_tuner"
    / "simulators"
    / "simphony"
    / "configs"
    / "devices"
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _get(root: Mapping[str, Any], path: Iterable[str]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _fingerprint(hardware: Mapping[str, Any]) -> str:
    payload = json.dumps(hardware, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _scope(path: Path) -> str:
    rel = path.relative_to(EXPERIMENT_ROOT).as_posix()
    if rel.startswith(f"{FORMAL_NAME}/baselines/"):
        return "formal_baselines_current"
    if rel.startswith(f"{FORMAL_NAME}/agent_phasefix/"):
        return "agent_invalid_phasefix"
    if rel.startswith(f"{FORMAL_NAME}/agent/"):
        return "agent_invalid_focus_starvation"
    if rel.startswith(f"{FORMAL_NAME}/agent_mature3_w10_d10/"):
        return "agent_formal_current"
    if rel.startswith("01_main_budget_atomic/agent/"):
        return "agent_pilot_legacy"
    if "/baselines/" in f"/{rel}":
        return "baseline_legacy_or_preflight"
    return "other_paper_runs"


def _valid_metrics(path: Path) -> bool:
    data = _read_json(path)
    if not isinstance(data, Mapping):
        return False
    objective = data.get("objective")
    if objective is None and isinstance(data.get("metrics"), Mapping):
        objective = data["metrics"].get("objective")
    try:
        return objective is not None and math.isfinite(float(objective))
    except (TypeError, ValueError):
        return False


def _active_device_values(hardware: Mapping[str, Any]) -> Dict[str, Any]:
    topology = _get(hardware, ("arch", "topology")) or {}
    core = str(topology.get("core_kind", "dptc_crossbar"))
    node = str(topology.get("node_kind", ""))
    slots = REQUIRED_DEVICE_SLOTS.get(core, {}).get(node, [])
    devices = hardware.get("devices") or {}
    out: Dict[str, Any] = {}
    for slot in slots:
        if core == "dptc_crossbar":
            branch = "node" if slot in {"phase_shifter", "photodetector", "coupler", "y_branch"} else "core"
            value = _get(devices, (branch, slot))
        elif core.startswith("mzi_mesh_"):
            branch = "node" if slot in {"coupler", "phase_shifter"} else "core"
            value = _get(devices, ("mesh", branch, slot))
        else:
            branch = "node" if slot in {"weight_mrr", "pcm_cell", "mzm"} else "core"
            value = _get(devices, ("weight_array", branch, slot))
        if value is not None:
            out[slot] = value
    return out


def _joint_cardinality(space: DesignSpace, catalog: Mapping[str, Mapping[str, Any]]) -> int:
    arch_counts = {
        param: len(space.candidates(param))
        for layer, param, _spec in space.params()
        if layer == "architecture"
    }
    knobs = math.prod(
        len(space.candidates(p))
        for p in ("sharing_policy", "temporal_accum", "laser_distribution")
    )
    ranks = len(space.candidates("mesh_rank"))
    total = 0
    for core in CORE_KINDS:
        family = TOPOLOGY_FAMILIES[core]
        arch_size = 1
        for param, count in arch_counts.items():
            status = (
                PARAM_SCOPE.get(param, {})
                .get("families", {})
                .get(family, {})
                .get("status", "effective")
            )
            arch_size *= count if status == "effective" else 1
        node = COUPLING_MATRIX[core]["node_kinds"][0]
        device_size = math.prod(
            max(1, len(catalog.get(slot, {})))
            for slot in REQUIRED_DEVICE_SLOTS[core][node]
        ) * 4
        topology_size = knobs if family == "crossbar_family" else knobs * ranks
        total += arch_size * topology_size * device_size
    return total


def _empty_stats() -> Dict[str, Any]:
    return {
        "attempted_snapshots": 0,
        "evaluated_snapshots": 0,
        "attempted_unique_configs": set(),
        "evaluated_unique_configs": set(),
        "attempted_axis_values": defaultdict(set),
        "evaluated_axis_values": defaultdict(set),
        "attempted_active_device_values": defaultdict(set),
        "evaluated_active_device_values": defaultdict(set),
        "source_runs": set(),
    }


def _add_snapshot(stats: Dict[str, Any], hardware: Mapping[str, Any], path: Path, evaluated: bool, space: DesignSpace) -> None:
    fp = _fingerprint(hardware)
    stats["attempted_snapshots"] += 1
    stats["attempted_unique_configs"].add(fp)
    stats["source_runs"].add(path.parent.parent.as_posix())
    for _layer, param, spec in space.params():
        value = _get(hardware, spec.get("map") or [])
        if value is not None:
            stats["attempted_axis_values"][param].add(str(value))
    for slot, value in _active_device_values(hardware).items():
        stats["attempted_active_device_values"][slot].add(str(value))
    if not evaluated:
        return
    stats["evaluated_snapshots"] += 1
    stats["evaluated_unique_configs"].add(fp)
    for _layer, param, spec in space.params():
        value = _get(hardware, spec.get("map") or [])
        if value is not None:
            stats["evaluated_axis_values"][param].add(str(value))
    for slot, value in _active_device_values(hardware).items():
        stats["evaluated_active_device_values"][slot].add(str(value))


def _serialize_stats(stats: Dict[str, Any], space: DesignSpace, catalog: Mapping[str, Mapping[str, Any]], joint: int) -> Dict[str, Any]:
    axes: Dict[str, Any] = {}
    for layer, param, _spec in space.params():
        candidates = [str(v) for v in space.candidates(param)]
        attempted = sorted(stats["attempted_axis_values"].get(param, set()))
        evaluated = sorted(stats["evaluated_axis_values"].get(param, set()))
        axes[param] = {
            "layer": layer,
            "candidate_count": len(candidates),
            "attempted_candidate_values": sorted(set(attempted) & set(candidates)),
            "evaluated_candidate_values": sorted(set(evaluated) & set(candidates)),
            "never_attempted_candidate_values": sorted(set(candidates) - set(attempted)),
            "never_evaluated_candidate_values": sorted(set(candidates) - set(evaluated)),
            "observed_outside_declared_candidates": sorted(set(evaluated) - set(candidates)),
        }
    devices: Dict[str, Any] = {}
    searchable_slots = {
        slot
        for per_core in REQUIRED_DEVICE_SLOTS.values()
        for slots in per_core.values()
        for slot in slots
    }
    for slot in sorted(searchable_slots):
        variants = catalog.get(slot, {})
        candidates = set(map(str, variants))
        attempted = set(stats["attempted_active_device_values"].get(slot, set()))
        evaluated = set(stats["evaluated_active_device_values"].get(slot, set()))
        devices[slot] = {
            "candidate_count": len(candidates),
            "attempted_variants": sorted(attempted & candidates),
            "evaluated_variants": sorted(evaluated & candidates),
            "never_attempted_variants": sorted(candidates - attempted),
            "never_evaluated_variants": sorted(candidates - evaluated),
        }
    unique_eval = len(stats["evaluated_unique_configs"])
    return {
        "source_run_count": len(stats["source_runs"]),
        "attempted_snapshots": stats["attempted_snapshots"],
        "evaluated_snapshots": stats["evaluated_snapshots"],
        "attempted_unique_configs": len(stats["attempted_unique_configs"]),
        "evaluated_unique_configs": unique_eval,
        "joint_conditioned_space": joint,
        "evaluated_joint_coverage_fraction": unique_eval / joint if joint else 0.0,
        "axes": axes,
        "active_device_variants": devices,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    all_stats = report["scopes"]["all_paper_runs"]
    lines = [
        "# Design-space coverage audit",
        "",
        f"Generated from `{report['experiment_root']}`. A snapshot is *attempted* when an iteration has "
        "`hardware.json`; it is *evaluated* only when the same iteration has a finite objective in "
        "`metrics.json`. Invalid Agent pilots remain useful for coverage auditing but are excluded from "
        "formal performance claims.",
        "",
        "## Scope summary",
        "",
        "| Scope | Runs | Attempted snapshots | Evaluated snapshots | Unique evaluated configs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["scopes"].items():
        lines.append(
            f"| {name} | {stats['source_run_count']} | {stats['attempted_snapshots']} | "
            f"{stats['evaluated_snapshots']} | {stats['evaluated_unique_configs']} |"
        )
    lines += [
        "",
        "## Aggregate evaluated parameter-value coverage",
        "",
        "| Layer | Parameter | Covered | Total | Never evaluated |",
        "|---|---|---:|---:|---|",
    ]
    for param, axis in all_stats["axes"].items():
        missing = ", ".join(axis["never_evaluated_candidate_values"]) or "—"
        lines.append(
            f"| {axis['layer']} | {param} | {len(axis['evaluated_candidate_values'])} | "
            f"{axis['candidate_count']} | {missing} |"
        )
    lines += [
        "",
        "## Aggregate active device-variant coverage",
        "",
        "| Device type | Covered | Total | Never evaluated |",
        "|---|---:|---:|---|",
    ]
    for slot, item in all_stats["active_device_variants"].items():
        missing = ", ".join(item["never_evaluated_variants"]) or "—"
        lines.append(
            f"| {slot} | {len(item['evaluated_variants'])} | {item['candidate_count']} | {missing} |"
        )
    lines += [
        "",
        "## Combination-level interpretation",
        "",
        f"- Unique successfully evaluated full hardware configurations: **{all_stats['evaluated_unique_configs']}**.",
        f"- Paradigm-conditioned legal-space cardinality: **{all_stats['joint_conditioned_space']:.6g}**.",
        f"- Exact full-combination coverage: **{all_stats['evaluated_joint_coverage_fraction']:.3e}**.",
        "- This tiny fraction is expected for a combinatorial co-design space; parameter-value coverage and "
        "  convergence/sample-efficiency curves are the meaningful diagnostics under a 50-evaluation budget.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    space = DesignSpace(spec=LAYERED_DESIGN_SPACE)
    catalog = discover_device_catalog(str(DEVICE_ROOT))
    joint = _joint_cardinality(space, catalog)
    stats_by_scope: Dict[str, Dict[str, Any]] = defaultdict(_empty_stats)
    all_stats = _empty_stats()
    hardware_paths = sorted(EXPERIMENT_ROOT.rglob("iter_*/hardware.json"))
    for path in hardware_paths:
        hardware = _read_json(path)
        if not isinstance(hardware, Mapping):
            continue
        evaluated = _valid_metrics(path.with_name("metrics.json"))
        scope = _scope(path)
        _add_snapshot(stats_by_scope[scope], hardware, path, evaluated, space)
        if scope.startswith("agent_"):
            _add_snapshot(stats_by_scope["all_agent_runs"], hardware, path, evaluated, space)
        if "baseline" in scope:
            _add_snapshot(stats_by_scope["all_baseline_runs"], hardware, path, evaluated, space)
        _add_snapshot(all_stats, hardware, path, evaluated, space)
    serialized = {
        name: _serialize_stats(stats, space, catalog, joint)
        for name, stats in sorted(stats_by_scope.items())
    }
    serialized["all_paper_runs"] = _serialize_stats(all_stats, space, catalog, joint)
    report = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "scope_note": "All iter_*/hardware.json snapshots under paper_experiments_20261116.",
        "scopes": serialized,
    }
    output_root = EXPERIMENT_ROOT / FORMAL_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "design_space_coverage_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "design_space_coverage_audit.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "hardware_snapshots": len(hardware_paths),
        "output_json": str(output_root / "design_space_coverage_audit.json"),
        "output_md": str(output_root / "design_space_coverage_audit.md"),
        "all": serialized["all_paper_runs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
