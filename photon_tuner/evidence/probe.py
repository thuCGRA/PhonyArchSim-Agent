"""Single-variable ablation probe (the deterministic verification harness).

For every (parameter, candidate value, workload) it perturbs the baseline
hardware, checks constraints, injects into SimPhony, *reads back* the injected
sub-architecture to confirm the parameter landed, simulates, and records which
metrics moved.  No LLM is involved -- this is the measurement layer; the
:class:`~photon_tuner.agents.evidence.EvidenceAnalystAgent` interprets the
resulting evidence afterwards.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from photon_tuner.core.search_space import DesignSpace
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.workload import total_macs
from photon_tuner.core.search_space import (
    default_hardware,
    materialize_hardware,
    set_device_choice,
)
from photon_tuner.evidence.store import EvidenceStore
from photon_tuner.evidence.sub_arch_map import (
    SUB_ARCH_MAP,
    confirm_device_injection,
    confirm_injection,
)
from photon_tuner.evidence.workload_features import workload_features


def _memory_energy_pj(report: Dict[str, Any]) -> float:
    """Total memory energy (pJ). Not part of the objective; recorded separately."""
    mem = report.get("memory")
    if not isinstance(mem, dict):
        return 0.0
    energy = mem.get("energy")
    if not isinstance(energy, dict):
        return 0.0
    lt = energy.get("LT")
    if not isinstance(lt, dict):
        return 0.0
    return sum(float(v) for v in lt.values() if isinstance(v, (int, float)))


def _pct(after: float, baseline: float) -> float:
    try:
        if float(baseline) == 0.0:
            return 0.0
        return (float(after) - float(baseline)) / float(baseline) * 100.0
    except (TypeError, ValueError):
        return 0.0


def _metric_deltas(baseline: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    return {
        "energy_pct": _pct(after["energy_pj"], baseline["energy_pj"]),
        "area_pct": _pct(after["area_um2"], baseline["area_um2"]),
        "latency_pct": _pct(after["latency_s"], baseline["latency_s"]),
        "objective_pct": _pct(after["objective"], baseline["objective"]),
    }


class SearchSpaceProbe:
    """Deterministic ablation harness wired to a real-mode ``SimphonyAdapter``."""

    def __init__(
        self,
        adapter: Any,
        design_space: DesignSpace,
        search_space: Dict[str, Any],
        store: EvidenceStore,
        output_root: Path | str,
        device_variants: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.adapter = adapter
        self.design_space = design_space
        self.search_space = search_space
        self.store = store
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device_variants = device_variants or {}
        self._run_seq = 0

    # -- evaluation -------------------------------------------------------- #

    def evaluate(self, hardware: Dict[str, Any], tag: str) -> Dict[str, Any]:
        """Inject + read back + simulate one hardware config."""
        hw = materialize_hardware(hardware)
        run_dir = self.output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        sub_arch, report = self.adapter.evaluate_with_readback(hw, run_dir)
        energy_pj = float(report["metrics"]["energy_pj"])
        macs = total_macs(report["workload"]["partition_cycles"]["LT"])
        metrics = {
            "latency_s": float(report["metrics"]["latency_s"]),
            "energy_pj": energy_pj,
            "area_um2": float(report["metrics"]["area_um2"]),
            "total_macs": macs,
            "pJ_per_mac": energy_pj / macs if macs else energy_pj,
            "core_insertion_loss": float(
                report["insertion_loss"]["LT"]["core_insertion_loss"]
            ),
        }
        metrics = metrics_with_objective(metrics, hw, self.search_space)
        return {
            "sub_arch": sub_arch,
            "report": report,
            "metrics": metrics,
            "memory_energy_pj": _memory_energy_pj(report),
            "run_dir": run_dir,
            "hardware": hw,
        }

    def run_baseline(
        self, hardware: Dict[str, Any], workload_name: str, tag: str = "baseline"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate the baseline once and extract workload features."""
        result = self.evaluate(hardware, tag)
        features = workload_features(workload_name, result["report"])
        return result, features

    # -- perturbation ------------------------------------------------------ #

    def _record(
        self,
        param: str,
        layer: str,
        workload_name: str,
        features: Dict[str, Any],
        old_value: Any,
        new_value: Any,
        hardware_map: List[str],
        sub_arch_map: List[Any],
        constraint_result: str,
        clamp_before: Any,
        baseline_eval: Dict[str, Any],
        after_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        deltas = _metric_deltas(baseline_eval["metrics"], after_eval["metrics"])
        affected = [m for m, d in deltas.items() if abs(d) > 0.01]
        mem_delta = _pct(
            after_eval["memory_energy_pj"], baseline_eval["memory_energy_pj"]
        )
        record = {
            "param": param,
            "layer": layer,
            "workload": workload_name,
            "workload_features": features,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "hardware_map": hardware_map,
            "sub_arch_map": sub_arch_map,
            "constraint_result": constraint_result,
            "clamp_before": None if clamp_before is None else str(clamp_before),
            "injection_confirmed": 1,
            "injection_readback": {},
            "baseline_energy_pj": baseline_eval["metrics"]["energy_pj"],
            "baseline_area_um2": baseline_eval["metrics"]["area_um2"],
            "baseline_latency_s": baseline_eval["metrics"]["latency_s"],
            "baseline_objective": baseline_eval["metrics"]["objective"],
            "after_energy_pj": after_eval["metrics"]["energy_pj"],
            "after_area_um2": after_eval["metrics"]["area_um2"],
            "after_latency_s": after_eval["metrics"]["latency_s"],
            "after_objective": after_eval["metrics"]["objective"],
            "memory_energy_delta_pct": mem_delta,
            "metric_deltas": deltas,
            "affected_metrics": affected,
            "is_null": 1 if not affected else 0,
            "trustworthy": 0,
        }
        return record

    def _next_tag(self, workload_name: str, param: str, value: Any) -> str:
        self._run_seq += 1
        safe = str(value).replace("/", "_").replace(":", "_")
        return f"{workload_name}/{param}={safe}/run_{self._run_seq:03d}"

    def probe_arch_param(
        self,
        param: str,
        baseline_hw: Dict[str, Any],
        baseline_eval: Dict[str, Any],
        features: Dict[str, Any],
        workload_name: str,
        max_values: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Probe every candidate value of one architecture parameter."""
        layer = self.design_space.layer_of(param)
        candidates = self.design_space.candidates(param)
        old_value = self.design_space.get(baseline_hw, param)
        hardware_map = list(self.design_space.param_spec(param)["map"])
        sub_arch_map = copy.deepcopy(SUB_ARCH_MAP.get(param, []))

        records: List[Dict[str, Any]] = []
        for value in candidates:
            if value == old_value:
                continue
            if max_values is not None and len(records) >= max_values:
                break

            perturbed = copy.deepcopy(baseline_hw)
            self.design_space.set(perturbed, param, value)

            violations = self.design_space.violations(perturbed)
            normalized = self.design_space.normalize(perturbed)
            eff_value = self.design_space.get(normalized, param)

            if eff_value != value:
                constraint_result = "violated" if violations else "clamped"
                clamp_before = value
            else:
                constraint_result = "violated" if violations else "pass"
                clamp_before = None

            after_eval = self.evaluate(
                normalized, self._next_tag(workload_name, param, eff_value)
            )
            confirmed, readback = confirm_injection(
                after_eval["sub_arch"], param, eff_value
            )

            record = self._record(
                param=param,
                layer=layer or "arch",
                workload_name=workload_name,
                features=features,
                old_value=old_value,
                new_value=eff_value,
                hardware_map=hardware_map,
                sub_arch_map=sub_arch_map,
                constraint_result=constraint_result,
                clamp_before=clamp_before,
                baseline_eval=baseline_eval,
                after_eval=after_eval,
            )
            record["injection_confirmed"] = 1 if confirmed else 0
            record["injection_readback"] = readback
            record["trustworthy"] = 1 if confirmed else 0
            self.store.record(record)
            records.append(record)
        return records

    def probe_device(
        self,
        level: str,
        device_type: str,
        baseline_hw: Dict[str, Any],
        baseline_eval: Dict[str, Any],
        features: Dict[str, Any],
        workload_name: str,
        max_values: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Probe every library variant of one device type at ``core``/``node``."""
        candidates = self.device_variants.get(device_type, [])
        old_value = baseline_hw.get("devices", {}).get(level, {}).get(device_type)
        hardware_map = ["devices", level, device_type]
        sub_arch_map = ["core"] + (["node"] if level == "node" else []) + [
            "devices", device_type,
        ]

        records: List[Dict[str, Any]] = []
        for value in candidates:
            if value == old_value:
                continue
            if max_values is not None and len(records) >= max_values:
                break

            perturbed = set_device_choice(
                baseline_hw, level, device_type, str(value)
            )
            after_eval = self.evaluate(
                perturbed, self._next_tag(workload_name, f"devices.{level}.{device_type}", value)
            )
            confirmed, readback = confirm_device_injection(
                after_eval["sub_arch"], level, device_type, value
            )

            record = self._record(
                param=f"devices.{level}.{device_type}",
                layer="device",
                workload_name=workload_name,
                features=features,
                old_value=old_value,
                new_value=str(value),
                hardware_map=hardware_map,
                sub_arch_map=sub_arch_map,
                constraint_result="pass",
                clamp_before=None,
                baseline_eval=baseline_eval,
                after_eval=after_eval,
            )
            record["injection_confirmed"] = 1 if confirmed else 0
            record["injection_readback"] = readback
            record["trustworthy"] = 1 if confirmed else 0
            self.store.record(record)
            records.append(record)
        return records
