"""SimPhony simulator adapter for the photonic DSE framework.

This adapter bridges the photonic ``hardware`` dict used by the agents and the
SimPhony simulator.  It supports two execution modes:

``analytic`` (default)
    A pure-Python, formula-level mirror of SimPhony's workflows (see
    :mod:`photon_tuner.simulators.analytic`).  Requires only PyYAML and lets the
    whole multi-agent loop run offline.

``real``
    Runs the actual :class:`onnarchsim.simulator.ONNArchSimulator` when
    PyTorch / TorchONN / pyutils are installed.  The simulator's new
    ``run_full_report`` method produces the same fine-grained JSON schema.

Both modes write the same artefacts into the run directory:
``hardware.json``, ``report.json``, per-workflow detail JSON files, and
``metrics.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from photon_tuner.common.io import write_json
from photon_tuner.core.objective import metrics_with_objective
from photon_tuner.core.schemas import EvaluationResult
from photon_tuner.core.search_space import materialize_hardware
from photon_tuner.core.workload import DEFAULT_WORKLOAD, total_macs
from photon_tuner.simulators.analytic import evaluate as analytic_evaluate
from photon_tuner.simulators.base import SimulatorAdapter


# Per model-family mapping configs.  ``transformer`` covers the shipped
# BERT (QBertViTBase) and MobileViT models (QConv2d + QLinear + QMatMul);
# ``cnn`` covers conv/linear-only models.
MODEL_TYPE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "transformer": {
        "nn_conversion_cfg": "configs/nn_mapping/simple_transformer.yml",
        "model2arch_map_cfg": "configs/architecture_mapping/simple_lt_transformer.yml",
    },
    "cnn": {
        "nn_conversion_cfg": "configs/nn_mapping/simple_cnn.yml",
        "model2arch_map_cfg": "configs/architecture_mapping/simple_lt_cnn.yml",
    },
}


class SimphonyAdapter(SimulatorAdapter):
    """Adapter that evaluates photonic hardware with SimPhony (or its mirror)."""

    def __init__(
        self,
        simphony_root: Path | str,
        output_root: Path | str,
        search_space: Optional[Dict[str, Any]] = None,
        mode: str = "auto",
        # Real-mode (ONNArchSimulator) construction inputs.
        nn_model: Any = None,
        nn_model_factory: Optional[str] = None,
        onn_conversion_cfg: Optional[str] = None,
        nn_conversion_cfg: Optional[str] = None,
        model2arch_map_cfg: Optional[str] = None,
        model_type: str = "transformer",
        arch_cfg_file: str = "configs/design/architectures/LT_hetero.yml",
        arch_version: str = "v1",
        input_shape: tuple = (2, 3, 224, 224),
        # Analytic-mode workload selection (see configs/workloads.json).
        workload_name: Optional[str] = None,
    ) -> None:
        self.simphony_root = Path(simphony_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.search_space = search_space or {}
        self.mode = mode
        self.nn_model = nn_model
        self.nn_model_factory = nn_model_factory
        self.onn_conversion_cfg = onn_conversion_cfg
        mapping = MODEL_TYPE_MAPPINGS.get(model_type, MODEL_TYPE_MAPPINGS["transformer"])
        self.nn_conversion_cfg = nn_conversion_cfg or mapping["nn_conversion_cfg"]
        self.model2arch_map_cfg = model2arch_map_cfg or mapping["model2arch_map_cfg"]
        self.model_type = model_type
        self.arch_cfg_file = arch_cfg_file
        self.arch_version = arch_version
        self.input_shape = input_shape
        # In ``real`` mode the workload is taken from the actual ONN model, so
        # this only steers the offline analytic mirror.
        self.workload_name = workload_name or DEFAULT_WORKLOAD

    # -- SimulatorAdapter interface ---------------------------------------- #

    def evaluate(self, hardware: Dict[str, Any], iteration: int) -> EvaluationResult:
        hardware = materialize_hardware(hardware)
        run_dir = self.output_root / f"iter_{iteration:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "hardware.json", hardware)

        report = self._run_report(hardware, run_dir)
        self._write_report_files(run_dir, report)

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
        metrics = metrics_with_objective(metrics, hardware, self.search_space)
        write_json(run_dir / "metrics.json", metrics)

        return EvaluationResult(
            run_dir=run_dir,
            hardware=hardware,
            metrics=metrics,
            detail_files={
                "report": run_dir / "report.json",
                "energy": run_dir / "energy_detail.json",
                "area": run_dir / "area_detail.json",
                "insertion_loss": run_dir / "insertion_loss.json",
                "workload": run_dir / "workload.json",
                "memory": run_dir / "memory.json",
            },
            raw_files={
                "hardware": run_dir / "hardware.json",
                "metrics": run_dir / "metrics.json",
            },
        )

    def schema(self) -> Dict[str, Any]:
        return {
            "adapter": "simphony",
            "mode": self.mode,
            "workload": self.workload_name,
            "search_space": self.search_space,
            "tunable_hardware_fields": {
                "optical_structure": [
                    "tiles",
                    "cores_per_tile",
                    "core_height",
                    "core_width",
                    "num_wavelength",
                    "work_freq",
                    "dataflow",
                    "precision_in_bit",
                    "precision_w_bit",
                    "precision_out_bit",
                ],
                "device_choices": ["devices.core.*", "devices.node.*"],
            },
            "detail_schema": {
                "report": "report.json — per-device / per-chip / per-layer energy, area, insertion loss, latency, memory breakdowns.",
                "energy": "energy_detail.json — per-device dynamic/static energy (pJ).",
                "area": "area_detail.json — per-device area (um^2).",
                "insertion_loss": "insertion_loss.json — node/core longest optical path (dB).",
                "workload": "workload.json — workload name/kind plus per-layer partition cycles and layer sizes.",
                "memory": "memory.json — per-layer memory latency/energy and hierarchy spec.",
            },
        }

    # -- Internal helpers --------------------------------------------------- #

    def _run_report(self, hardware: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        if self.mode == "analytic":
            return self._run_analytic(hardware)
        if self.mode == "real":
            return self._run_real_report(hardware, run_dir)
        # auto: prefer the real simulator, fall back to the analytic mirror.
        try:
            return self._run_real_report(hardware, run_dir)
        except (ImportError, ModuleNotFoundError):
            return self._run_analytic(hardware)

    def _run_analytic(self, hardware: Dict[str, Any]) -> Dict[str, Any]:
        """Analytic mirror entry point.

        Kept as the single call site so every path (``analytic`` and the
        ``auto`` fallback) forwards the selected workload -- an earlier version
        duplicated the call and the ``auto`` fallback silently dropped it.
        """
        return analytic_evaluate(
            hardware,
            str(self.simphony_root),
            workload_name=self.workload_name,
        )

    def _run_real_report(self, hardware: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Run the real ONNArchSimulator and return its fine-grained report."""
        _, report = self.evaluate_with_readback(hardware, run_dir)
        return report

    def evaluate_with_readback(
        self, hardware: Dict[str, Any], run_dir: Path
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run one real-mode eval, returning ``(injected_sub_arch, report)``.

        The injected sub-arch is read back from the simulator *after*
        ``_inject_hardware`` so callers can confirm the searched parameter truly
        landed (the core of the validity verification harness).
        """
        try:
            from onnarchsim.simulator import ONNArchSimulator
        except ImportError as exc:
            raise ImportError(
                "real SimPhony mode requires torchonn/pyutils (not installed); "
                "use --mode analytic for the offline mirror."
            ) from exc

        nn_model = _resolve_nn_model(self.nn_model, self.nn_model_factory)
        if nn_model is None:
            raise ImportError(
                "real SimPhony mode requires an ONN/nn model: pass it to the "
                "adapter, or set --nn-model-factory (module:function)."
            )

        # SimPhony resolves ``configs/nn_mapping/...``, ``./cacti`` and
        # ``./configs/memory`` relative to the process CWD, so run the simulator
        # from the SimPhony checkout root while keeping every run output
        # (``run_dir``) on an absolute path.
        cwd = os.getcwd()
        os.chdir(self.simphony_root)
        try:
            simulator = ONNArchSimulator(
                nn_model=nn_model,
                onn_conversion_cfg=self.onn_conversion_cfg,
                nn_conversion_cfg=self.nn_conversion_cfg,
                onn_model=None,
                model2arch_map_cfg=self.model2arch_map_cfg,
                devicelib_root=str(self.simphony_root / "configs" / "devices"),
                device_cfg_files=["*/*.yml"],
                arch_cfg_file=str(self.simphony_root / self.arch_cfg_file),
                arch_version=self.arch_version,
                input_shape=self.input_shape,
                log_path=str(run_dir / "simphony.log"),
            )
            _inject_hardware(simulator, hardware, str(self.simphony_root))
            injected = read_sub_arch(simulator)
            report = simulator.run_full_report(
                output_path=str(run_dir / "report.json")
            )
        finally:
            os.chdir(cwd)
        # ``run_full_report`` returns the report already; the JSON is written too.
        if not isinstance(report, dict):
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        return injected, report

    def _write_report_files(self, run_dir: Path, report: Dict[str, Any]) -> None:
        """Split the full report into per-workflow detail files."""
        write_json(run_dir / "report.json", report)
        write_json(run_dir / "energy_detail.json", report.get("energy", {}))
        write_json(run_dir / "area_detail.json", report.get("area", {}))
        write_json(run_dir / "insertion_loss.json", report.get("insertion_loss", {}))
        write_json(run_dir / "workload.json", report.get("workload", {}))
        write_json(run_dir / "memory.json", report.get("memory", {}))


def _inject_hardware(
    simulator: Any,
    hardware: Dict[str, Any],
    simphony_root: str,
) -> None:
    """Inject the searched hardware dict into a freshly built ONNArchSimulator.

    ONNArchSimulator reads its architecture from fixed YAML files and ignores
    the searched ``hardware`` dict entirely, so without this step real-mode
    evaluations would always simulate the same LT baseline.  We rebuild the
    sub-architecture with the analytic ``build_sub_arch`` (the same code path as
    offline mode) and replace the simulator's resolved sub-architecture plus the
    per-layer miniblock and layer->sub-arch mapping, so tiles / cores / geometry
    / frequency / precision / memory / devices and the L4 topology overrides all
    take effect.
    """
    from photon_tuner.simulators.analytic import (
        build_sub_arch,
        load_device_catalog,
    )

    catalog = load_device_catalog(f"{simphony_root}/configs/devices")
    sub_arch = build_sub_arch(hardware, simphony_root, catalog)

    name = next(iter(getattr(simulator, "sub_arch_dictionary", {}) or {}), None)
    if name is None:
        raise ValueError("ONNArchSimulator has no sub-architecture to inject into")
    simulator.sub_arch_dictionary[name] = sub_arch

    tiles = int(sub_arch["tiles"])
    cores = int(sub_arch["cores_per_tile"])
    height = int(sub_arch["core"]["height"])
    width = int(sub_arch["core"]["width"])
    for layer in simulator.sub_arch_to_layer_mapping.get(name, []):
        # miniblock order is [tiles, cores_per_tile, height, width].
        simulator.arch_blocks[layer] = [tiles, cores, height, width]
        simulator.layer_to_sub_arch_mapping[layer] = sub_arch


def read_sub_arch(simulator: Any) -> Dict[str, Any]:
    """Return the injected sub-architecture dict from a built simulator.

    ``_inject_hardware`` replaces ``simulator.sub_arch_dictionary[name]`` with
    the ``build_sub_arch`` output; this helper reads that state back out so the
    verification harness can confirm a searched parameter actually landed.
    """
    dictionary = getattr(simulator, "sub_arch_dictionary", {}) or {}
    name = next(iter(dictionary), None)
    if name is None:
        return {}
    return dict(dictionary[name])


def _resolve_nn_model(nn_model: Any, factory: Optional[str]) -> Any:
    """Return the ONN/nn model, loading it from a ``module:function`` factory
    string when an object was not passed directly."""
    if nn_model is not None:
        return nn_model
    if not factory:
        return None
    if ":" not in factory:
        raise ValueError("--nn-model-factory must be 'module.path:function_name'")
    module_name, function_name = factory.split(":", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, function_name)()
