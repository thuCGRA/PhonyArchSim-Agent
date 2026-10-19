"""Hierarchical, multi-granularity, adaptive design space (real-mode oriented).

This module supersedes the flat ``DEFAULT_ARCH_SEARCH_SPACE`` with a layered
search space that mirrors the physical decomposition of a photonic accelerator:

1. ``system``       -- 系统架构层: tiles / cores / dataflow / precision / memory
2. ``optical_core`` -- 光学计算核心层: crossbar geometry / wavelengths / clock
3. ``device``       -- 器件选择层: library-variant selection per device type
4. ``device_param`` -- 器件参数层: library cfgs, FIXED (selection-only; no override)

Three properties, as requested:

* **Hierarchical** -- parameters are grouped into the four layers above and the
  hardware paths are declared explicitly (``map``) so a param always knows which
  layer it belongs to and where it lives in the ``hardware`` dict.
* **Multi-granularity** -- every numeric parameter declares a coarse ``values``
  grid plus a ``step``; ``kind`` distinguishes ``int`` (linear), ``bits``
  (precision) and ``enum`` (dataflow) parameters.
* **Adaptive** -- :meth:`DesignSpace.refine` narrows the grid around a design
  point, subdividing the step so the search can zoom in after coarse exploration.

Real-mode orientation is captured by two mechanisms:

* ``mode`` annotation (``both`` vs ``real``) -- ``real`` parameters (e.g. GLB/RF
  cache sizes) only affect the real CACTI memory model, not the analytic mirror.
* :meth:`DesignSpace.violations` / ``normalize`` enforce the exact feasibility
  asserts that real SimPhony's ``energy.py`` performs and the analytic mirror
  dropped: ``work_freq <= MZM symbol rate``, ``in_bit <= DAC prec``,
  ``out_bit <= ADC prec``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Declarative layered spec (default; loadable/overridable from JSON).
# --------------------------------------------------------------------------- #

# ``map`` is the path from the root of a ``hardware`` dict to the parameter.
# ``mode``: "both" = consumed by analytic mirror AND real SimPhony;
#          "real" = consumed only by the real CACTI memory model.
# ``refine`` marks linear numeric params that can be zoomed adaptively.
LAYERED_DESIGN_SPACE: Dict[str, Any] = {
    "version": 2,
    "layers": {
        "system": {
            "label": "system (系统架构层)",
            "params": {
                "tiles": {"kind": "int", "values": [1, 2, 4, 8], "step": 1,
                          "map": ["arch", "system", "tiles"], "mode": "both", "refine": False},
                "cores_per_tile": {"kind": "int", "values": [1, 2, 4], "step": 1,
                                   "map": ["arch", "system", "cores_per_tile"], "mode": "both", "refine": False},
                "dataflow": {"kind": "enum",
                             "values": ["output_stationary", "weight_stationary", "input_stationary"],
                             "map": ["arch", "system", "dataflow"], "mode": "both"},
                "precision_in_bit": {"kind": "bits", "values": [2, 4, 8],
                                     "map": ["arch", "optical_core", "precision", "in_bit"], "mode": "both",
                                     "refine": False, "constraint": "in_bit <= dac.prec"},
                "precision_w_bit": {"kind": "bits", "values": [2, 4, 8],
                                    "map": ["arch", "optical_core", "precision", "w_bit"], "mode": "both",
                                    "refine": False},
                "precision_out_bit": {"kind": "bits", "values": [2, 4, 8],
                                      "map": ["arch", "optical_core", "precision", "out_bit"], "mode": "both",
                                      "refine": False, "constraint": "out_bit <= adc.prec"},
                "hbm_bandwidth": {"kind": "int", "values": [512, 1024, 2048], "step": 128,
                                  "map": ["arch", "memory", "HBM", "bandwidth"], "mode": "both",
                                  "refine": True},
                "glb1_cache_size": {"kind": "int", "values": [64, 256, 485, 1024, 4096], "step": 64,
                                    "map": ["arch", "memory", "GLB1", "cache_size"], "mode": "real",
                                    "refine": True},
                "glb2_cache_size": {"kind": "int", "values": [64, 256, 485, 1024, 4096], "step": 64,
                                    "map": ["arch", "memory", "GLB2", "cache_size"], "mode": "real",
                                    "refine": True},
                "rf_cache_size": {"kind": "int", "values": [64, 256, 485, 1024, 4096], "step": 64,
                                  "map": ["arch", "memory", "RF", "cache_size"], "mode": "real",
                                  "refine": True},
            },
        },
        "optical_core": {
            "label": "optical_core (光学计算核心层)",
            "params": {
                "core_height": {"kind": "int", "values": [8, 12, 16], "step": 2,
                                "map": ["arch", "optical_core", "height"], "mode": "both", "refine": True},
                "core_width": {"kind": "int", "values": [8, 12, 16], "step": 2,
                               "map": ["arch", "optical_core", "width"], "mode": "both", "refine": True},
                "num_wavelength": {"kind": "int", "values": [8, 12, 16], "step": 2,
                                   "map": ["arch", "optical_core", "num_wavelength"], "mode": "both", "refine": True},
                "work_freq": {"kind": "int", "values": [5, 10, 15, 20, 25], "step": 5,
                              "map": ["arch", "optical_core", "work_freq"], "mode": "both", "refine": True,
                              "constraint": "work_freq <= mzm.symbol_rate"},
            },
        },
        "topology": {
            "label": "topology (拓扑与电路层)",
            "params": {
                "sharing_policy": {
                    "kind": "enum",
                    "values": ["no_sharing", "adc_shared", "adc_tia_shared"],
                    "map": ["arch", "topology", "sharing_policy"], "mode": "both",
                },
                "temporal_accum": {
                    "kind": "enum",
                    "values": ["off", "default", "deep"],
                    "map": ["arch", "topology", "temporal_accum"], "mode": "both",
                },
                "laser_distribution": {
                    "kind": "enum",
                    "values": ["H*W", "H*W*N", "H*W*N*C"],
                    "map": ["arch", "topology", "laser_distribution"], "mode": "both",
                },
            },
        },
        "device": {
            "label": "device (器件选择层)",
            "kind": "selection",
            "params": {
                # Instance-level device-type switches (not library types):
                # laser_source toggles laser instance i4 on/off-chip;
                # splitter_kind toggles fan-out splitter i6 laser_splitter/mmi.
                "laser_source": {
                    "kind": "enum",
                    "values": ["on_chip", "off_chip"],
                    "map": ["devices", "core", "laser_source"], "mode": "both",
                },
                "splitter_kind": {
                    "kind": "enum",
                    "values": ["laser_splitter", "mmi"],
                    "map": ["devices", "core", "splitter_kind"], "mode": "both",
                },
            },
        },
        "device_param": {"label": "device_param (器件参数层)", "kind": "fixed",
                         "note": "library cfgs are fixed; only variant selection is searchable"},
    },
}


def _lookup(hw: Dict[str, Any], path: Iterable[str]) -> Any:
    node: Any = hw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _assign(hw: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    node = hw
    *parents, leaf = list(path)
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


class DesignSpace:
    """A hierarchical, multi-granularity, adaptive search space."""

    def __init__(
        self,
        spec: Optional[Dict[str, Any]] = None,
        device_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.spec = copy.deepcopy(spec or LAYERED_DESIGN_SPACE)
        self.layers: Dict[str, Any] = self.spec.get("layers", {})
        # {device_type: {device_name: {"cfgs": {...}, "type": ...}}}
        self.device_catalog: Dict[str, Any] = device_catalog or {}

    # -- introspection ----------------------------------------------------- #

    def _iter_params(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        for layer_name, layer in self.layers.items():
            for param_name, spec in (layer.get("params") or {}).items():
                yield layer_name, param_name, spec

    def params(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(self._iter_params())

    def layer_of(self, param: str) -> Optional[str]:
        for layer_name, param_name, _ in self._iter_params():
            if param_name == param:
                return layer_name
        return None

    def param_spec(self, param: str) -> Dict[str, Any]:
        for _, param_name, spec in self._iter_params():
            if param_name == param:
                return spec
        raise KeyError(f"Unknown design-space parameter: {param}")

    def candidates(self, param: str) -> List[Any]:
        spec = self.param_spec(param)
        return list(spec.get("values", []))

    # -- hardware get/set -------------------------------------------------- #

    def get(self, hw: Dict[str, Any], param: str) -> Any:
        return _lookup(hw, self.param_spec(param)["map"])

    def set(self, hw: Dict[str, Any], param: str, value: Any) -> None:
        _assign(hw, self.param_spec(param)["map"], value)

    # -- feasibility (real SimPhony asserts) ------------------------------- #

    def _device_cfg(self, hw: Dict[str, Any], device_type: str) -> Dict[str, Any]:
        devices = hw.get("devices", {})
        name = (devices.get("core", {}).get(device_type)
                or devices.get("node", {}).get(device_type))
        if not name:
            return {}
        variant = self.device_catalog.get(device_type, {}).get(str(name), {})
        return dict(variant.get("cfgs", {}) or {})

    def max_work_freq(self, hw: Dict[str, Any]) -> Optional[float]:
        """MZM symbol-rate ceiling (modulation_speed / testing_bit), in GHz."""
        cfg = self._device_cfg(hw, "mzm")
        if not cfg or "modulation_speed" not in cfg:
            return None
        testing_bit = float(cfg.get("testing_bit", 1) or 1)
        return float(cfg["modulation_speed"]) / testing_bit

    def min_work_freq_ceiling(self, hw: Dict[str, Any]) -> Optional[float]:
        """Tightest device clock ceiling, in GHz.

        ``min(MZM symbol rate, DAC sample_rate, ADC sample_rate, photodetector
        bandwidth)`` -- the converters and the detector are independent caps,
        and the tightest one is what ``normalize`` must clamp to.
        """
        ceilings: List[float] = []
        mzm = self.max_work_freq(hw)
        if mzm is not None:
            ceilings.append(mzm)
        for device_type, cfg_key in (
            ("dac", "sample_rate"),
            ("adc", "sample_rate"),
            ("photodetector", "bandwidth"),
        ):
            cfg = self._device_cfg(hw, device_type)
            limit = cfg.get(cfg_key)
            if limit is not None:
                ceilings.append(float(limit))
        return min(ceilings) if ceilings else None

    def violations(self, hw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return the feasibility constraints a hardware dict violates."""
        out: List[Dict[str, Any]] = []
        core = hw.get("arch", {}).get("optical_core", {})
        precision = core.get("precision", {})

        max_freq = self.max_work_freq(hw)
        freq = core.get("work_freq")
        if max_freq is not None and freq is not None and float(freq) > max_freq:
            out.append({
                "param": "work_freq",
                "constraint": "work_freq <= mzm.symbol_rate",
                "current": float(freq),
                "limit": max_freq,
                "message": f"work_freq {freq} GHz exceeds MZM symbol rate {max_freq} GHz",
            })

        # Converter sample rates and detector bandwidth are independent clock
        # ceilings (the MZM check above is only the modulator's own symbol-rate
        # cap).  These bound the free-lunch ``work_freq`` knob physically.
        for device_type, cfg_key in (("dac", "sample_rate"), ("adc", "sample_rate")):
            dev_cfg = self._device_cfg(hw, device_type)
            limit = dev_cfg.get(cfg_key)
            if limit is not None and freq is not None and float(freq) > float(limit):
                out.append({
                    "param": "work_freq",
                    "constraint": f"work_freq <= {device_type}.{cfg_key}",
                    "current": float(freq),
                    "limit": float(limit),
                    "message": f"work_freq {freq} GHz exceeds {device_type} {cfg_key} {limit} GHz",
                })

        pd_cfg = self._device_cfg(hw, "photodetector")
        pd_bw = pd_cfg.get("bandwidth")
        if pd_bw is not None and freq is not None and float(freq) > float(pd_bw):
            out.append({
                "param": "work_freq",
                "constraint": "work_freq <= photodetector.bandwidth",
                "current": float(freq),
                "limit": float(pd_bw),
                "message": f"work_freq {freq} GHz exceeds photodetector bandwidth {pd_bw} GHz",
            })

        dac_cfg = self._device_cfg(hw, "dac")
        in_bit = precision.get("in_bit")
        if "prec" in dac_cfg and in_bit is not None and int(in_bit) > int(dac_cfg["prec"]):
            out.append({
                "param": "precision_in_bit",
                "constraint": "in_bit <= dac.prec",
                "current": int(in_bit),
                "limit": int(dac_cfg["prec"]),
                "message": f"in_bit {in_bit} exceeds DAC precision {dac_cfg['prec']}",
            })

        adc_cfg = self._device_cfg(hw, "adc")
        out_bit = precision.get("out_bit")
        if "prec" in adc_cfg and out_bit is not None and int(out_bit) > int(adc_cfg["prec"]):
            out.append({
                "param": "precision_out_bit",
                "constraint": "out_bit <= adc.prec",
                "current": int(out_bit),
                "limit": int(adc_cfg["prec"]),
                "message": f"out_bit {out_bit} exceeds ADC precision {adc_cfg['prec']}",
            })

        return out

    # -- normalization ----------------------------------------------------- #

    def _clamp_to_candidates(self, hw: Dict[str, Any], param: str) -> None:
        candidates = self.candidates(param)
        if not candidates:
            return
        spec = self.param_spec(param)
        if spec.get("kind") == "enum":
            current = str(self.get(hw, param))
            if current not in [str(c) for c in candidates]:
                self.set(hw, param, candidates[0])
            return
        try:
            current = float(self.get(hw, param))
        except (TypeError, ValueError):
            return
        best = min(candidates, key=lambda c: abs(float(c) - current))
        self.set(hw, param, best)

    def normalize(self, hw: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp every searchable param to candidates and enforce feasibility."""
        result = copy.deepcopy(hw)
        for _, param, _ in self._iter_params():
            self._clamp_to_candidates(result, param)

        # Feasibility: clamp to the largest candidate that stays within the
        # tightest device ceiling (mirrors real SimPhony's asserts instead of
        # crashing).
        max_freq = self.min_work_freq_ceiling(result)
        if max_freq is not None:
            feasible = [c for c in self.candidates("work_freq") if float(c) <= max_freq]
            if feasible:
                current = float(result["arch"]["optical_core"].get("work_freq", 0))
                result["arch"]["optical_core"]["work_freq"] = min(
                    feasible, key=lambda c: abs(float(c) - current)
                )
        for param, device in (("precision_in_bit", "dac"), ("precision_out_bit", "adc")):
            cfg = self._device_cfg(result, device)
            if "prec" in cfg:
                feasible = [c for c in self.candidates(param) if int(c) <= int(cfg["prec"])]
                if feasible:
                    current = int(result["arch"]["optical_core"]["precision"].get(
                        param.replace("precision_", ""), 0))
                    result["arch"]["optical_core"]["precision"][param.replace("precision_", "")] = min(
                        feasible, key=lambda c: abs(int(c) - current)
                    )
        return result

    # -- editing ----------------------------------------------------------- #

    def step(self, hw: Dict[str, Any], param: str, direction: int) -> Dict[str, Any]:
        """Step one parameter to the next/previous candidate (normalized)."""
        result = copy.deepcopy(hw)
        candidates = self.candidates(param)
        if not candidates:
            return result
        spec = self.param_spec(param)
        if spec.get("kind") == "enum":
            current = str(self.get(result, param))
            if current not in [str(c) for c in candidates]:
                self.set(result, param, candidates[0])
            else:
                index = [str(c) for c in candidates].index(current)
                index = max(0, min(len(candidates) - 1, index + direction))
                self.set(result, param, candidates[index])
            return self.normalize(result)
        ordered = sorted(set(candidates), key=lambda v: float(v))
        try:
            current = float(self.get(result, param))
        except (TypeError, ValueError):
            self.set(result, param, ordered[0])
            return self.normalize(result)
        if current not in [float(v) for v in ordered]:
            self.set(result, param, min(ordered, key=lambda v: abs(float(v) - current)))
        else:
            index = [float(v) for v in ordered].index(current)
            index = max(0, min(len(ordered) - 1, index + direction))
            self.set(result, param, ordered[index])
        return self.normalize(result)

    def modify(self, hw: Dict[str, Any], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply explicit ``{parameter, value}`` updates (then normalize)."""
        result = copy.deepcopy(hw)
        for update in updates:
            param = str(update.get("parameter", ""))
            value = update.get("value")
            spec = self.param_spec(param)  # raises KeyError on unknown param
            if spec.get("kind") == "enum":
                self.set(result, param, str(value))
            else:
                self.set(result, param, int(round(float(value))))
        return self.normalize(result)

    # -- adaptive refinement ---------------------------------------------- #

    def refine(self, center_hw: Dict[str, Any], factor: int = 2) -> "DesignSpace":
        """Return a finer-grained copy of this space centered on ``center_hw``.

        Linear ``refine`` params get a subdivided step over a local window
        ``[center - step, center + step]``; geometric / enum params keep their
        grid.
        """
        new = copy.deepcopy(self)
        new.device_catalog = self.device_catalog
        for layer_name, param_name, spec in new._iter_params():
            if not spec.get("refine") or spec.get("kind") not in ("int", "bits"):
                continue
            values = [float(v) for v in spec.get("values", [])]
            if not values:
                continue
            step = float(spec.get("step", 1))
            try:
                center = float(self.get(center_hw, param_name))
            except (TypeError, ValueError, KeyError):
                continue
            new_step = max(1.0, step / max(1, factor))
            lo = max(min(values), center - step)
            hi = min(max(values), center + step)
            refined: List[float] = []
            v = lo
            while v <= hi + 1e-9:
                refined.append(round(v))
                v += new_step
            if center not in refined:
                refined.append(round(center))
            spec["values"] = sorted(set(int(v) for v in refined))
            spec["step"] = int(new_step)
        return new

    # -- LLM-facing summary ------------------------------------------------ #

    def summary(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for layer_name, param_name, spec in self._iter_params():
            rows.append({
                "layer": layer_name,
                "parameter": param_name,
                "kind": spec.get("kind"),
                "values": spec.get("values"),
                "step": spec.get("step"),
                "mode": spec.get("mode", "both"),
                "constraint": spec.get("constraint"),
                "refine": bool(spec.get("refine")),
            })
        return rows


def build_design_space(
    search_space: Optional[Dict[str, Any]] = None,
    device_catalog: Optional[Dict[str, Any]] = None,
) -> DesignSpace:
    """Build a :class:`DesignSpace` from a JSON/dict spec.

    Accepts both the new layered format (``{"layers": {...}}``) and the legacy
    flat format (``{"arch": {...}}``).  The flat format is migrated onto the
    layered default so memory / feasibility remain available.
    """
    if search_space and search_space.get("layers"):
        return DesignSpace(spec=search_space, device_catalog=device_catalog)

    # Legacy flat spec: start from the layered default, then narrow the
    # candidate grids of the parameters it actually mentions.
    spec = copy.deepcopy(LAYERED_DESIGN_SPACE)
    flat = (search_space or {}).get("arch") or {}
    if flat:
        aliases = {
            "tiles": "tiles",
            "cores_per_tile": "cores_per_tile",
            "core_height": "core_height",
            "core_width": "core_width",
            "num_wavelength": "num_wavelength",
            "work_freq": "work_freq",
            "dataflow": "dataflow",
            "hbm_bandwidth": "hbm_bandwidth",
            "glb1_cache_size": "glb1_cache_size",
            "glb2_cache_size": "glb2_cache_size",
            "rf_cache_size": "rf_cache_size",
            "sharing_policy": "sharing_policy",
            "temporal_accum": "temporal_accum",
            "laser_distribution": "laser_distribution",
        }
        for layer in spec["layers"].values():
            for param, p_spec in (layer.get("params") or {}).items():
                key = aliases.get(param)
                if key and key in flat:
                    p_spec["values"] = list(flat[key])
        prec = flat.get("precision") or {}
        for bit_key, param in (("in_bit", "precision_in_bit"),
                               ("w_bit", "precision_w_bit"),
                               ("out_bit", "precision_out_bit")):
            if bit_key in prec:
                spec["layers"]["system"]["params"][param]["values"] = list(prec[bit_key])
    return DesignSpace(spec=spec, device_catalog=device_catalog)
