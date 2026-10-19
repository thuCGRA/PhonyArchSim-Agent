"""Objective function for photonic design-space exploration.

The scalarisation follows the shape popularised by LLM-driven chiplet DSE
frameworks (the work this project is *architecturally* inspired by — see
``NOTICE``): a product of normalised latency, energy, and cost.  For a photonic
accelerator the cost term is chip *area* (a first-order fabrication-cost proxy)
and the energy term is *energy per MAC*, the figure photonic papers quote::

    objective = latency_s * pJ_per_mac * area_um2

For a fixed workload ``pJ_per_mac = energy_pj / total_macs``, so this ranks
identically to ``latency_s * energy_pj * area_um2`` (up to the constant
``1 / total_macs``) while normalising the objective's magnitude and making it
comparable across workloads.

Lower is better.  Two extensions make the scalarisation actually use the
trade-offs SimPhony *does* report (e.g. ``num_wavelength``: more wavelengths
cut latency/energy but grow area):

* **Configurable weights** -- ``search_space["objective_weights"]`` maps
  ``{latency, energy, area}`` (or the metric keys) to exponents, giving a
  weighted geometric mean.  ``energy`` / ``energy_pj`` alias to ``pJ_per_mac``.
* **Insertion-loss budget penalty** -- an optional one-sided soft penalty keeps
  the longest optical path under ``search_space["il_budget"]`` (dB).  It anchors
  the *optical link budget* (laser power vs photodetector sensitivity) instead
  of a digital throughput target.

Weights only *reallocate* trade-offs that already exist in the metric model;
they cannot create one where SimPhony's formulas are monotonic (e.g.
``work_freq`` is a free lunch in both the mirror and the real SimPhony, and is
bounded by device-bandwidth hard constraints rather than this scalarisation).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


DEFAULT_METRIC_KEYS = ("latency_s", "pJ_per_mac", "area_um2")
_WEIGHT_ALIASES = {
    "latency": "latency_s",
    "latency_s": "latency_s",
    "energy": "pJ_per_mac",
    "energy_pj": "pJ_per_mac",
    "pJ_per_mac": "pJ_per_mac",
    "area": "area_um2",
    "area_um2": "area_um2",
}


def objective_weights(search_space: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Return ``{metric_key: exponent}`` with defaults of 1.0 (equal weight).

    Accepts ``search_space["objective_weights"]`` keyed by either short names
    (``latency`` / ``energy`` / ``area``) or the metric keys themselves.
    """
    raw = (search_space or {}).get("objective_weights") if isinstance(search_space, Mapping) else None
    weights = {key: 1.0 for key in DEFAULT_METRIC_KEYS}
    if not isinstance(raw, Mapping):
        return weights
    for name, exponent in raw.items():
        key = _WEIGHT_ALIASES.get(str(name))
        if key is None:
            continue
        weights[key] = _positive_float(exponent, default=1.0)
    return weights


def base_objective(
    metrics: Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Weighted product of the three required positive metrics."""
    weights = weights or {key: 1.0 for key in DEFAULT_METRIC_KEYS}
    value = 1.0
    for key in DEFAULT_METRIC_KEYS:
        value *= _required_positive_metric(metrics, key) ** float(weights.get(key, 1.0))
    return value


def objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> float:
    return base_objective(metrics, objective_weights(search_space)) * il_budget_penalty(
        metrics, search_space
    )["il_budget_penalty"]


def metrics_with_objective(
    metrics: Mapping[str, Any],
    hardware: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Return the metrics dict enriched with the scalar objective."""
    weights = objective_weights(search_space)
    enriched: Dict[str, float] = {key: float(metrics[key]) for key in metrics}
    enriched["base_objective"] = base_objective(metrics, weights)
    # Expose weights as *scalar* keys (not a nested dict) so this function stays
    # idempotent: the tuner re-runs it on the already-enriched metrics, and a
    # dict value would make ``float(...)`` above raise TypeError.
    for key, weight in weights.items():
        enriched[f"weight_{key}"] = weight
    enriched.update(il_budget_penalty(metrics, search_space))
    enriched["objective"] = objective(metrics, hardware, search_space)
    return enriched


def il_budget_penalty(
    metrics: Mapping[str, Any],
    search_space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """One-sided soft penalty when the optical link exceeds its insertion-loss budget.

    Anchors the *optical link budget* -- the maximum total insertion loss (dB)
    the laser can overcome while still delivering the photodetector's
    sensitivity.  A design whose ``core_insertion_loss`` exceeds
    ``search_space["il_budget"]`` is punished on the over-budget side only
    (under-provisioning is already priced in via latency), with a bounded
    quadratic so it stays a soft constraint.
    """
    ss = search_space if isinstance(search_space, Mapping) else {}
    target = _positive_float(ss.get("il_budget"), default=None)
    actual = _positive_float(metrics.get("core_insertion_loss"), default=None)
    if target is None or actual is None:
        return {"il_budget_penalty": 1.0}
    lam = _positive_float(ss.get("il_budget_lambda"), default=1.0)
    ratio = actual / target
    penalty = 1.0 + lam * max(0.0, ratio - 1.0) ** 2
    return {
        "now_insertion_loss": actual,
        "il_budget": target,
        "il_budget_penalty": penalty,
    }


def _required_positive_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        raw = metrics[key]
    except KeyError as exc:
        raise KeyError(f"Evaluation metrics missing required key: {key}") from exc
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation metric {key} must be numeric, got {raw!r}") from exc
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"Evaluation metric {key} must be finite and positive, got {value!r}")
    return value


def _positive_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0.0 or not math.isfinite(parsed):
        return default
    return parsed
