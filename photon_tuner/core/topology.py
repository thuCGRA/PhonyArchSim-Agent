"""L4 topology / circuit search-space templates.

SimPhony's circuit-level knobs live in the core ``netlist`` and are
string/expression based (``scaling_rules``, ``temporal_accum_factor``,
``laser_power_distribution``).  Those must never be hand-written by the LLM, so
this module maps small, white-listed *semantic* enums to the concrete netlist
values.  The search space therefore stays a controlled template space rather
than free-form connectivity.

The three exposed enums are:

* ``sharing_policy``  -- how DAC/ADC/TIA are shared across the crossbar.  Only
  the ADC (instance ``i2``) and TIA (instance ``i8``) scaling rules change;
  every photonic instance keeps the ``dptc.yml`` default so device counts stay
  coupled to the crossbar topology.
* ``temporal_accum``   -- temporal (time-multiplexed) accumulation depth.
* ``laser_distribution`` -- optical power distribution factor (a white-listed
  expression over R/H/C/W/N).
"""

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Candidate enums (mirrored by design_space.py's ``topology`` layer).
# --------------------------------------------------------------------------- #

SHARING_POLICIES: List[str] = ["no_sharing", "adc_shared", "adc_tia_shared"]

# ADC (i2) and TIA (i8) scaling-rule overrides per sharing policy.
_SHARING_OVERRIDES: Dict[str, Dict[str, str]] = {
    "no_sharing": {"i2": "H*W*R*C", "i8": "H*W*R*C"},       # every node owns ADC+TIA
    "adc_shared": {"i2": "(H*W)*R", "i8": "H*W*R*C"},       # LT default: ADC shared per tile
    "adc_tia_shared": {"i2": "(H*W)*R", "i8": "(H*W)*R"},   # ADC+TIA shared per tile
}

TEMPORAL_ACCUM: List[str] = ["off", "default", "deep"]
_TEMPORAL_DURATIONS: Dict[str, int] = {"off": 1, "default": 3, "deep": 6}

LASER_DISTRIBUTIONS: List[str] = ["H*W", "H*W*N", "H*W*N*C"]

DEFAULT_SHARING_POLICY = "adc_shared"
DEFAULT_TEMPORAL_ACCUM = "default"
DEFAULT_LASER_DISTRIBUTION = "H*W"


def apply_sharing_policy(netlist: Dict[str, Any], policy: str) -> None:
    """Override the ADC/TIA scaling rules for the chosen sharing policy."""
    if policy not in SHARING_POLICIES:
        raise ValueError(
            f"Unknown sharing policy {policy!r}; expected one of {SHARING_POLICIES}"
        )
    rules = dict(netlist.get("scaling_rules") or {})
    rules.update(_SHARING_OVERRIDES[policy])
    netlist["scaling_rules"] = rules


def apply_temporal_accum(netlist: Dict[str, Any], temporal: str) -> None:
    """Set the temporal accumulation duration (1 == off)."""
    if temporal not in TEMPORAL_ACCUM:
        raise ValueError(
            f"Unknown temporal accumulation {temporal!r}; expected one of {TEMPORAL_ACCUM}"
        )
    cfg = dict(netlist.get("temporal_accum_factor") or {})
    cfg["duration"] = _TEMPORAL_DURATIONS[temporal]
    netlist["temporal_accum_factor"] = cfg


def apply_laser_distribution(netlist: Dict[str, Any], distribution: str) -> None:
    """Set the laser power distribution factor.

    ``laser_power_distribution`` is the key the analytic mirror reads;
    ``laser_power_distribution_factor`` is the key the real ``energy.py`` reads
    (an upstream field-name mismatch in SimPhony).  Write both so the two modes
    agree on the same optical power budget.
    """
    if distribution not in LASER_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown laser distribution {distribution!r}; expected one of {LASER_DISTRIBUTIONS}"
        )
    netlist["laser_power_distribution"] = distribution
    netlist["laser_power_distribution_factor"] = distribution


def apply_topology(
    netlist: Dict[str, Any],
    sharing_policy: str,
    temporal_accum: str,
    laser_distribution: str,
) -> None:
    """Apply all three L4 circuit-level choices to a core netlist."""
    apply_sharing_policy(netlist, sharing_policy)
    apply_temporal_accum(netlist, temporal_accum)
    apply_laser_distribution(netlist, laser_distribution)
