"""Cost of reducing parallel MRR weight bank partial sums after the ADCs."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


# A first-order digital cost model. The defaults are scenario assumptions,
# rather than synthesized or measured area, energy or timing for this design.
DEFAULT_MODEL = {
    "energy_pj_per_adder_bit": 0.003125,
    "area_um2_per_adder_bit": 4.0,
    "cycles_per_batch": 1,
}


def estimate_weight_bank_reduction(
    sub_arch: Mapping[str, Any],
    layer_cycles: Mapping[str, Sequence[int]],
) -> Dict[str, Any] | None:
    """Estimate the extra cost of adding the two digitized group outputs.

    Count one add per valid output, spectral tile and signed forward. All
    physical cores can add in parallel. Charge one extra, non-overlapped
    digital cycle per batch by default. This deliberately takes no credit
    for reducing the number of reads through analog temporal accumulation.
    """
    core = sub_arch["core"]
    groups = int(core.get("weight_bank_groups", 1))
    if groups == 1:
        return None
    if groups != 2 or core.get("name") != "mrr_weight_bank":
        raise ValueError("digital group reduction requires two groups in an MRR weight bank")

    output_bits = int(core["precision"]["out_bit"])
    frequency_hz = float(core["work_freq"]) * 1e9
    parallel_cores = int(sub_arch["tiles"]) * int(sub_arch["cores_per_tile"])
    if output_bits < 1 or not math.isfinite(frequency_hz) or frequency_hz <= 0 or parallel_cores < 1:
        raise ValueError("digital group reduction requires positive output bits, clock and core count")

    model = DEFAULT_MODEL.copy()
    overrides = core.get("digital_reduction_model", {})
    for key in model:
        if key in overrides:
            model[key] = float(overrides[key])
        if not math.isfinite(model[key]) or model[key] <= 0:
            raise ValueError(f"digital reduction coefficient {key} must be finite and positive")
    if model["cycles_per_batch"] != int(model["cycles_per_batch"]):
        raise ValueError("digital reduction cycles_per_batch must be an integer")

    # Sign-extend partials before adding. One guard bit preserves the exact
    # integer sum of two same-scale b-bit signed codes without overflow.
    adder_bits = output_bits + 1
    energy_per_add_pj = adder_bits * model["energy_pj_per_adder_bit"]
    area_per_adder_um2 = adder_bits * model["area_um2_per_adder_bit"]
    layers = {}
    for layer, (iter_n, iter_d, iter_m, n, _d, m, forward_w, forward_x) in layer_cycles.items():
        signed_forwards = int(forward_w) * int(forward_x)
        additions = int(n) * int(m) * int(iter_d) * signed_forwards
        batches = int(iter_n) * int(iter_d) * int(iter_m) * signed_forwards
        digital_cycles = batches * int(model["cycles_per_batch"])
        layers[layer] = {
            "additions": additions,
            "batches": batches,
            "cycles": digital_cycles,
            "energy_pj": additions * energy_per_add_pj,
            "latency_s": digital_cycles / frequency_hz,
        }

    return {
        "groups": groups,
        "wavelengths_per_group": int(core["group_wavelengths"]),
        "adc_partial_bits": output_bits,
        "accumulator_bits": adder_bits,
        "adder_count": parallel_cores,
        "frequency_hz": frequency_hz,
        "additions": sum(layer["additions"] for layer in layers.values()),
        "energy_pj": sum(layer["energy_pj"] for layer in layers.values()),
        "area_um2": area_per_adder_um2 * parallel_cores,
        "latency_s": sum(layer["latency_s"] for layer in layers.values()),
        "by_layer": layers,
        "model": model,
        "assumptions": (
            "One same-scale signed adder per physical core. Each pair of ADC outputs is "
            "sign-extended and added with one guard bit. Costs use a configurable linear "
            "per-bit model, exclude wiring, leakage and final requantization, and serialize "
            "the reduction with compute. This models reduction cost, not quantization "
            "error or model accuracy."
        ),
    }
