# Post-Sweep Fixes (apply after the 5× sweep, before the convergence experiment)

These are the fixes deferred to keep the 5× sweep internally consistent
(all 20 runs on identical code). Apply them, re-run the unit tests, then launch
the vit_base 5-iteration convergence experiment.

## 1. Tolerant `level` resolver in `modify_device_parameter`
File: `photon_tuner/toolbox/device_tools.py`, `_tool_modify_device_parameter` (~line 121).

The LLM sometimes passes a device *instance* (`level="core_mzm_0_i0"`) where the
tool requires `level ∈ {"core","node"}` (run_2: `applied=1 rejected=1`). Normalize:

```python
level = str(arguments.get("level") or "")
if level.startswith("core"):
    level = "core"
elif level.startswith("node"):
    level = "node"
```

Place before the `if level not in {"core","node"} ...` check.

## 1b. Document valid `modify_hardware_parameter` names + no-op guard
File: `photon_tuner/llm/prompts.py` (solution prompt) and
`photon_tuner/core/search_space.py` (`modify_arch_parameter`).

vit_small run_0 used `in_bit` (device-level name) where the arch parameter is
`precision_in_bit`, and also made a no-op `modify_device_parameter(LT_MZM→LT_MZM)`.
Fix: (a) list the valid arch parameter names in the solution prompt
(`work_freq, num_wavelength, core_width, core_height, tiles, cores_per_tile,
dataflow, precision_in_bit, precision_w_bit, precision_out_bit, sharing_policy,
temporal_accum, laser_distribution`); (b) optionally accept `in_bit`/`w_bit`/
`out_bit` aliases in `modify_arch_parameter`; (c) treat a device change to the
already-selected variant as a no-op (return the current choice instead of a
"changed" result).

## 2b. Solution agent: sanity-check device area/energy before a swap
File: `photon_tuner/llm/prompts.py` (solution prompt) and/or
`photon_tuner/toolbox/device_tools.py`.

vit_base run_1 swapped LT_MZM→ACADEMIA_EO_MZM_1x1 (736000 vs 5200 µm² = 141×)
×1536 copies, inflating area +2429.6% and regressing the objective +853.8%.
The LLM applied the swap without seeing the target device's area. Fix: require
`inspect_device_config(target)` (or a compact "area/energy per bit" field in
`list_device_candidates`) before any `modify_device_parameter`, and/or reject a
swap whose area×count blows past the current total by a threshold.

## 2. Optical agent: inspect a variant before recommending it
File: `photon_tuner/llm/prompts.py`, optical agent mission (~line 99).

Add to the optical prompt: "Before recommending a device variant, call
`inspect_device_config` on that variant and verify the claimed direction
(e.g. that its `insertion_loss` is actually lower/higher as claimed). Do not
recommend a variant whose config you have not inspected." This targets the
run_1 hallucination (recommended SL_MZM as "reduce attenuation" without ever
inspecting SL_MZM's 4.5 dB loss).

## 3. On-chip memory-energy over-count (÷ block_size)
File: `simphony/onnarchsim/workflow/memory.py`, energy section (~lines 554-610).

GLB2/GLB1/RF multiply CACTI's "energy per access" (pJ per `block_size`-byte
access) by the byte count → over-count ×block_size. HBM is correct. Change the
GLB2/GLB1/RF `memory_total_energy[...] +=` terms to divide by the level's
`block_size` (from the memory params / spec). The offline tool
`scripts/analysis/correct_memory_energy.py` already computes the corrected values for the
sweep; this code fix makes the simulator's `report.memory.energy` correct going
forward (still NOT in the objective scalar).
