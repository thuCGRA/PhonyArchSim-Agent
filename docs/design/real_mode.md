# Real-Mode DSE Execution — Methodology & Findings

This file documents how the photonic DSE framework executes against the **real**
SimPhony simulator (`onnarchsim.simulator.ONNArchSimulator`) and the **real** LLM,
and records the correctness fixes made along the way. It is the working notes for
the "Methods / Implementation" section of the paper.

## 1. Environment

- Standalone CPython **3.10.22** (via `uv`) in `/home/fyq/epic/.venv-simulator`.
- Key pins: `torch==1.12.1+cpu`, `torchvision==0.13.1+cpu`, `torchonn==0.0.8`,
  `torchonn-pyutils==0.0.3.1`, `mmcv-lite==2.0.0`, `mmengine==0.7.4`, `numpy==1.23.5`.
- Activate: `source scripts/env/activate_real.sh` (sets `PYTHONPATH=simphony:<root>`).
- The CACTI memory model (`simphony/cacti/cacti`) is compiled and runs on CPU.
- Full dependency manifest: `scripts/env/requirements-real.txt`.

## 2. Pipeline (real mode)

```
CLI (main.py) -> SimphonyAdapter.evaluate(hardware, iter)
    -> ONNArchSimulator(...)                    # loads device DB + arch config,
                                                # NN->ONN conversion, layer workload
    -> _inject_hardware(sim, hardware)          # SEE BELOW
    -> run_full_report()                        # energy / area / latency / memory
    -> metrics_with_objective()                 # objective = energy*area*latency (weights=1)
```

The multi-agent loop (model → layer → optical → solution) is driven by the real
LLM (`DeepSeek-V4-Pro-0813` via paratera) with a ReAct tool loop, producing a new
`hardware` dict each iteration; the adapter then re-simulates that hardware.

## 3. Hardware injection (the critical fix)

`ONNArchSimulator` reads its architecture from fixed YAML files
(`LT_hetero.yml -> LT.yml -> dptc.yml/dode.yml`) and **ignores** the searched
`hardware` dict. Without an explicit injection step, every evaluation would
simulate the same fixed LT baseline, so the search would be a no-op.

`SimphonyAdapter._run_real_report` therefore rebuilds the sub-architecture from
the searched `hardware` dict via `build_sub_arch()` (the same code path the
offline analytic mirror uses) and replaces the simulator's resolved
sub-architecture **before** `run_full_report()`. Mapping:

| searched `hardware` | SimPhony sub-arch |
|---|---|
| `arch.tiles` / `cores_per_tile` / `dataflow` | `tiles` / `cores_per_tile` / `dataflow` |
| `arch.core.{width,height,num_wavelength,work_freq}` | `core.{width,height,num_wavelength,work_freq}` |
| `arch.core.precision.{in,w,out}_bit` | `core.precision.{in,w,out}_bit` |
| `arch.memory.*` | `memory.*` |
| `devices.core.{type}` | `core.devices.{type} = [name]` |
| `devices.node.{type}` | `core.node.devices.{type} = [name]` |

Verification (work_freq 5 → 10 GHz, all else fixed):

| metric | work_freq=5 | work_freq=10 |
|---|---|---|
| latency_s | 0.0155 | 0.0077 |
| energy_pj | 9.81e9 | 7.51e9 |
| area_um2 | 6.65e7 | 6.65e7 |
| objective | 1.01e16 | 3.86e15 |

## 4. Upstream field-name bug (fixed by the injection)

SimPhony's `configs/design/cores/dptc.yml` declares
`laser_power_distribution: 'H*W'`, but `onnarchsim/workflow/energy.py` reads the
key **`laser_power_distribution_factor`** (defaulting to `1` when absent). The
upstream simulator therefore silently ignores its own laser-power distribution.

`apply_laser_distribution()` writes **both** keys so the analytic mirror and the
real simulator agree on the optical power budget; the injected baseline energy is
`9.81e9 pJ` (distribution applied) versus `9.02e9 pJ` for the un-patched native
path (distribution dropped).

## 5. Workloads

Real-mode workloads are the model-zoo networks, all built on CPU with 4-bit
`QConv2d`/`QLinear`/`QMatMul` (mapped by `simple_transformer.yml` +
`simple_lt_transformer.yml`):

| workload | model | quantized layers |
|---|---|---|
| vit_tiny | `QBertViTBase`(dim 192, depth 4, heads 3) | 21 |
| vit_small | `QBertViTBase`(dim 384, depth 6, heads 6) | 31 |
| vit_base | `QBertViTBase`(dim 768, depth 12, heads 12) | 61 |
| mobilevit | `QMobileViT`(224×224) | 78 |

## 6. Sweep protocol

- `scripts/experiments/sweep.py` runs each workload `--runs` times, one full DSE invocation per
  run (fresh `runs/sweep/<workload>/run_<i>/` output root), sequentially.
- Each run records `exit_code`, elapsed time, best objective/energy/area/latency,
  and `distinct_hardware` (whether the proposal actually changed the design) into
  `runs/sweep/sweep_summary.jsonl`.
- The LLM request/response trace (`raw_json/`, `prompt_calls.json`) is kept per
  run for offline audit.

## 7. Search-space completeness (every knob is wired)

Each searched knob was varied one-at-a-time from the vit_tiny baseline and the
real simulator's response recorded (computation energy / area / computation
latency):

| knob | energy | area | latency | interpretation |
|---|---|---|---|---|
| tiles 4→8 | −6% | +88% | −50% | tiling parallelism |
| cores_per_tile 2→4 | −2% | +88% | −48% | tiling parallelism |
| width 12→24 | −18% | +58% | −48% | crossbar M dimension |
| height 12→24 | −7% | +79% | −50% | crossbar N dimension |
| num_wavelength 12→24 | −19% | +35% | −48% | D-dimension tiling |
| work_freq 5→10 GHz | −24% | 0 | −50% | static energy ÷ f |
| precision.in_bit 4→8 | +300% | 0 | 0 | DAC / laser power ∝ 2^in_bit |
| precision.out_bit 4→8 | +5.5% | 0 | 0 | ADC energy |
| precision.w_bit 4→8 | 0 | 0 | 0 | *computation* unchanged |
| dataflow out→weight | −9% | 0 | −0.1% | cycle mapping |

`precision.w_bit` (weight bit-width) does **not** change computation energy/area
/latency — it only scales the **weight tensor size in the memory subsystem**
(`memory.py` uses `weight_precision` for `w_bits`), so it affects memory
latency/energy, not the optical MVM.  This is the correct physical separation:
`in_bit` drives the DAC/laser, `out_bit` the ADC, and `w_bit` the stored-weight
footprint.

Note on the objective scope: the DSE scalar `energy_pj` and `area_um2` are the
**computation** energy/area (from `simu_energy`/`simu_area`); memory energy is
computed separately in the report (`report.memory.energy`) but is not part of the
objective scalar, while `latency_s` *does* include memory latency.  This is the
existing report-layer design and should be stated as an explicit scope choice in
the paper.

## 8b. Critical fix: memory-latency unit bug (`memory.py:399`)

Found during the vit_tiny sweep. `onnarchsim/workflow/memory.py` computed the
per-cache bandwidth with a **double 1e9 division**:

```python
real_cycle_time   = extract_cycle_time(cacti_output)   # ns
real_cycle_time_s = real_cycle_time / 1e9              # ns -> s   (correct)
bandwidth_per_module = bus_width / real_cycle_time_s   # bits/s    (correct)
real_bandwidth = bus_width / (real_cycle_time_s / 1e9) / 8   # <-- BUG: /1e9 twice
```

`real_cycle_time_s` is already in seconds, so dividing by 1e9 again inflated
`real_bandwidth` by 10⁹ and made every memory latency (`GLB2/GLB1/RF_*_latency`)
10⁹× too small: vit_tiny memory latency was `4.45e-13 s` (≈ 0) instead of
`4.45e-4 s`. Because `latency_s = computation + memory`, the objective's latency
term silently ignored memory.

Fix (applied): `real_bandwidth = bus_width / real_cycle_time_s / 8`.

After the fix, memory latency is the **dominant** term (vit_tiny mem/comp ratio
1.30), `latency_s` roughly doubles, and — importantly — work_freq scaling is
diluted (memory latency is CACTI-based, independent of core frequency), which is
a more realistic memory-bound behavior. The first sweep (`runs/sweep`) was
invalidated and re-run from scratch as `runs/sweep` (buggy copy kept at
`runs/sweep_buggy_latency`).

**Secondary, deferred**: the on-chip memory *energy* (GLB2/GLB1/RF) uses CACTI
"dynamic read/write energy **per access**" (pJ per 8-byte block) multiplied by
the byte count, so it over-counts by the block size (8×). HBM energy is correct
(pJ/Byte). This only affects `report.memory.energy` (NOT the objective scalar,
which is computation energy) — vit_tiny: HBM 5.41e8, GLB2 1.29e8, GLB1 2.34e8,
RF 2.27e8 pJ. To be corrected and re-reported offline after the sweep (the layer
sizes and CACTI spec are already persisted, so no re-run is needed).

Note: the analysis tools (`inspect_layer_details`) *do* expose `memory_energy_pj`
and per-layer `memory_latency_s` to the LLM agents, even though the objective
scalar excludes memory energy. So a mid-sweep memory-energy fix would change the
agents' information inconsistently across runs; the fix is therefore deferred to
keep all 20 sweep runs identical in agent inputs. The scope mismatch (agents see
memory energy that the objective ignores) is itself a paper caveat.

**Offline/analytic mirror consistency**: the offline `analytic.py::memory_cost`
models memory latency with the HBM bandwidth only (one-time load), so it is
~52× optimistic vs the real simulator's full GLB2→GLB1→RF iteration-aware model
(vit_tiny: analytic ≈ 8.5e-6 s vs real 4.45e-4 s). Real mode is authoritative;
the offline mode under-counts memory latency and should be reported as a fast
approximation, not a substitute.

## 8c. Prompt↔tool schema mismatch (LLM architecture edits silently falling back)

Found in the first fixed run. The solution-generation prompt's JSON template
showed `step_hardware_parameter` and `modify_device_parameter` examples but
**no `modify_hardware_parameter` example**, so the LLM guessed
`{"parameter": "dataflow", "value": "weight_stationary"}` — but that tool's
schema is `{"updates": [{"parameter", "value"}, ...]}`. The edit was rejected
(`requires an 'updates' list`), leaving the hardware unchanged.

The tuner's `_repair_duplicate` then detected the no-op as a duplicate of the
baseline and stepped `work_freq` 5→10 (a *rule-based* escape), producing the
+40.18% improvement. I.e. **the improvement was the fallback's work, not the
LLM's** — the LLM's architecture edit was silently dropped. Without this fix the
LLM could only reliably do device substitutions (`link_edits`) and single-param
`step_hardware_parameter`, inflating the rule-based fallback's apparent share.

Fix: the prompt now documents both action kinds (modify = `updates` list of
`{parameter, value}`; step = `parameter`/`direction`) and the JSON template
shows a `modify_hardware_parameter` example. The partial sweep was invalidated
(`runs/sweep_buggy_prompt`) and re-run.

## 8d. Case-retrieval (RAG) is not exercised by the 1-iteration sweep

`tuner.py` commits a case to `HistoryVectorStore` only **after** each proposal
eval (`_commit_case` at line 282, inside the iteration loop) — never after the
baseline eval. The solution agent retrieves cases *before* proposing, so in a
1-iteration run the store is empty and `retrieved_cases == []` (verified:
run_0/run_1 history.db has 1 row only *after* completion). The RAG therefore
kicks in from the **second** proposal onward (iteration ≥ 2).

Consequence for honesty: the 5×1 sweep measures *LLM DSE without case
retrieval* (5 independent 1-step searches per workload). The case-retrieval
feature is untested by the sweep and must be exercised by a multi-iteration
convergence run (planned: vit_base, 5 iterations).

## 8e. Optical-agent factual error propagated to the solution agent (run_1)

The optical agent recommended SL_MZM with a **backwards reason**: "selecting
SL_MZM ... to reduce optical attenuation", but the device library says SL_MZM
`insertion_loss: 4.5 dB` vs LT_MZM `1.2 dB` (SL_MZM is *worse*, not better).
The solution agent then echoed it verbatim ("Reduce the dominant core insertion
loss with the ... SL_MZM swap") and applied the edit. So a hallucination in one
agent propagated to the next — a real multi-agent reliability risk.

The search result is still valid because the simulator evaluates ground truth:
`cal_laser_power` maps `insertion_loss → laser power → laser energy`, so the
penalty is captured. Measured on vit_tiny run_1 (iter_000 LT_MZM → iter_001
SL_MZM): MZM energy 5.43e7 → 6.03e6 pJ (**−8.9×**, 450→50 fJ/bit) while laser
energy 1.79e7 → 2.30e7 pJ (**+28.5%**, from +3.3 dB loss and ER 10→6 dB). Net
objective gain +51.46% (with work_freq) is therefore real, but the LLM's *stated*
reason for the device swap was factually wrong. Contrast run_0's solution agent,
which correctly noted "SL_MZM adds ~3.3 dB insertion loss" and rejected it —
the same LLM is inconsistent across runs.

Root cause (from the run_1 tool trace): the optical agent called
`list_device_candidates` (names only) and `inspect_device_config(mzm→LT_MZM)`
but **never inspected SL_MZM's config**, then recommended SL_MZM assuming it
was lower-loss. A robustness fix is to require `inspect_device_config` on any
recommended variant before recommending it, or to fact-check the recommendation's
stated direction against the library. This is a *reasoning-quality* fix (the
search results are already valid), so it is recorded for the paper rather than
triggering a third sweep restart.

## 8f. Intermittent `modify_device_parameter` level-argument error (run_2)

The tool requires `level in {"core","node"}`, but the LLM sometimes passes a
device *instance* name (`level="core_mzm_0_i0"`). run_2's SL_MZM swap was
rejected (`applied=1 rejected=1`), yielding +40.18% (work_freq only) vs run_1's
+51.46% (same intent, correct `level="core"`). Impact is moderate — `work_freq`
edits always succeed — but it intermittently drops device-swap gains. Fix
(tolerant level resolver + prompt clarification) is deferred to the post-sweep
convergence experiment to keep this sweep internally consistent; it does not
invalidate the sweep's results (rejected edits degrade gracefully).

## 8g. LLM-edit failure is recurrent; the fallback can mask it (vit_small run_0)

The round-11 prompt fix corrected the `updates` *format*, but the LLM still
makes parameter-*name* errors. vit_small run_0 proposed
`modify_hardware_parameter(updates=[{"parameter":"in_bit","value":2}])`
(device-level name; the arch name is `precision_in_bit`) and a no-op
`modify_device_parameter(LT_MZM→LT_MZM)`. Both failed → `_repair_duplicate`
stepped `work_freq` 5→10, so the +44.96% is **rule-based, not LLM-driven**.
This makes it essential, for the paper, to distinguish each run's improvement
as *LLM-driven* (vit_tiny 5/5: `step_hardware_parameter(work_freq,+1)` applied)
vs *fallback-driven* (vit_small run_0). The deferred fix (POST_SWEEP_FIXES §1b)
documents the valid parameter names and no-op guards.

## 8h. Proposals can regress; the DSE's keep-best protects the result (vit_base run_0)

The LLM can propose *worse* hardware. vit_base run_0 made a 5-edit proposal
(height/width 12→16, coupler/MZM/PD swaps) that cut energy -45.1% and latency
-27.9% but inflated area +322.7% (6.65e7→2.81e8 µm²), so the objective
*regressed* +67.3%. `trajectory.json` marks it `is_best=False` and keeps the
baseline — the DSE's keep-best step means the **delivered** result is never
worse than the baseline. For reporting, distinguish **proposal quality** (can be
negative) from **DSE best** (always ≥ baseline); `scripts/analysis/analyze_sweep.py` now
prints both (`prop%` and `best%`) plus a regression count.

Note: the height/width lever's area cost is **workload-dependent** — +51.4% area
for vit_small but +322.7% for vit_base, because the 61-layer model's largest
linear layer needs a much larger chip when the core tile grows 12→16. The
earlier "area is workload-independent" observation holds only for the *fixed*
12×12 / 4-tile baseline; it stops holding once the LLM edits the geometry.

## 8. Device-library / energy-model caveats (found during per-run analysis)

The multi-agent search often proposes *device substitutions*, which are real
differences in the SimPhony device library and therefore physically meaningful:

| substitution | effect | verifiable device property |
|---|---|---|
| `LT_MZM` → `SL_MZM` | energy −33% on the MZM | efficiency **50** vs **450** fJ/bit (9×), length 250 vs 260 µm; higher insertion loss 4.5 vs 1.2 dB (recovered via laser power) |
| `DAC_2` → `DAC_1` | DAC energy −33% | see caveat below |

Caveat on the DAC model: `energy.py` has **two scaling paths** for DAC power —
a figure-of-merit path (`2·FoM·in_bit·f`, linear in bit-width) when `FoM` is
present, and a fallback (`P ∝ 2^N/(N+1)`, exponential) when `FoM` is `null`.
`DAC_1` (`FoM=33.6`, `power=169 mW`) is ~11× inconsistent with its own Walden
figure of merit (169 mW / (2^12·14 GS/s) ≈ 2.95 fJ/c-step).  Because the model
scales power with the *used* bit-width, a high-precision DAC can appear
*lower*-energy at 4-bit operation than a lower-power but lower-precision DAC.
This is an upstream SimPhony data/model detail and must be reported as a
limitation when a DAC substitution appears in the search trajectory; the MZM
substitutions above are not affected.
