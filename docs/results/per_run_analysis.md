# Per-Run Analysis Log

Per-run observations from the real-mode 4-workload × 5-run sweep (objective =
`latency_s × energy_pj × area_um2`, lower is better). Each run is baseline eval →
model/layer/optical/solution agents → proposal eval, against the real
`ONNArchSimulator` and real `DeepSeek-V4-Pro-0813`.

## vit_tiny (21 layers, baseline objective 1.13538e13)

### run_0 — work_freq 5→10, +40.18% (LLM-driven) ✅
- `step_hardware_parameter(work_freq, +1)`, `edits applied=1 rejected=0`.
- LLM rationale (sound): cycle-heavy attention/linear matmuls dominate; stepping
  work_freq cuts latency without increasing per-op energy/area.
- LLM **rejected SL_MZM** (correctly): "SL_MZM adds ~3.3 dB insertion loss and
  the artifacts do not expose laser power / photodetector sensitivity, so a link
  edit is not yet justified." Energy hotspots noted: core_mzm 43.4e6 pJ, dac
  48.5e6 pJ, photodetector 42.5e6 pJ, mrr 22.3e6 pJ.

### run_1 — work_freq 5→10 + LT_MZM→SL_MZM, +51.46% (LLM-driven) ✅
- `step_hardware_parameter(work_freq,+1)` + `modify_device_parameter(mzm→SL_MZM)`,
  `edits applied=2 rejected=0`.
- Combined two levers (frequency + device) → +51.46% vs run_0's +40.18% (work_freq only).
- **Factual-error chain**: optical agent claimed SL_MZM "reduces optical
  attenuation" (wrong — SL_MZM 4.5 dB > LT_MZM 1.2 dB); solution agent echoed it.
  Simulator still net-beneficial: MZM energy −8.9× (5.43e7→6.03e6 pJ) dominates
  laser +28.5% (1.79e7→2.30e7 pJ). See REAL_MODE.md §8e.

### run_2 — work_freq 5→10 only, +40.18% (device edit rejected) ⚠️
- `step_hardware_parameter(work_freq,+1)` ok; `modify_device_parameter` REJECTED:
  LLM passed `level="core_mzm_0_i0"` (a device *instance*) where the tool requires
  `level in {"core","node"}`. `edits applied=1 rejected=1`.
- Same intent as run_1 (work_freq + SL_MZM), but the SL_MZM swap failed → +40.18%
  instead of run_1's +51.46%. Demonstrates an intermittent tool-argument error
  (run_1 passed `level="core"` correctly) that degrades ~half the device-edit
  attempts. Fix (tolerant level resolver) deferred to the post-sweep convergence
  experiment to keep the sweep internally consistent.

### run_3 — work_freq 5→10 only, +40.18% (correctly retained LT_MZM) ✅
- `step_hardware_parameter(work_freq,+1)`, `applied=1 rejected=0`.
- Optical agent **inspected both** LT_MZM and SL_MZM this time and correctly
  noted "SL_MZM ... 4.5 dB insertion loss and 6 dB extinction ratio would
  materially erode the loss budget" → recommended *retaining* LT_MZM. Contrast
  run_1's optical agent (never inspected SL_MZM → hallucinated "reduce
  attenuation"). Confirms the "inspect-before-recommend" fix (§8e) is the right one.
- Nuance: the loss-focused optical agent missed SL_MZM's energy win (450→50
  fJ/bit dominates the +28.5% laser penalty), so run_3 is conservative (+40.18%)
  while run_1's *accidentally* beneficial swap reached +51.46%. Multi-agent
  local-vs-global objective misalignment worth reporting.

## vit_small (31 layers, baseline objective 2.90683e14)

### run_0 — +44.96% via rule-based fallback (LLM edits both failed) ⚠️
- `modify_hardware_parameter(updates=[{"parameter":"in_bit","value":2}])` → FAIL
  (`Unknown architecture parameter: in_bit`; the arch name is `precision_in_bit`).
- `modify_device_parameter(level="core", mzm, LT_MZM)` → "ok" but a **no-op**
  (LT_MZM → LT_MZM).
- `edits applied=1 rejected=1`, yet hardware diff is only `work_freq 5→10` — the
  `_repair_duplicate` fallback stepped work_freq after both LLM edits failed.
  So this run's +44.96% is **rule-based, not LLM-driven** (unlike vit_tiny, whose
  `step_hardware_parameter(work_freq,+1)` was the LLM's own successful edit).
- Two new LLM edit defects: (a) parameter-name mismatch `in_bit` vs
  `precision_in_bit` (device- vs arch-level naming); (b) no-op device change.

### run_1 — dataflow output_stationary→weight_stationary, +8.61% (LLM-driven) ✅
- `modify_hardware_parameter(updates=[{"parameter":"dataflow","value":"weight_stationary"}])`,
  `edits applied=1 rejected=0` — **first successful `modify_hardware_parameter`
  (multi-param tool) since the round-11 prompt fix**, confirming the fix.
- Dataflow is a much weaker lever than work_freq (+8.61% vs +44.96% fallback on
  run_0), so this run's LLM chose a suboptimal lever for vit_small. Shows the
  search explores diverse levers (work_freq / device / dataflow) with varying gain.
- Also exposed an `analyze_sweep.py` bug: `hw_delta` only diffed `arch.core` +
  `devices.core`, missing `arch.dataflow`; fixed to a full recursive diff.

### run_2 — work_freq 5→10, +44.96% (LLM-driven) ✅
- `step_hardware_parameter(work_freq,+1)`, `applied=1 rejected=0`.
- vit_small edit mix so far: 1 fallback (run_0), 1 dataflow (run_1), 1 work_freq
  (run_2) — LLM edit success 2/3, more divergent than vit_tiny's 5/5 work_freq.

### run_3 — LT_MZM→SL_MZM, +7.46% (LLM-driven) ✅
- `modify_device_parameter(level="core", mzm, SL_MZM)`, `applied=1 rejected=0`.
- Device swap is a weak lever for vit_small (+7.46% vs work_freq's +44.96%).
- Lever-selection observation: vit_small's LLM tried dataflow (run_1), work_freq
  (run_2), SL_MZM (run_3) — only 1/4 self-selected the strong work_freq lever.

### run_4 — 4-edit compound proposal, +46.40% (LLM-driven) ✅
- `step_hardware_parameter(core_height,+1)` → height 12→16; `(core_width,+1)`
  → width 12→16; `(work_freq,+1)` → 5→10; `modify_device_parameter(dac, DAC_1)`.
  `edits applied=4 rejected=0` — first multi-edit run, richest search action so far.
- Net +46.40% (vs work_freq-only +44.96%): the extra height/width/DAC edits add
  ~+1.4pp. Demonstrates the search can compose several levers in one proposal.

vit_small (5/5) summary: fallback 1, LLM 4 (LLM success 80%). LLM levers tried:
dataflow, work_freq, SL_MZM, and a 4-edit compound — only work_freq is strong.

## vit_base (61 layers, baseline objective 1.53546e16)

### run_0 — 5-edit proposal WORSE by -67.34% (rejected; best = baseline) ⚠️
- LLM proposed: height 12→16, width 12→16, coupler→ACADEMIA_DC_2x2,
  mzm→SL_MZM, PD→ACADEMIA_Analog_PD. `edits applied=5 rejected=0`.
- Metrics: energy -45.1%, latency -27.9% (both improve) but **area +322.7%**
  (6.65e7→2.81e8 µm²) dominates → objective +67.3% (worse).
- The height/width lever's area cost is **workload-dependent**: +51.4% for
  vit_small but +322.7% for vit_base (the 61-layer model's largest layer needs a
  much bigger chip when tiles grow 12→16). This is the first *regressive*
  proposal (LLM can hurt, not only help).
- `trajectory.json` shows `is_best=False`, best stays baseline — the DSE's
  keep-best mechanism correctly rejects bad proposals, so the *delivered* result
  is never worse than baseline. Distinguish **proposal quality** (can regress)
  from **DSE best** (always ≥ baseline).

### run_1 — 7-edit proposal, -853.76% (worse; rejected) ⚠️
- `edits applied=7 rejected=0`: height/width 12→16, dataflow→weight_stationary,
  mzm→ACADEMIA_EO_MZM_1x1, laser→LT_Laser, dac→DAC_1, PD→ACADEMIA_Analog_PD.
- Area +2429.6% (6.65e7→1.68e9 µm²) dominates → objective +853.8%. Root cause:
  the `ACADEMIA_EO_MZM_1x1` device is **141× larger** than LT_MZM (736000 vs
  5200 µm², 460×1600 µm) × 1536 copies = 1.13e9 µm², plus height/width 12→16
  raised device counts (1152→1536/2048).
- Key finding: the device library has extreme area spread, and the LLM swapped
  to a giant MZM without inspecting its area. This is exactly the failure the
  planned "inspect-before-recommend" optical-prompt fix targets, and motivates a
  further fix: the solution agent should sanity-check a device's area/energy
  before applying a swap.

### run_2 — work_freq 5→10 + SL_MZM + coupler, +57.43% (LLM-driven) ✅
- `edits applied=3 rejected=0`: work_freq 5→10, mzm→SL_MZM, coupler→Customized_DC_2x2.
- Metrics: energy -39.4%, latency -32.8%, area +4.6% → objective -57.4%.
- work_freq's benefit **scales with model size**: +40.18% (vit_tiny) < +44.96%
  (vit_small) < +57.43% (vit_base). But the LLM self-selected work_freq only
  1/3 for vit_base (vs 5/5 vit_tiny) — lever selection degrades with size.

- Diagnosis-to-action gap: all 3 vit_base runs diagnose "energy/switching
  dominates", yet run_0/1 map that to *geometry* (bigger tiles, area-catastrophic)
  while only run_2 maps it to *work_freq* (the correct, low-area lever). The LLM
  diagnoses correctly but its diagnosis→lever mapping is unreliable at scale.

### run_3 — SL_MZM only, +7.57% (LLM-driven) ✅
- `modify_device_parameter(SL_MZM)`, applied=1 rejected=0. Weak lever again
  (+7.57%, consistent with vit_small run_3's +7.46%).

### run_4 — work_freq 5→10 + SL_MZM + LT_Laser, +61.03% (LLM-driven) ✅
- `edits applied=3 rejected=0`. Best vit_base run.

vit_base (5/5) summary: LLM edit success 100%, but **2/5 proposals regressed**
(geometry → area catastrophe) and were rejected by keep-best. Delivered
best_mean +25.21% (std 27.94%) vs proposal_mean -159.01% — the keep-best
mechanism is what turns a very negative proposal distribution into a positive
delivered result.

## mobilevit (baseline objective 1.01491e15, energy 2.300e9, latency 6.63e-3)

### run_0 — SL_MZM only, +7.58% (LLM-driven) ✅
- `modify_device_parameter(SL_MZM)`, applied=1 rejected=0.
- SL_MZM swap gain is workload-independent and weak: +7.46% (vit_small),
  +7.57% (vit_base), +7.58% (mobilevit) — consistent ~+7.5%.

### run_1 — dataflow output→weight_stationary, +2.97% (LLM-driven) ✅
- `modify_hardware_parameter(updates=[{"parameter":"dataflow","value":"weight_stationary"}])`
  applied=1 rejected=0 (2nd successful use of the multi-param tool).
- dataflow is the weakest lever and workload-dependent: +8.61% (vit_small) vs
  +2.97% (mobilevit).
