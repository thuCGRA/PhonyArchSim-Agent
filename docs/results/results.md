# Real-Mode DSE Results

Working results of the real-mode design-space-exploration sweep. Methodology and
correctness fixes are documented in [`REAL\_MODE.md`](REAL_MODE.md); per-run
analysis in [`PER\_RUN\_ANALYSIS.md`](PER_RUN_ANALYSIS.md). Numbers update as
`runs/sweep/` fills in.

> Status: \*\*vit\_tiny 5/5 · vit\_small 0/5 · vit\_base 0/5 · mobilevit 0/5\*\*

## 1\. Determinism

The real simulator is fully reproducible: across every run of the same workload,
the baseline `objective` is bit-identical.

|workload|baseline runs|spread|
|-|-|-|
|vit\_tiny|5/5|**0.00%** (1.135381496e13)|
|vit\_small|—|—|
|vit\_base|—|—|
|mobilevit|—|—|

## 2\. Baseline metrics (work\_freq=5 GHz, 12×12, 4 tiles, 4-bit, output-stationary)

*With the memory-latency unit fix (REAL\_MODE.md §8b).*

|workload|layers|energy\_pj|area\_um2|latency\_s|objective|
|-|-|-|-|-|-|
|vit\_tiny|21|2.166e8|6.65e7|7.88e-4|1.13538e13|
|vit\_small|31|1.247e9|6.65e7|(pending)|2.90683e14|
|vit\_base|61|9.813e9|6.65e7|(pending)|(pending)|
|mobilevit|78|2.300e9|6.65e7|(pending)|(pending)|

`area\_um2` is workload-independent (a hardware property), and energy scales
\~45× from tiny→base (matching the \~46.5× FLOPs ratio), confirming the metric
model is physically self-consistent. Latency now includes memory latency at the
correct scale (memory latency ≈ 1.30× computation for vit\_tiny).

## 3\. Per-run search results (objective lower is better)

### vit\_tiny

|run|proposal|objective (baseline→proposal)|Δ|
|-|-|-|-|
|0|`work\_freq: 5→10`|1.135e13 → 6.792e12|+40.18%|
|1|`work\_freq: 5→10` + `mzm: LT\_MZM→SL\_MZM`|1.135e13 → 5.511e12|**+51.46%**|
|2|`work\_freq: 5→10` (SL\_MZM rejected: level err)|1.135e13 → 6.792e12|+40.18%|
|3|`work\_freq: 5→10` (correctly retained LT\_MZM)|1.135e13 → 6.792e12|+40.18%|
|4|`work\_freq: 5→10`|1.135e13 → 6.792e12|+40.18%|

* **Stable gain**: `work\_freq` 5→10 GHz is proposed every run (+40.18%).
* **Higher-variance gain**: the `SL\_MZM` device swap adds \~+11% (run\_1 → +51.46%)
but is reached inconsistently — hallucinated into it (run\_1), rejected on a
`level`-argument error (run\_2), or conservatively declined on a loss-focused
analysis (run\_3). See PER\_RUN\_ANALYSIS.md.
* **mean±std (5/5)**: +42.44% ± 4.51%; best-of-N = 5.511e12.
* **Driven-by**: all 5 vit\_tiny runs are **LLM-driven** (`step\_hardware\_parameter (work\_freq,+1)` applied by the LLM). `scripts/analysis/analyze\_sweep.py` now auto-labels
each run `LLM` vs `fallback` (the `\_repair\_duplicate` work\_freq escape).

### vit\_small

|run|proposal|objective (baseline→proposal)|Δ|driven|
|-|-|-|-|-|
|0|(LLM edits failed → fallback `work\_freq 5→10`)|2.907e14 → 1.600e14|+44.96%|**fallback**|
|1|`dataflow: output→weight\_stationary`|2.907e14 → 2.657e14|+8.61%|LLM|
|2|`work\_freq 5→10`|2.907e14 → 1.600e14|+44.96%|LLM|
|3|`mzm: LT\_MZM→SL\_MZM`|2.907e14 → 2.690e14|+7.46%|LLM|
|4|4-edit: height/width 12→16 + work\_freq + DAC|2.907e14 → 1.558e14|+46.40%|LLM|

* run\_0's LLM proposed `in\_bit→2` (wrong name; arch name is `precision\_in\_bit`)
and a no-op `LT\_MZM→LT\_MZM`; both failed, so `\_repair\_duplicate` stepped
work\_freq. The +44.96% is rule-based, not LLM-driven (REAL\_MODE.md §8g).
* run\_1 is the first successful `modify\_hardware\_parameter(updates=...)` since
the prompt fix — the LLM changed dataflow (+8.61%, a weaker lever than
work\_freq).
* run\_4 is the first 4-edit compound (height/width/work\_freq/DAC, 0 rejected),
+46.40% — the richest search action observed.

### vit\_base (61 layers)

|run|proposal|objective (baseline→proposal)|Δ proposal|Δ best|driven|
|-|-|-|-|-|-|
|0|5-edit: height/width 12→16 + coupler/SL\_MZM/PD|1.535e16 → 2.569e16|**-67.34% (回归)**|0% (拒绝)|LLM|
|1|7-edit: height/width + dataflow + MZM→ACADEMIA\_EO\_MZM\_1x1|1.535e16 → 1.464e17|**-853.76% (回归)**|0% (拒绝)|LLM|
|2|`work\_freq 5→10` + SL\_MZM + coupler|1.535e16 → 6.537e15|+57.43%|+57.43%|LLM|
|3|`mzm: LT\_MZM→SL\_MZM`|1.535e16 → 1.419e16|+7.57%|+7.57%|LLM|
|4|`work\_freq 5→10` + SL\_MZM + LT\_Laser|1.535e16 → 5.983e15|+61.03%|+61.03%|LLM|

* Baseline: objective 1.53546e16 (energy 9.813e9 pJ, latency 0.0235 s, area
6.650e7 µm²).
* run\_0's proposal is the first *regression*: energy -45.1%, latency -27.9% but
**area +322.7%** (height/width 12→16 inflates the 61-layer model's chip size)
→ objective +67.3%. The DSE's keep-best correctly rejected it (`is\_best=False`),
so the delivered best stays the baseline. See REAL\_MODE.md §8h.
* run\_1 regressed -853.76%: the `ACADEMIA\_EO\_MZM\_1x1` MZM is 141× larger than
LT\_MZM (736000 vs 5200 µm²) × 1536 copies → area +2429.6%. The LLM swapped it
without inspecting its area — motivates the inspect-before-recommend fix.

## mobilevit

|run|proposal|objective (baseline→proposal)|Δ|driven|
|-|-|-|-|-|
|0|`mzm: LT\_MZM→SL\_MZM`|1.015e15 → 9.380e14|+7.58%|LLM|
|1|`dataflow: output→weight\_stationary`|1.015e15 → 9.847e14|+2.97%|LLM|
|2–4|(running)|—|—|—|

* Baseline: objective 1.01491e15 (energy 2.300e9 pJ, latency 6.63e-3 s, area
6.650e7 µm²).
* SL\_MZM swap gain is workload-independent (\~+7.5%: vit\_small +7.46%, vit\_base
+7.57%, mobilevit +7.58%).

## 4\. Caveats / findings reported

Correctness fixes applied (invalidating earlier sweeps):

1. **Memory-latency unit bug** — `memory.py` double-divided by 1e9; fixed
(latency 3.43e-4 → 7.88e-4 s).
2. **LLM architecture-edit prompt mismatch** — `modify\_hardware\_parameter`
needs `updates`, prompt now documents it (was silently falling back to a
rule-based work\_freq step).

Deferred / documented (not in the objective scalar, or reasoning-only):
3. **On-chip memory-energy over-count** (×`block\_size`) — `report.memory.energy`
only; offline correction tool ready (`scripts/analysis/correct\_memory\_energy.py`).
4. **Analytic mirror memory latency 52× optimistic** (HBM-only) — offline only.
5. **RAG not exercised** by 1-iteration runs (store empty at first proposal).
6. **Optical-agent hallucination** (run\_1: recommended SL\_MZM as "reduce
attenuation" without inspecting its 4.5 dB) — but the swap was still
net-beneficial (energy dominates); simulator ground-truth keeps results valid.
7. **`modify\_device\_parameter` `level`-argument error** (run\_2) — intermittent;
tolerant-resolver fix deferred to the convergence experiment.
8. **DAC energy model** mixes FoM (linear) and fallback (exponential) paths;
`DAC\_1` FoM=33.6 vs power=169 mW is \~11× inconsistent.

See REAL\_MODE.md §8 for details and POST\_SWEEP\_FIXES.md for the deferred fixes.

