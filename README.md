# photonic-tuner

An LLM multi-agent, RAG-assisted design-space-exploration (DSE) framework for
**photonic** deep-learning accelerators.  It tunes architecture parameters,
optical-core geometry, and device-library selections against a SimPhony
evaluator, using a normalised objective of `latency × energy-per-MAC × area`.

## What it does

Four agents cooperate in an imperative ReAct loop (no LangGraph, so the whole
framework runs offline with a mock LLM):

```
model_level  ->  layer_level  ->  optical_level  ->  solution_generation
```

- **model_level** — finds globally important bottleneck layers from whole-model
  evidence.
- **layer_level** — attributes each candidate layer's cost to compute / memory /
  communication / buffer / device root causes.
- **optical_level** *(photonic-specific)* — diagnoses the optical link: the
  insertion-loss budget along the longest light paths and the optical power
  budget (laser output vs. path loss vs. photodetector sensitivity), then picks
  concrete library *variants*.
- **solution_generation** — combines the above with retrieved historical cases
  (RAG) to emit the next legal hardware configuration.

## Quick start

Offline smoke test (mock LLM + analytic mirror, no network, no API key):

```bash
python3 main.py --mode analytic --provider mock --iterations 4
```

Real SimPhony simulator + real LLM (one-time setup, then run).  Needs Python 3.10,
`g++`, and network access:

```bash
bash scripts/env/setup_real.sh              # venv + deps + compile CACTI
cp configs/llm.example.json configs/llm.json   # then fill in api_key
source scripts/env/activate_real.sh
python3 main.py --mode real \
    --provider paratera --model DeepSeek-V4-Pro-0813 \
    --nn-model-factory tests.factory:build_vit_tiny --workload vit_tiny
```

Run the test suite:

```bash
python3 tests/run_tests.py
```

## Repository layout

```
photon_tuner/
  agents/          # model / layer / optical / solution agents
  core/            # schemas, objective, search space
  llm/             # mock + OpenAI-compatible clients, prompts, tracing
  rag/             # hashing/sentence-transformer embeddings, SQLite history store
  runtime/         # ReAct loop, tuning loop, result coercion
  simulators/      # adapter interface, analytic mirror, SimPhony adapter
  toolbox/         # analysis / device / common toolboxes
configs/           # initial hardware, search space, workloads, llm.example.json
scripts/           # env activation, sweep/analysis scripts
tests/             # unit + pipeline tests
```

Design notes and reference papers are kept locally only (gitignored); the key
references are linked in the Acknowledgements below.

## Acknowledgements

The multi-agent, evidence-gathering architecture (a model-level analyser, a
layer-level diagnosis agent, a solution generator, a shared toolbox, and a RAG
history store) is **inspired by** the *Beacon* framework — "LLM Multi-Agent
Driven Hardware Design Space Exploration for Heterogeneous Multi-Chiplet Deep
Learning Accelerators".  This repository is an independent, photonic-domain
reimplementation: the design space, simulator adapters, objective, and toolboxes
target SimPhony photonic accelerators, and the code does not reuse Beacon's
implementation.

### Third-party vendored code

- [`photon_tuner/simulators/simphony/`](photon_tuner/simulators/simphony) is a
  vendored copy of [ScopeX-ASU/SimPhony](https://github.com/ScopeX-ASU/SimPhony)
  (MIT license; paper arXiv:2411.13715).  See its bundled
  [`LICENSE`](photon_tuner/simulators/simphony/LICENSE) and
  [`VENDORED.md`](photon_tuner/simulators/simphony/VENDORED.md).  Its CACTI
  sub-component is from Hewlett-Packard (see the headers under `cacti/`).
