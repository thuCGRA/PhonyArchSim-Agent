#!/usr/bin/env bash
# ============================================================================
# bootstrap.sh — recreate this project's full environment from a fresh clone.
#
# The repository's .gitignore keeps the following OUT of git; this script
# recreates every one of them so a newcomer can reproduce the working setup:
#
#   .tools/                uv + uv-managed CPython toolchains (download caches inside are disposable)
#   .venv-agent/           agent / runtime venv  (Python >=3.11; analytic + LLM DSE)
#   .venv-simulator/       real-SimPhony venv    (Python 3.10; torch 1.12 + torchonn)
#   configs/llm.json       LLM provider config   (copied from configs/llm.example.json)
#
# Usage:
#   bash scripts/env/bootstrap.sh                     # full (agent + simulator)
#   SKIP_SIMULATOR=1 bash scripts/env/bootstrap.sh    # agent only (fast, offline-ish)
#   WITH_SENTENCE_TRANSFORMERS=1 bash scripts/env/bootstrap.sh
#   USE_AGENT_LOCK=1 bash scripts/env/bootstrap.sh    # exact pins (pulls CUDA torch ~5GB)
#   AGENT_PY=3.12 SIM_PY=3.10 bash scripts/env/bootstrap.sh
#   DRY_RUN=1 bash scripts/env/bootstrap.sh           # print actions, change nothing
#
# Prerequisites: bash, git, curl.  The simulator additionally needs g++, make and
# ~2 GB of network (torch 1.12.1 + torchonn build + CACTI compile).
# Idempotent: re-running reuses existing venvs / toolchains / built binary.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

AGENT_PY="${AGENT_PY:-3.14}"                 # runtime needs >=3.11 (langgraph)
SIM_PY="${SIM_PY:-3.10}"                     # torch 1.12.1 needs <=3.10
WITH_SENTENCE_TRANSFORMERS="${WITH_SENTENCE_TRANSFORMERS:-0}"
USE_AGENT_LOCK="${USE_AGENT_LOCK:-0}"
SKIP_SIMULATOR="${SKIP_SIMULATOR:-0}"
DRY_RUN="${DRY_RUN:-0}"

TOOLS="$ROOT/.tools"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_CACHE_DIR="$TOOLS/cache"
export MPLCONFIGDIR="$TOOLS/mplconfig"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
run() { if [ "$DRY_RUN" = "1" ]; then printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

log "[1/5] prerequisites"
have git  || die "git is required"
have curl || die "curl is required"
[ -f "$ROOT/requirements.txt" ] || die "run from the project root (requirements.txt missing)"
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$TOOLS" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$MPLCONFIGDIR"
fi

log "[2/5] ensure uv (.tools/uv)"
if [ -x "$TOOLS/uv" ]; then
  UV="$TOOLS/uv"; echo "    reuse $UV"
elif have uv; then
  UV="$(command -v uv)"; echo "    use system uv: $UV"
else
  echo "    installing uv into $TOOLS ..."
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [dry-run] curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=%q sh\n' "$TOOLS"
    UV="$TOOLS/uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS" sh
    UV="$TOOLS/uv"
  fi
fi
[ "$DRY_RUN" = "1" ] || [ -x "$UV" ] || die "uv not found/executable at $UV"

log "[3/5] agent venv  .venv-agent  (python $AGENT_PY)"
run "$UV" python install "$AGENT_PY"
if [ -x "$ROOT/.venv-agent/bin/python" ]; then
  echo "    reuse existing .venv-agent"
else
  run "$UV" venv --python "$AGENT_PY" "$ROOT/.venv-agent"
fi
if [ "$USE_AGENT_LOCK" = "1" ] && [ -f "$ROOT/scripts/env/requirements-agent.lock.txt" ]; then
  echo "    installing EXACT pins from scripts/env/requirements-agent.lock.txt"
  run "$UV" pip install --python "$ROOT/.venv-agent/bin/python" -r "$ROOT/scripts/env/requirements-agent.lock.txt"
else
  run "$UV" pip install --python "$ROOT/.venv-agent/bin/python" -r "$ROOT/requirements.txt"
  if [ "$WITH_SENTENCE_TRANSFORMERS" = "1" ]; then
    run "$UV" pip install --python "$ROOT/.venv-agent/bin/python" "sentence-transformers>=2.7"
  fi
fi

if [ "$SKIP_SIMULATOR" = "1" ]; then
  log "[4/5] simulator venv SKIPPED (SKIP_SIMULATOR=1)"
else
  log "[4/5] simulator venv  .venv-simulator  (python $SIM_PY)"
  have g++  || die "g++ is required (torchonn C++ extension + CACTI build)"
  have make || die "make is required (CACTI build)"
  run "$UV" python install "$SIM_PY"
  if [ "$DRY_RUN" = "1" ]; then
    SIM_PY_BIN="<uv python find $SIM_PY>"
  else
    SIM_PY_BIN="$("$UV" python find "$SIM_PY")"
  fi
  echo "    base interpreter: $SIM_PY_BIN"
  # Delegate to the project's proven installer (venv + wheels + patches + CACTI).
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [dry-run] PYTHON=%q bash %q\n' "$SIM_PY_BIN" "$ROOT/scripts/env/setup_real.sh"
  else
    PYTHON="$SIM_PY_BIN" bash "$ROOT/scripts/env/setup_real.sh"
  fi
fi

log "[5/5] LLM config"
if [ ! -f "$ROOT/configs/llm.json" ] && [ -f "$ROOT/configs/llm.example.json" ]; then
  run cp "$ROOT/configs/llm.example.json" "$ROOT/configs/llm.json"
  echo "    created configs/llm.json (edit provider/model/base_url as needed)"
else
  echo "    configs/llm.json already present"
fi

log "done"
cat <<EOF

============================== summary ==============================
  agent venv     : .venv-agent/bin/python
  simulator venv : $([ "$SKIP_SIMULATOR" = "1" ] && echo "(skipped)" || echo ".venv-simulator/bin/python")
  llm config     : configs/llm.json (from example; secrets via OPENAI_API_KEY)

  Next:
    export OPENAI_API_KEY=sk-...            # only for a real (non-mock) LLM
    .venv-agent/bin/python main.py --provider mock --iterations 4
    # with a real LLM (simulator venv must exist — real mode is the only mode):
    .venv-agent/bin/python main.py --provider <p> --model <m> \\
        --nn-model-factory tests.factory:build_vit_tiny --workload vit_tiny
====================================================================
EOF
