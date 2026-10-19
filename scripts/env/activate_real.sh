#!/usr/bin/env bash
# Activate the real-mode SimPhony environment (Python 3.10 venv + deps).
# Source it from anywhere:
#   source scripts/env/activate_real.sh
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export VIRTUAL_ENV="$_PROJECT_ROOT/.venv-real"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export UV_CACHE_DIR="$_PROJECT_ROOT/.tools/cache"
export UV_PYTHON_INSTALL_DIR="$_PROJECT_ROOT/.tools/python"
export MPLCONFIGDIR="$_PROJECT_ROOT/.tools/mplconfig"
export PYTHONPATH="$_PROJECT_ROOT/photon_tuner/simulators/simphony:$_PROJECT_ROOT"

echo "[activate_real] python=$(python --version 2>&1)  simphony_root=$_PROJECT_ROOT/photon_tuner/simulators/simphony"
