#!/usr/bin/env bash
# Activate the agent / runtime environment (.venv-agent, Python >=3.11).
# Source it from anywhere:
#   source scripts/env/activate_agent.sh
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export VIRTUAL_ENV="$_PROJECT_ROOT/.venv-agent"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$_PROJECT_ROOT/.tools/python"
export UV_CACHE_DIR="$_PROJECT_ROOT/.tools/cache"
export MPLCONFIGDIR="$_PROJECT_ROOT/.tools/mplconfig"

echo "[activate_agent] python=$(python --version 2>&1)  project=$_PROJECT_ROOT"
