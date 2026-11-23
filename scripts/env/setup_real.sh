#!/usr/bin/env bash
# Build the CPU SimPhony environment. The public entry point is bootstrap.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/env/environment.conf"
VENV="$ROOT/.venv-simulator"
PY="${PYTHON:-python$SIM_PY}"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]] || { echo "SimPhony requires Linux x86_64 (including WSL2)." >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
for tool in curl tar make g++; do command -v "$tool" >/dev/null; done

[[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV"
py="$VENV/bin/python"
"$py" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
"$py" -m pip install --disable-pip-version-check -r "$ROOT/$SIM_REQUIREMENTS"
"$py" "$ROOT/scripts/env/patch_gnustack.py"

# These upstream training packages import themselves from setup.py before
# installation. Read their version files directly to permit a clean CPU build.
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
for repo in pyutility pytorch-onn; do
    if [[ "$repo" == pyutility ]]; then
        rev="$PYUTILITY_REV"
        pkg=pyutils
    else
        rev="$TORCHONN_REV"
        pkg=torchonn
    fi
    curl --fail --show-error --silent --location --retry 3 \
        "https://codeload.github.com/JeremieMelo/$repo/tar.gz/$rev" -o "$work/$repo.tar.gz"
    tar -xzf "$work/$repo.tar.gz" -C "$work"
    src="$work/$repo-$rev"
    if [[ "$pkg" == pyutils ]]; then
        sed -i 's/from pyutils.version import __version__/exec(open("pyutils\/version.py").read())/; s/name="pyutils"/name="torchonn-pyutils"/' "$src/setup.py"
        sed -i '/^from pyutils.datasets import \*$/d' "$src/pyutils/__init__.py"
    else
        sed -i 's/from torchonn import __version__/exec(open("torchonn\/version.py").read())/' "$src/setup.py"
    fi
    "$py" -m pip install --disable-pip-version-check --no-deps --no-build-isolation "$src"
done

make -C "$ROOT/photon_tuner/simulators/simphony/cacti" opt
"$py" -c 'from mcp.server.fastmcp import FastMCP; import torch, torchonn, pyutils'
echo "SimPhony is ready in $VENV"
