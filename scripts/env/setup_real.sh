#!/usr/bin/env bash
# One-shot setup for SimPhony "real" mode: creates the Python 3.10 venv,
# installs the frozen dependency set, patches the two known upstream issues
# (torch GNU_STACK + pyutils.datasets), and compiles the CACTI binary.
#
# Prerequisites (not automated):
#   - Python 3.10 (torch 1.12.1 does not support 3.11+)
#   - g++ (x86_64) for the torchonn C++ extension and the CACTI build
#   - network access (downloads ~2 GB of wheels)
#
# Idempotent: safe to re-run; existing venv / built binary are reused.
set -euo pipefail

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_VENV="$_PROJECT_ROOT/.venv-real"

echo "==> [1/8] locating Python <=3.10"
_PYTHON="${PYTHON:-}"
if [ -z "$_PYTHON" ]; then
  for c in python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 && { _PYTHON="$c"; break; }
  done
fi
[ -n "$_PYTHON" ] || { echo "error: no python3 found" >&2; exit 1; }
"$_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 10) else 1)' \
  || { echo "error: need Python <=3.10 (torch 1.12); got $("$_PYTHON" --version 2>&1)" >&2; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "error: g++ required" >&2; exit 1; }
echo "    using $_PYTHON ($("$_PYTHON" --version 2>&1))"

echo "==> [2/8] creating venv $_VENV"
[ -x "$_VENV/bin/python" ] || "$_PYTHON" -m venv "$_VENV"
"$_VENV/bin/python" -m pip install -q --upgrade pip setuptools==80.10.2 wheel

echo "==> [3/8] installing torch/torchvision (CPU-only, PyTorch index)"
"$_VENV/bin/python" -m pip install -q \
  torch==1.12.1+cpu torchvision==0.13.1+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> [4/8] installing torchonn from git (C++ ext, --no-build-isolation)"
"$_VENV/bin/python" -m pip install -q --no-build-isolation \
  "torchonn @ git+https://github.com/JeremieMelo/pytorch-onn@cd9de49b82fbdc7d5d7e7b46f0eb30cb3ba7821c"

echo "==> [5/8] installing torchonn-pyutils from git (renamed dist + datasets patch)"
_TMP="$(mktemp -d)"
trap 'rm -rf "$_TMP"' EXIT
git clone -q https://github.com/JeremieMelo/pyutility "$_TMP/pyutility"
git -C "$_TMP/pyutility" checkout -q f31eeb5d7a1988f14720ddd4f856c68d22c58b40
# dist is "torchonn-pyutils" (import name stays "pyutils") to match requirements-real.txt
sed -i 's/name="pyutils"/name="torchonn-pyutils"/' "$_TMP/pyutility/setup.py"
"$_VENV/bin/python" -m pip install -q --no-build-isolation "$_TMP/pyutility"

echo "==> [6/8] disabling pyutils.datasets (needs torchvision, unused by SimPhony)"
"$_VENV/bin/python" - <<'PY'
import pathlib, sysconfig
init = pathlib.Path(sysconfig.get_paths()["purelib"]) / "pyutils" / "__init__.py"
text = init.read_text()
old = "from pyutils.datasets import *"
new = ("# NOTE: pyutils.datasets requires torchvision and is not used by SimPhony,\n"
       "# so it is disabled here to keep the dependency set minimal.\n"
       "# from pyutils.datasets import *")
if old in text and "disabled here" not in text:
    init.write_text(text.replace(old, new))
    print("    patched", init)
else:
    print("    already patched (or line not found)")
PY

echo "==> [7/8] installing remaining runtime deps"
"$_VENV/bin/python" -m pip install -q \
  mmcv-lite==2.0.0 mmengine==0.7.4 networkx==2.8.8 multimethod==2.0.2 \
  pyyaml==6.0 einops matplotlib pandas scipy scienceplots svglib \
  opencv-python termcolor

echo "==> [8/8] patching torch GNU_STACK + compiling CACTI"
"$_VENV/bin/python" "$_PROJECT_ROOT/scripts/env/patch_gnustack.py"
( cd "$_PROJECT_ROOT/photon_tuner/simulators/simphony/cacti" && make -f cacti.mk )

cat <<EOF

[setup_real] done.
Next:
  1. cp configs/llm.example.json configs/llm.json   # and fill in api_key
  2. source scripts/env/activate_real.sh
  3. python3 main.py --mode real --provider <p> --model <m> \\
       --nn-model-factory tests.factory:build_vit_tiny --workload vit_tiny
EOF
