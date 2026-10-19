"""Centralised, portable path resolution for the photonic DSE framework.

Everything here is derived from *this file's* physical location, so the project
works no matter where it is checked out (``/home/fyq/epic``, a Windows mount
like ``\\wsl.localhost\\Ubuntu\\home\\fyq\\epic``, or any other prefix).  Nothing
depends on the process CWD or on a hard-coded host path.

Convention (see the portability notes in the repo):
- Resolve paths *lazily* from the anchors below; never persist an absolute host
  path into a result file.
- Result files record paths *relative to* their run/output anchor instead.
"""

from __future__ import annotations

from pathlib import Path

# photon_tuner/common/paths.py
#   parents[0] = common/   parents[1] = photon_tuner/   parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CONFIG_ROOT = PROJECT_ROOT / "configs"

# Vendored third-party SimPhony checkout (see simulators/simphony/VENDORED.md).
SIMPHONY_ROOT = PACKAGE_ROOT / "simulators" / "simphony"

# Global, cross-run history store (RAG memory).  Deliberately *not* per-run:
# it accumulates across every tuning run, so it lives in a shared data dir.
MEMORY_DB_DIR = PROJECT_ROOT / "memory" / "database"
MEMORY_DB_PATH = MEMORY_DB_DIR / "history.db"
