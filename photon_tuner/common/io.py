"""Dependency-free JSON / CSV helpers shared across the photonic tuner.

Everything here relies only on the standard library so the framework can run
offline on a plain Python 3.10+ interpreter (no pandas or third-party IO).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def read_json(path: Path | str) -> Dict[str, Any]:
    """Load a UTF-8 JSON document from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON, creating any missing directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a sequence of dicts to a CSV file with the given column order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_first_row(path: Path | str) -> Dict[str, float]:
    """Return the first CSV row as ``{column: float}``, skipping non-numeric cells."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    numeric: Dict[str, float] = {}
    for key, value in first.items():
        if value in (None, ""):
            continue
        try:
            numeric[key] = float(value)
        except ValueError:
            continue
    return numeric


def load_metrics(path: Path | str) -> Dict[str, float]:
    """Read scalar metrics from either a JSON object or a one-row CSV file."""
    target = Path(path)
    if target.suffix.lower() == ".json":
        data = read_json(target)
        return {
            str(key): float(value) for key, value in data.items() if isinstance(value, (int, float))
        }
    return load_csv_first_row(target)
