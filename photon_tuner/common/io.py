"""光子调优器共享的、无外部依赖的 JSON / CSV 辅助函数。

这里的一切仅依赖标准库，使框架能在纯 Python 3.10+ 解释器上离线运行
（无需 pandas 或第三方 I/O 库）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def read_json(path: Path | str) -> Dict[str, Any]:
    """从磁盘加载一个 UTF-8 JSON 文档。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    """将 ``payload`` 写成缩进的 JSON，并创建缺失的目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path | str, record: Dict[str, Any]) -> None:
    """将一条 JSON 记录追加为一行 JSONL，并创建缺失的目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path | str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """按给定列顺序将一组字典写入 CSV 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_first_row(path: Path | str) -> Dict[str, float]:
    """返回 CSV 首行，形如 ``{列名: float}``，跳过非数值单元格。"""
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
    """从 JSON 对象或单行 CSV 文件中读取标量指标。"""
    target = Path(path)
    if target.suffix.lower() == ".json":
        data = read_json(target)
        return {
            str(key): float(value) for key, value in data.items() if isinstance(value, (int, float))
        }
    return load_csv_first_row(target)
