#!/usr/bin/env python3
"""生成可供实现审计和论文制表使用的能力清单。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from photon_tuner.capabilities import build_capability_manifest, validate_agent_capability_surface


def _names(items: Iterable[Dict[str, Any]]) -> str:
    return "、".join(str(item["name"]) for item in items)


def render_markdown(manifest: Dict[str, Any]) -> str:
    tool_groups: Dict[str, list] = defaultdict(list)
    for item in manifest["tools"]:
        tool_groups[item["category"]].append(item)
    skill_groups: Dict[str, list] = defaultdict(list)
    for item in manifest["atomic_skills"]:
        skill_groups[item["category"]].append(item)

    lines = [
        "# Agent 能力清单",
        "",
        "> 本文件由 `scripts/analysis/capability_inventory.py` 自动生成。",
        "",
        "## 总览",
        "",
        "| 层次 | 数量 | 说明 |",
        "| --- | ---: | --- |",
        f"| 确定性工具 | {manifest['summary']['tool_count']} | 资源接口；写工具仅供宿主调用 |",
        f"| 原子技能 | {manifest['summary']['atomic_skill_count']} | 单一专业问题；不调用其他技能 |",
        f"| 组合工作流 | {manifest['summary']['workflow_count']} | 技能间条件分支和循环 |",
        "",
        "## 工具箱（按资源分类）",
        "",
        "| 分类 | 数量 | 工具 |",
        "| --- | ---: | --- |",
    ]
    for category in sorted(tool_groups):
        items = tool_groups[category]
        lines.append(f"| {category} | {len(items)} | {_names(items)} |")
    lines += ["", "## 原子技能（按专业任务分类）", "", "| 分类 | 数量 | 技能 |", "| --- | ---: | --- |"]
    for category in sorted(skill_groups):
        items = skill_groups[category]
        lines.append(f"| {category} | {len(items)} | {_names(items)} |")
    lines += ["", "## 组合工作流", "", "| 名称 | 单一工作流目标 | 技能节点 |", "| --- | --- | --- |"]
    for workflow in manifest["composite_workflows"]:
        nodes = " → ".join(node["skill"] for node in workflow["nodes"])
        lines.append(f"| {workflow['name']} | {workflow['goal']} | {nodes} |")
    lines += ["", "## Agent 角色白名单", "", "| 角色 | 能力数 | 能力 |", "| --- | ---: | --- |"]
    for role, names in manifest.get("role_bindings", {}).items():
        lines.append(f"| {role} | {len(names)} | {'、'.join(names)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=Path("docs/design/capability_inventory.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/design/capability_inventory.md"))
    args = parser.parse_args()
    validate_agent_capability_surface()
    manifest = build_capability_manifest()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
