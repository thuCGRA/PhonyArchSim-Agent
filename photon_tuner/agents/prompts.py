"""面向基于 LangGraph 的光子 DSE 智能体的角色提示词。

提示词是*配置*而非代码：存放在 ``configs/prompts/prompts.yaml``（每个角色一个
YAML 块标量），并在导入时加载与校验。编辑 YAML 即可调整措辞而无需改动 Python。
每个提示词是相应角色智能体的 ``system_prompt``；LangGraph 运行时会自行注入工具面
与规划脚手架，因此这些提示词仅描述*领域角色*及其证据落地规则。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

# 本模块必须暴露为 ``<ROLE>_PROMPT`` 常量（以及 ``load_prompt(role)``）的六个角色名。
_ROLE_NAMES = ("model", "architecture", "topology", "device", "solution", "evaluate")

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "prompts.yaml"


def _load_prompts(path: Path) -> Dict[str, str]:
    """加载并校验提示词配置；任何缺失都快速失败。"""
    if not path.is_file():
        raise FileNotFoundError(f"prompt configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"prompt configuration must be a mapping, got {type(raw).__name__}")
    prompts: Dict[str, str] = {}
    for role in _ROLE_NAMES:
        value = raw.get(role)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompts.yaml must define a non-empty string for role {role!r}")
        prompts[f"{role.upper()}_PROMPT"] = value.strip()
    return prompts


_PROMPTS = _load_prompts(_PROMPTS_PATH)

MODEL_PROMPT = _PROMPTS["MODEL_PROMPT"]
ARCHITECTURE_PROMPT = _PROMPTS["ARCHITECTURE_PROMPT"]
TOPOLOGY_PROMPT = _PROMPTS["TOPOLOGY_PROMPT"]
DEVICE_PROMPT = _PROMPTS["DEVICE_PROMPT"]
SOLUTION_PROMPT = _PROMPTS["SOLUTION_PROMPT"]
EVALUATE_PROMPT = _PROMPTS["EVALUATE_PROMPT"]


def load_prompt(role: str) -> str:
    """返回某个角色的提示词文本（``model`` … ``evaluate``）。"""
    if role not in _ROLE_NAMES:
        raise ValueError(f"unknown role {role!r}; expected one of {_ROLE_NAMES}")
    return _PROMPTS[f"{role.upper()}_PROMPT"]
