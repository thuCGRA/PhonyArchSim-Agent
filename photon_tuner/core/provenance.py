"""器件库来源审计（可追溯性检查）。

将"每个 ``cfgs`` 数值都能追溯到真实论文"这一规则落实为可执行的检查，
而非口头承诺。每个变体被归类为以下三类之一：

* ``traceable``   — 携带非空 ``ref``（论文 DOI / 会议论文集 URL）。
* ``baseline``    — 可搜索但 ``ref`` 为空：即内置的 SimPhony 器件库基线
  （ScopeX-ASU/SimPhony，arXiv:2411.13715 / DAC 2025）。它们以该批量来源上报，
  绝不猜测逐器件的 DOI，因为凭空编造 DOI 本身就是造假。
* ``placeholder`` — ``ref`` 为空且位于 :data:`EXCLUDED_SEARCH_VARIANTS` 中
  （仿真器所需的理想化/结构默认值，例如 ``P_pi=0`` 的 ``Customized_PS``）。
  这些并非真实论文器件。

:func:`verify_refs` 还会解析已有的 ``ref`` 值，使验收标准中
"DOI 可解析" 一项可以被实际检查，而非仅凭假设。
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

import yaml

from photon_tuner.core.search_space import EXCLUDED_SEARCH_VARIANTS

CLASS_ORDER = ("traceable", "baseline", "placeholder")

# 内置（无 ref）SimPhony 基线器件的批量来源。
BASELINE_SOURCE: Dict[str, str] = {
    "source": "SimPhony vendored device library (ScopeX-ASU/SimPhony)",
    "ref": "https://arxiv.org/abs/2411.13715",
    "note": "curated baseline; individual device-level DOI not recorded upstream",
}

# 为每个 *baseline*（ref 为空、可搜索）变体提供的逐器件来源说明。
# 这使 "未报告字段明确省略或标注" 在逐器件层面明确化，而非使用笼统的统一说明，
# 同时绝不猜测逐器件 DOI。每个 baseline 变体都必须在此有对应条目
# （test_provenance 会强制检查全覆盖），从而确保新增的无 ref 器件无法
# 未经标注地漏过。
#
# 子类别：
#   * "LT" baseline  — SimPhony 的 'LT' 子架构（Hamerly-2019 启发的 crossbar）。
#   * framework default — 结构/理想化的仿真器默认值。
#   * ACADEMIA reference — 学术模型，上游未填写 ref，论文未定位。
#   * electronic / generic / unresolved — 按标签区分。
BASELINE_PROVENANCE_NOTES: Dict[str, str] = {
    # SimPhony 'LT' 基线设计（上游未记录逐器件论文）。
    "coupler/LT_DC_2x2": "SimPhony 'LT' baseline directional coupler (5.25um, IL 0.33dB); individual device DOI not recorded upstream",
    "mzm/LT_MZM": "SimPhony 'LT' baseline MZM (260um, IL 1.2dB, ER 10dB, 450fJ/bit @25G); individual device DOI not recorded upstream",
    "photodetector/LT_PD": "SimPhony 'LT' baseline photodetector (1.1mW dynamic); individual device DOI not recorded upstream",
    "micro_comb/LT_Micro_Comb": "SimPhony 'LT' baseline micro-comb (1184nm x 1184um); individual device DOI not recorded upstream",
    "mrr_rerouter/LT_MRR_REROUTER": "SimPhony 'LT' baseline microring rerouter (IL 0.93dB, 0.275mW); individual device DOI not recorded upstream",
    "y_branch/LT_Y_Branch": "SimPhony 'LT' baseline y-branch (IL 0.1dB); individual device DOI not recorded upstream",
    "off_chip_laser/LT_Laser": "SimPhony 'LT' off-chip laser model (WPE 0.3, +2dB coupling); individual device DOI not recorded upstream",
    # 框架结构默认值（按设计并非真实论文器件）。
    "coupler/Customized_DC_2x2": "framework structural default directional coupler; not a published device",
    "laser_splitter/Customized_Laser_Splitter": "framework structural default splitter; not a published device",
    "mmi/Customized_1x10_MMI": "framework structural default 1x10 MMI splitter; not a published device",
    "on_chip_laser/Customized_Laser": "framework on-chip laser model (WPE 0.2); not a published device",
    "large_insertion/Large_Insertion_Test": "test fixture (50dB IL, -45dB crosstalk); not a physical device",
    # 'ACADEMIA' 学术参考模型（上游 ref 为空；论文未定位）。
    "coupler/ACADEMIA_DC_2x2": "academic directional-coupler model (200um, IL 0.5dB); per-device DOI not recorded upstream",
    "crossing/ACADEMIA_WG_Crossing": "academic waveguide-crossing model (100um, IL 0.25dB, -45dB crosstalk); per-device DOI not recorded upstream",
    "crossing/ACADEMIA_WG_Crossing_Linear": "academic waveguide-crossing model (21um, IL 0.13dB, -35dB crosstalk); per-device DOI not recorded upstream",
    "crossing/ACADEMIA_WG_Crossing_Trigonometric": "academic waveguide-crossing model (8um, IL 0.23dB, -30dB crosstalk); per-device DOI not recorded upstream",
    "mzm/ACADEMIA_EO_MZM_1x1": "academic MZM model (1600um, 450fJ/bit @50G, ER 15dB); per-device DOI not recorded upstream",
    "mzm/ACADEMIA_EO_MZM_2x2": "academic MZM model (1600um, 450fJ/bit @25G, ER 15dB); per-device DOI not recorded upstream",
    "phase_shifter/ACADEMIA_TO_PS": "academic thermo-optic phase shifter (P_pi 30mW, 10us); per-device DOI not recorded upstream",
    # 电子器件（上游 ref 为空）。
    "dac/DAC_1": "16nm 12-bit 14GS/s DAC (169mW, FoM 33.6); per-device DOI not recorded upstream and not located via Crossref",
    "integrator/Integrator_1": "integrator (560um2, 0.3mW); per-device DOI not recorded upstream",
    # 通用 / 未解析。
    "waveguide/WG_Single_Strip": "standard SOI strip waveguide (500nm, 0.11dB/cm); generic technology, no single paper",
    "mzm/SL_MZM": "MZM (250um, 50fJ/bit @25G, ER 6dB); 'SL' prefix unresolved, per-device DOI not located",
    # 第 2 轮物理构造（无单一论文来源；以库内真实器件为锚做同量级外推，
    # 绝不冒充任何论文的实测值）。
    "dac/DAC_LP_28nm": "constructed (no single paper): extrapolated from DAC_4 (16nm, 28GS/s, 88mW, 0.03mm2) to a 28nm/20GS/s node; supplies the low-power/large-footprint envelope anchor",
    "dac/DAC_HS_16nm": "constructed (no single paper): extrapolated from DAC_5 (16nm, 56GS/s, 280mW, 0.011mm2) with more aggressive time interleaving; supplies the small-area/high-power anchor",
    "adc/ADC_SAR_2GS_LP": "constructed (no single paper): extrapolated from ADC_SAR_1 (14nm, 10GS/s, 14.8mW, 0.00285mm2) down to 2GS/s in 28nm; supplies the ultra-low-power envelope anchor",
    "adc/ADC_SAR_30GS": "constructed (no single paper): 16nm mid-speed TI-SAR placed between ADC_SAR_1 and ADC_SAR_90GS; enables a low-power option in the 10-30GHz window",
    # 权重组 / PCM 单元（以真实论文为锚的折中构造，非论文实测）。
    "weight_mrr/WG_MRR_WEIGHT_LP": "constructed (no single paper): extrapolated from Tait-2017 weight-bank MRR (WG_MRR_WEIGHT_TAIT) to a smaller-footprint/lower-hold-power point",
    "pcm_cell/PCM_CELL_LP": "constructed (no single paper): extrapolated from Feldmann-2021 PCM cell (PCM_CELL_FELDMANN) to a lower-loss/larger-footprint point",
}

_USER_AGENT = {"User-Agent": "Mozilla/5.0 (provenance-audit)"}


def load_full_catalog(devicelib_root: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """加载所有器件 YAML（不过滤搜索排除项），组织为 ``{type: {name: body}}``。"""
    catalog: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(devicelib_root, "*", "*.yml"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            continue
        for device_type, variants in data.items():
            if not isinstance(variants, dict):
                continue
            bucket = catalog.setdefault(str(device_type), {})
            for name, body in variants.items():
                if isinstance(body, dict):
                    bucket[str(name)] = body
    return catalog


def classify_variant(device_type: str, name: str, body: Dict[str, Any]) -> str:
    """返回 ``traceable`` / ``baseline`` / ``placeholder`` 之一。"""
    ref = str((body or {}).get("ref", "") or "").strip()
    if ref:
        return "traceable"
    if name in EXCLUDED_SEARCH_VARIANTS.get(device_type, frozenset()):
        return "placeholder"
    return "baseline"


def resolve_variant(device_type: str, name: str, body: Dict[str, Any]) -> Dict[str, str]:
    """单个变体的完整来源记录（类别 + ref + 来源 + 说明）。"""
    cls = classify_variant(device_type, name, body)
    ref = str((body or {}).get("ref", "") or "").strip()
    if cls == "traceable":
        return {"class": cls, "ref": ref, "source": "device YAML ref", "note": ""}
    if cls == "placeholder":
        return {"class": cls, "ref": "", "source": "idealized simulator placeholder",
                "note": "not a physical device (excluded from search)"}
    note = BASELINE_PROVENANCE_NOTES.get(
        f"{device_type}/{name}", BASELINE_SOURCE["note"]
    )
    return {"class": cls, "ref": BASELINE_SOURCE["ref"],
            "source": BASELINE_SOURCE["source"], "note": note}


def audit_device_provenance(devicelib_root: str) -> Dict[str, Any]:
    """来源报告：分类后的变体列表 + 各类型计数 + 逐变体记录。"""
    catalog = load_full_catalog(devicelib_root)
    buckets: Dict[str, List[str]] = {k: [] for k in CLASS_ORDER}
    variants: Dict[str, Dict[str, str]] = {}
    by_type: Dict[str, Dict[str, int]] = {}
    for device_type, vs in sorted(catalog.items()):
        counts: Dict[str, int] = {k: 0 for k in CLASS_ORDER}
        for name, body in sorted(vs.items()):
            cls = classify_variant(device_type, name, body)
            key = f"{device_type}/{name}"
            buckets[cls].append(key)
            variants[key] = resolve_variant(device_type, name, body)
            counts[cls] += 1
        by_type[device_type] = counts
    return {
        "total": sum(len(buckets[k]) for k in CLASS_ORDER),
        "traceable": buckets["traceable"],
        "baseline": buckets["baseline"],
        "placeholder": buckets["placeholder"],
        "by_type": by_type,
        "variants": variants,
    }


# --------------------------------------------------------------------------- #
# ref 可解析性（"DOI 可解析"）。
# --------------------------------------------------------------------------- #


def _check_http(url: str, timeout: int = 12) -> Dict[str, Any]:
    try:
        req = urllib.request.Request(url, method="GET", headers=_USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status}
    except Exception as exc:  # noqa: BLE001 — 如实报告任何失败
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_doi(doi: str, timeout: int = 12) -> Dict[str, Any]:
    """通过 Crossref（权威注册机构）解析 DOI，而非脆弱的 HEAD 请求。"""
    url = f"https://api.crossref.org/works/{doi}"
    try:
        req = urllib.request.Request(url, method="GET", headers=_USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
            title = (body.get("message", {}).get("title") or [""])[0]
            return {"ok": True, "status": resp.status, "title": title[:120]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def verify_refs(devicelib_root: str) -> Dict[str, Any]:
    """解析每个可追溯的 ``ref``，返回逐变体的可解析性报告。

    DOI 类型的 ref 通过 Crossref 校验；其他 URL（arXiv / IEEE / Nature）
    直接抓取。该函数依赖网络，用于一次性审计，而非单元测试。
    """
    catalog = load_full_catalog(devicelib_root)
    report: Dict[str, Any] = {}
    for device_type, variants in sorted(catalog.items()):
        for name, body in sorted(variants.items()):
            ref = str((body or {}).get("ref", "") or "").strip()
            if not ref:
                continue
            key = f"{device_type}/{name}"
            if "doi.org/" in ref:
                doi = ref.split("doi.org/", 1)[1].split("?")[0].split("#")[0]
                report[key] = {"ref": ref, **_check_doi(doi)}
            else:
                report[key] = {"ref": ref, **_check_http(ref)}
    return report
