"""拓扑/电路结构搜索空间：合法模板与耦合矩阵。

五阶段流水线将 *电路结构* 作为一等搜索轴：设计点使用哪个 ``core_kind``
（光学矩阵-向量引擎）与 ``node_kind``（每个单元的运算节点），以及保证组合设计点
物理一致的跨层耦合规则。

本模块遵循两条硬约束：

1. **真实性** —— ``core_kind``/``node_kind`` 只有在 SimPhony 解析镜像能实际运行时才算
   *可求值(evaluable)*。当前 8 个 core 模板（crossbar 家族 2 + MZI mesh 3 + WDM 权重
   阵列 3）与 5 个 node 模板全部解析可求值，且 real 路径（torchonn ONNArchSimulator）
   全部打通（经 ``.venv-simulator`` 端到端验证）。
2. **单一事实来源** —— 数值耦合约束（``work_freq <= min(MZM/DAC/ADC/PD 速率链)``、
   ``in_bit <= DAC prec``、``out_bit <= ADC prec``）位于
   :class:`~photon_tuner.core.search_space.DesignSpace.violations`；
   本模块仅在其上叠加 *种类级* 兼容性（core_kind <-> node_kind），数值行则委托出去。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 合法的 core / node 种类。
# --------------------------------------------------------------------------- #

CORE_KINDS: Dict[str, Dict[str, Any]] = {
    "dptc_crossbar": {
        "evaluable": True,
        "real": True,
        "layout": "crossbar",
        "universal": True,
        "coherent": True,
        "range_forwards": 1,
        "note": "Hamerly 2019 photoelectric crossbar (10.1103/PhysRevX.9.021032); "
        "implemented by configs/design/cores/dptc.yml with the DODE node.",
    },
    "mzi_mesh_reck": {
        "evaluable": True,
        "real": True,
        "layout": "reck",
        "universal": True,
        "coherent": True,
        "range_forwards": 1,
        "note": "Reck triangular MZI mesh: depth 2N-3, N(N-1)/2 MZIs, crossing-bearing "
        "(10.1103/PhysRevLett.73.58). Evaluated by the phase-2 analytic mesh evaluator "
        "(worst-case IL uses the full 2N-3 depth cascade) and, with the corrected "
        "n_modes x n_modes partition miniblock, by the real ONNArchSimulator path.",
    },
    "mzi_mesh_clements": {
        "evaluable": True,
        "real": True,
        "layout": "clements",
        "universal": True,
        "coherent": True,
        "range_forwards": 1,
        "note": "Clements rectangular MZI mesh: depth N, N(N-1)/2 MZIs, crossing-free. "
        "Implemented by the phase-2 analytic mesh evaluator (simulators/mesh_model.py + "
        "build_mesh_sub_arch) with the tunable-MZI node; evaluable in both analytic and "
        "real ONNArchSimulator modes.",
    },
    "mzi_mesh_butterfly": {
        "evaluable": True,
        "real": True,
        "layout": "butterfly",
        "universal": False,
        "coherent": True,
        "range_forwards": 1,
        "note": "Butterfly FFT mesh: M*log2(M)/2 MZIs, depth log2(M), non-universal, needs crossings. "
        "MZI count/depth follow the classical FFT butterfly (Cooley-Tukey 1965, "
        "10.1090/S0025-5718-1965-0178586-1); each butterfly maps to one 2x2 MZI via the "
        "standard coupler+phase-shifter decomposition (Clements/Reck). No single photonic-mesh "
        "DOI (a prior Tang-2021 attribution was retracted: 10.1021/acsphotonics.1c00419 is a "
        "10-port *universal* processor). Structural evaluation (device count/depth/IL/area/"
        "latency, following the paper's Tab. I as a Pos-Neg non-universal PTC with I=1) is "
        "analytic-evaluable; real=True（ONNArchSimulator 以标准 MZI-mesh 网表表达其结构，"
        "非专用光子 FFT 网表）。",
    },
    "mrr_weight_bank": {
        "evaluable": True,
        "real": True,
        "layout": "weight_array",
        "universal": True,
        "coherent": False,
        "range_forwards": 2,
        "note": "Tait 2017 silicon photonic weight bank (10.1038/s41598-017-07754-z): N 微环并行做 "
        "WDM 权重（drop-port 耦合比），差分检测实现正负 → R+，I=2。分析可求值 "
        "（build_weight_array_sub_arch）；real=True（ONNArchSimulator 消费通用 netlist）。",
    },
    "pcm_crossbar": {
        "evaluable": True,
        "real": True,
        "layout": "weight_array",
        "universal": True,
        "coherent": False,
        "range_forwards": 4,
        "note": "Feldmann 2021 integrated photonic tensor core (10.1038/s41586-020-03070-1): "
        "GST 相变单元吸收态编码权重（R+），四段分解实现全值域 → I=4，写入 ~100ns。"
        "分析可求值（build_weight_array_sub_arch）；real=True（ONNArchSimulator 消费通用 netlist）。",
    },
    "wdm_link": {
        "evaluable": True,
        "real": True,
        "layout": "weight_array",
        "universal": True,
        "coherent": True,
        "range_forwards": 1,
        "note": "Xu 2021 single WDM link (10.1038/s41586-020-03063-0): 微梳谱线静态编码权重 + "
        "宽带调制器编码输入 + PD 累加，单空间通道 × W 波长，I=1。分析可求值 "
        "（build_weight_array_sub_arch）；real=True（ONNArchSimulator 消费通用 netlist）。",
    },
    # 注意：没有 ``tempo_crossbar``。TeMPO 的时域复用并不构成一种独立电路拓扑——
    # 它的全部物理内容（crossbar 网表 + temporal_accum 时域累加轴 + 慢光 EOM 器件
    # 选型）都能由 ``dptc_crossbar`` 本身表达，因此注册成第二个 core_kind 只会复制
    # 一份逐位相同的评估结果。详见 ``tests/test_topology_duplicates.py``。
}

NODE_KINDS: Dict[str, Dict[str, Any]] = {
    "dode": {
        "evaluable": True,
        "real": True,
        "note": "Dot-product engine node (y-branch + phase shifter + coupler + balanced PDs); "
        "implemented by configs/design/nodes/dode.yml.",
    },
    "tunable_mzi": {
        "evaluable": True,
        "real": True,
        "note": "2x2 unitary rotation node (2 directional couplers + 2 phase shifters). "
        "Implemented by the phase-2 analytic mesh evaluator (_tunable_mzi_node); evaluable "
        "in both analytic and real ONNArchSimulator modes.",
    },
    "weight_mrr": {
        "evaluable": True,
        "real": True,
        "note": "微环权重单元（Tait 2017 权重组，drop-port 耦合比编码权重）；"
        "分析可求值（_weight_element_node）；real=True（通用 netlist，经 .venv-simulator 验证）。",
    },
    "pcm_cell": {
        "evaluable": True,
        "real": True,
        "note": "GST 相变权重单元（Feldmann 2021，吸收态编码权重）；"
        "分析可求值（_weight_element_node）；real=True（通用 netlist，经 .venv-simulator 验证）。",
    },
    "wdm_modulator": {
        "evaluable": True,
        "real": True,
        "note": "单 WDM 链路调制器节点（微梳谱线权重 × 宽带调制器输入）；"
        "分析可求值（_weight_element_node，器件为 mzm）；real=True（通用 netlist，经 .venv-simulator 验证）。",
    },
}

# core_kind -> 兼容的 node_kinds。种类映射即结构耦合：
# crossbar 需要点积节点，mesh 需要 tunable-MZI 节点。
COUPLING_MATRIX: Dict[str, Dict[str, List[str]]] = {
    "dptc_crossbar": {"node_kinds": ["dode"]},
    "mzi_mesh_reck": {"node_kinds": ["tunable_mzi"]},
    "mzi_mesh_clements": {"node_kinds": ["tunable_mzi"]},
    "mzi_mesh_butterfly": {"node_kinds": ["tunable_mzi"]},
    "mrr_weight_bank": {"node_kinds": ["weight_mrr"]},
    "pcm_crossbar": {"node_kinds": ["pcm_cell"]},
    "wdm_link": {"node_kinds": ["wdm_modulator"]},
}

# 每个 (core_kind, node_kind) 组合在网表中需要的器件类型槽位。
# 对应 SimPhony 的 core/node 器件表，用于标记
# ``hardware["devices"]`` 中缺少必需器件类型的模板。
REQUIRED_DEVICE_SLOTS: Dict[str, Dict[str, List[str]]] = {
    "dptc_crossbar": {
        "dode": [
            "mzm", "dac", "adc", "tia", "mrr_rerouter", "laser_splitter",
            "micro_comb", "on_chip_laser", "waveguide",
            "phase_shifter", "photodetector", "coupler", "y_branch",
        ],
    },
    "mzi_mesh_reck": {
        "tunable_mzi": [
            "off_chip_laser", "photodetector", "adc", "tia", "dac", "waveguide",
            "coupler", "phase_shifter", "crossing",
        ],
    },
    "mzi_mesh_clements": {
        "tunable_mzi": [
            "off_chip_laser", "photodetector", "adc", "tia", "dac", "waveguide",
            "coupler", "phase_shifter",
        ],
    },
    "mzi_mesh_butterfly": {
        "tunable_mzi": [
            "off_chip_laser", "photodetector", "adc", "tia", "dac", "waveguide",
            "coupler", "phase_shifter", "crossing",
        ],
    },
    "mrr_weight_bank": {
        "weight_mrr": [
            "off_chip_laser", "micro_comb", "photodetector", "adc", "tia", "dac",
            "waveguide", "weight_mrr",
        ],
    },
    "pcm_crossbar": {
        "pcm_cell": [
            "off_chip_laser", "micro_comb", "photodetector", "adc", "tia", "dac",
            "waveguide", "pcm_cell",
        ],
    },
    "wdm_link": {
        "wdm_modulator": [
            "off_chip_laser", "micro_comb", "photodetector", "adc", "tia", "dac",
            "waveguide", "mzm",
        ],
    },
}

# WDM 权重阵列范式（MRR 权重组 / PCM 交叉开关 / 单 WDM 链路）使用独立的器件表
# ``hardware["devices"]["weight_array"]``，与 crossbar（core/node）和 mesh（mesh）并列。
WEIGHT_ARRAY_CORE_KINDS = frozenset({"mrr_weight_bank", "pcm_crossbar", "wdm_link"})

# --------------------------------------------------------------------------- #
# 参数 × 拓扑族的有效性元数据。
#
# 背景：三个网表构建器（``netlist.build_sub_arch`` / ``build_mesh_sub_arch`` /
# ``build_weight_array_sub_arch``）对同一批架构参数的处理并不对称——有的直接采用
# ``hardware`` 值，有的被拓扑层参数遮蔽，有的被无条件写死。若不显式声明，搜索层会
# 继续在无效维度上消耗评估预算，且闸门无法给出可操作反馈。
#
# 本表是"某参数在某拓扑族下是否真正生效"的**单一事实来源**，供三处消费：
#   1. :meth:`DesignSpace.candidates` —— 按 ``core_kind`` 收窄候选；
#   2. ``agents.evaluate`` 的 ``_ARCH_LEGAL_PARAMS`` —— 按拓扑收窄允许推荐项；
#   3. 结构合法性闸门 —— 拒绝时给出可操作理由，而非静默通过。
#
# 四类性质（不可混同，否则会把缺陷固化成"特性"）：
#
#   ``effective``                 正常生效。
#   ``fixed_by_design``           物理上就该固定；禁止搜索，理由见 ``reason``。
#   ``superseded_by``             被另一参数取代；禁止搜索并改用 ``by`` 指定的参数。
#   ``unsupported_by_backend``    物理上有意义、但网表尚未接线；标记为可实现缺口
#                                 （``implementable=True``），而非当作"无效"。
#   ``not_applicable``            该拓扑族根本没有这个概念。
#
# 未在 ``PARAM_SCOPE`` 中出现的参数默认对全部拓扑族 ``effective``。
# --------------------------------------------------------------------------- #

TOPOLOGY_FAMILIES: Dict[str, str] = {
    "dptc_crossbar": "crossbar_family",
    "mzi_mesh_reck": "mesh_family",
    "mzi_mesh_clements": "mesh_family",
    "mzi_mesh_butterfly": "mesh_family",
    "mrr_weight_bank": "weight_array",
    "pcm_crossbar": "weight_array",
    "wdm_link": "weight_array",
}

# 拓扑族 -> 人类可读名称（用于闸门消息）。
FAMILY_LABELS: Dict[str, str] = {
    "crossbar_family": "crossbar 家族",
    "mesh_family": "MZI 网格",
    "weight_array": "WDM 权重阵列",
}

PARAM_SCOPE: Dict[str, Dict[str, Any]] = {
    "core_height": {
        "families": {
            "mesh_family": {
                "status": "superseded_by",
                "by": "mesh_rank",
                "reason": "网格的矩阵维度由拓扑层 mesh_rank 决定，height 只是其派生量"
                "（MZI 总数 = 2N(N-1)），两者描述同一物理量且 mesh_rank 优先。",
                "action": "网格阶数 N 改由 topology 阶段的 mesh_rank 控制；架构阶段请改用 tiles / cores_per_tile / work_freq 等生效参数",
            },
            "weight_array": {
                "status": "superseded_by",
                "by": "mesh_rank",
                "reason": "WDM 权重阵列的并行权重数即 height，由 mesh_rank 决定。",
                "action": "并行权重数改由 topology 阶段的 mesh_rank 控制；架构阶段请改用 tiles / cores_per_tile / work_freq 等生效参数",
            },
        },
    },
    "core_width": {
        "families": {
            "mesh_family": {
                "status": "superseded_by",
                "by": "mesh_rank",
                "reason": "网格核心的 width 恒为 1（height 已表示 MZI 数），矩阵维度来自"
                " mesh_rank。",
                "action": "网格阶数 N 改由 topology 阶段的 mesh_rank 控制",
            },
            "weight_array": {
                "status": "superseded_by",
                "by": "mesh_rank",
                "reason": "权重阵列的 width 恒为 1（height 已表示并行权重数），阵列宽度来自"
                " mesh_rank。",
                "action": "并行权重数改由 topology 阶段的 mesh_rank 控制",
            },
        },
    },
    "num_wavelength": {
        "families": {
            "mesh_family": {
                # 注意：这是**可实现的缺口**，不是"无意义"。
                # Simphony 的四条工作流都消费 N（utils.evaluate_factors 会把 \bN\b
                # 替换成 num_wavelength；energy/area/insertion_loss 各自读取
                # core.num_wavelength），但网格网表的 scaling_rules 里一个 N 都没有，
                # 且 num_wavelength 被硬编码为 1。
                "status": "unsupported_by_backend",
                "implementable": True,
                "reason": "波长并行网格在物理上是真实架构方向，Simphony 也支持 N；"
                "但当前网格网表未把 N 接入 scaling_rules，且 num_wavelength 硬编码为 1。",
                "action": "暂不可搜索；如需启用，需为网格网表设计 N 的器件缩放规则"
                "（PD/ADC/激光是否随波长数缩放）并在 real 模式下校验",
            },
            "weight_array": {
                "status": "superseded_by",
                "by": "mesh_rank",
                "reason": "MRR/PCM 权重阵列一个波长承载一个权重，"
                "num_wavelength 与并行权重数（mesh_rank）是物理恒等关系。",
                "action": "并行权重数改由 topology 阶段的 mesh_rank 控制",
            },
        },
    },
    "dataflow": {
        "families": {
            "mesh_family": {
                "status": "fixed_by_design",
                "reason": "静态网格靠热光相位调谐重配置（reconfig_time_ns=10000，约 10 µs），"
                "比时钟周期慢约三个数量级，物理上无法逐周期换权重，故固定为"
                " weight_stationary。",
                "action": "该拓扑下数据流不可搜索",
                "impl_reference": "core.node.weight_stationary_phase_shifter = True",
            },
            "weight_array": {
                "status": "fixed_by_design",
                "reason": "权重由 MRR/PCM 静态保持或由微梳静态编码，架构上即"
                " weight-stationary。",
                "action": "该拓扑下数据流不可搜索",
            },
        },
    },
    "mesh_rank": {
        "families": {
            "crossbar_family": {
                "status": "not_applicable",
                "reason": "crossbar 的矩阵维度由 architecture 层的 core_height/core_width "
                "直接给出，不存在酉网格阶数这一概念。",
                "action": "crossbar 的阵列几何改由 architecture 阶段的 core_height / core_width 设定",
            },
        },
    },
}


# 受拓扑族影响的全部参数（架构层 + 拓扑层里"按拓扑分族"的那些）。
SCOPED_PARAMS: List[str] = list(PARAM_SCOPE)


# --------------------------------------------------------------------------- #
# 物理强制值：网表装配器会**忽略搜索值、写死成某个常量**的参数。
#
# 与 ``PARAM_SCOPE`` 的区别：那里声明"该参数在此族下不生效"，这里记录"若仍给出
# 一个值，装配器实际会把它覆盖成什么"。没有这张表，一个设计点可以声称
# ``dataflow=output_stationary`` / ``num_wavelength=12``，而网表按
# ``weight_stationary`` / ``1`` 求值——设计点与仿真事实静默不一致，agent 会
# 基于一个从未被仿真的配置做决策。
#
# 只收录**常量**强制；由 ``mesh_rank`` 派生的量（核心高度/并行权重数）已在
# ``PARAM_SCOPE`` 里标为 ``superseded_by``，其取值随 ``mesh_rank`` 变化，不在此表。
# --------------------------------------------------------------------------- #

PHYSICALLY_FORCED: Dict[str, Dict[str, Any]] = {
    "core_width": {
        "families": {
            "mesh_family": {
                "value": 1,
                "why": "MZI 网格核心的 width 恒为 1；height 已表示 MZI 数量，"
                "矩阵维度由顶层的 matrix_height/matrix_width（= mesh_rank）给出。",
            },
            "weight_array": {
                "value": 1,
                "why": "波长并行权重阵列 width 恒为 1；height 表示并行权重数"
                "（= 波长数 = mesh_rank）。",
            },
        },
    },
    "num_wavelength": {
        "families": {
            "mesh_family": {
                "value": 1,
                "why": "相干酉网格的移相器有色散（φ = 2π·Δn_eff·L/λ），不同波长得到"
                "不同矩阵，无法用 WDM 并行权重；且网表未接入波长解复用器与 λ×n 探测器。",
            },
            "weight_array": {
                "derive_from": "mesh_rank",
                "why": "一个波长承载一个权重：并行权重数与波长数是同一个物理量，"
                "由 topology 层的 mesh_rank 给出。",
            },
        },
    },
    "dataflow": {
        "families": {
            "mesh_family": {
                "value": "weight_stationary",
                "why": "热光移相器重配置约 10 µs，比时钟周期慢三个数量级，"
                "物理上无法逐周期换权重。",
            },
            "weight_array": {
                "value": "weight_stationary",
                "why": "MRR/PCM 权重由谐振或相变态静态保持，架构上即 weight-stationary。",
            },
        },
    },
}


#: 强制参数在 hardware 字典中的位置（与 ``DesignSpace`` 的 ``map`` 保持一致）。
_FORCED_LOCATIONS: Dict[str, Tuple[str, str]] = {
    "core_width": ("core", "width"),
    "num_wavelength": ("core", "num_wavelength"),
    "dataflow": ("system", "dataflow"),
    "core_height": ("core", "height"),
}


def _forced_spec(param: str, core_kind: Optional[str]) -> Dict[str, Any]:
    spec = PHYSICALLY_FORCED.get(str(param)) or {}
    return (spec.get("families") or {}).get(core_family(core_kind)) or {}


def forced_value(param: str, core_kind: Optional[str] = None) -> Optional[Any]:
    """返回 ``param`` 在该拓扑族下被网表写死的常量值；无常量强制时返回 ``None``。"""
    return _forced_spec(param, core_kind).get("value")


def forced_reason(param: str, core_kind: Optional[str] = None) -> Optional[str]:
    """返回该强制值的物理依据（人类可读）。"""
    return _forced_spec(param, core_kind).get("why")


def forced_params(core_kind: Optional[str] = None) -> Dict[str, Any]:
    """该拓扑族下全部被写死成常量值的参数。"""
    family = core_family(core_kind)
    out: Dict[str, Any] = {}
    for param, spec in PHYSICALLY_FORCED.items():
        per_family = (spec.get("families") or {}).get(family) or {}
        if "value" in per_family:
            out[param] = per_family["value"]
    return out


def _effective_forced(hardware: Optional[Dict[str, Any]]) -> Dict[str, Tuple[Any, str]]:
    """解析出 ``{param: (应取的值, 依据)}``，同时处理常量与派生两类强制。"""
    hardware = hardware or {}
    core_kind = _current_core_kind(hardware)
    topo = ((hardware.get("arch") or {}).get("topology") or {})
    out: Dict[str, Tuple[Any, str]] = {}
    for param, spec in PHYSICALLY_FORCED.items():
        per_family = (spec.get("families") or {}).get(core_family(core_kind)) or {}
        why = per_family.get("why") or "topology constraint"
        if "value" in per_family:
            out[param] = (per_family["value"], why)
        elif "derive_from" in per_family:
            source = per_family["derive_from"]
            if source in topo and topo.get(source) is not None:
                out[param] = (topo.get(source), why)
    return out


def effective_forced_params(hardware: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """该设计点下全部被写死参数的**实际取值**（含由 ``mesh_rank`` 派生的那些）。"""
    return {param: value for param, (value, _why) in _effective_forced(hardware).items()}


def forced_value_violations(hardware: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """返回设计点声称的值与其拓扑物理强制值不一致的违规列表。

    这些是**物理上不成立的组合**：装配器不会采纳所声明的值，实际网表按强制值
    求值。调用方（求值闸门、证据探针、工具）据此拒绝或纠正候选，而不是让一个
    从未被仿真的配置产生数值。
    """
    hardware = hardware or {}
    core_kind = _current_core_kind(hardware)
    arch = hardware.get("arch", {}) or {}
    core = arch.get("optical_core", {}) or {}
    system = arch.get("system", {}) or {}
    out: List[Dict[str, Any]] = []
    for param, (want, why) in _effective_forced(hardware).items():
        where, field = _FORCED_LOCATIONS.get(param, (None, None))
        if where is None:
            continue
        source = core if where == "core" else system
        if field not in source:
            continue
        got = source.get(field)
        if str(got) == str(want):
            continue
        out.append({
            "param": param,
            "constraint": f"{param} == {want!r}",
            "current": got,
            "limit": want,
            "core_kind": core_kind,
            "family": core_family(core_kind),
            "message": (
                f"{param}={got!r} is physically impossible under "
                f"{core_kind!r}: the netlist builder uses {want!r} instead "
                f"({why})"
            ),
        })
    return out


def coerce_forced_values(hardware: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把被写死的参数改写为物理强制值，使设计点与仿真事实一致。

    返回新字典（不改原对象）。用于 ``DesignSpace.normalize`` 与
    ``switch_topology_template``——归一化后的设计点应当就是实际被仿真的那一个。
    """
    import copy as _copy

    result = _copy.deepcopy(hardware) if isinstance(hardware, dict) else {}
    arch = result.setdefault("arch", {})
    containers = {
        "core": arch.setdefault("optical_core", {}),
        "system": arch.setdefault("system", {}),
    }
    for param, (want, _why) in _effective_forced(result).items():
        entry = _FORCED_LOCATIONS.get(param)
        if entry is None:
            continue
        where, field = entry
        container = containers.get(where)
        if container is None or field not in container:
            continue
        current = container.get(field)
        # 保持数值类型（int 强制值 → int），避免设计点里混入无意义的浮点。
        if isinstance(want, int) and isinstance(current, (int, float)):
            container[field] = int(want)
        else:
            container[field] = want
    return result


def physical_violations(
    hardware: Optional[Dict[str, Any]],
    design_space: Optional[Any] = None,
    mode: str = "real",
) -> List[Dict[str, Any]]:
    """设计点是否物理成立——耦合合法性 + 强制值一致性 + 数值跨层约束。

    在 :func:`validate_coupling` 之上增加"被写死的参数"检查，因此能在求值前
    拒绝那些**看似合法、实际不会被仿真**的组合。
    """
    out: List[Dict[str, Any]] = list(validate_coupling(hardware, design_space, mode))
    for item in forced_value_violations(hardware):
        out.append({
            "scope": "forced_value",
            "param": item.get("param"),
            "constraint": item.get("constraint"),
            "current": item.get("current"),
            "limit": item.get("limit"),
            "message": item.get("message"),
        })
    return out



def core_family(core_kind: Optional[str]) -> str:
    """返回 ``core_kind`` 所属的拓扑族；未知种类回退到 crossbar 家族。"""
    return TOPOLOGY_FAMILIES.get(str(core_kind or ""), "crossbar_family")


def _current_core_kind(hardware: Optional[Dict[str, Any]]) -> str:
    """从 hardware 字典读取当前 ``core_kind``（缺省与装配器一致为 dptc_crossbar）。"""
    topo = ((hardware or {}).get("arch", {}) or {}).get("topology", {}) or {}
    return str(topo.get("core_kind", "dptc_crossbar"))


def param_status(param: str, core_kind: Optional[str] = None) -> Dict[str, Any]:
    """返回 ``param`` 在 ``core_kind`` 所属拓扑族下的有效性状态。

    返回字典至少包含 ``param`` / ``status`` / ``family``，并按状态附带
    ``reason``（为何如此）、``action``（agent 该如何调整）、
    ``by``（``superseded_by`` 时的替代参数）、``implementable``
    （``unsupported_by_backend`` 时是否属于可补齐的缺口）。
    """
    family = core_family(core_kind)
    spec = PARAM_SCOPE.get(str(param)) or {}
    per_family = (spec.get("families") or {}).get(family) or {}
    status = str(per_family.get("status") or "effective")
    out: Dict[str, Any] = {
        "param": str(param),
        "family": family,
        "core_kind": str(core_kind or ""),
        "status": status,
    }
    for key in ("reason", "action", "by", "implementable", "impl_reference"):
        if key in per_family:
            out[key] = per_family[key]
    return out


def is_param_effective(param: str, core_kind: Optional[str] = None) -> bool:
    """``param`` 在该拓扑族下是否真正影响仿真结果。"""
    return param_status(param, core_kind)["status"] == "effective"


def hidden_params(
    core_kind: Optional[str] = None,
    params: Optional[List[str]] = None,
) -> List[str]:
    """在该拓扑族下应**禁止搜索**的参数（``effective`` 之外的全部）。

    ``params`` 省略时使用 :data:`SCOPED_PARAMS`。
    """
    candidates = list(params) if params is not None else list(SCOPED_PARAMS)
    return [p for p in candidates if not is_param_effective(p, core_kind)]


def effective_params(
    core_kind: Optional[str] = None,
    params: Optional[List[str]] = None,
) -> List[str]:
    """在该拓扑族下**应当搜索**的参数。"""
    candidates = list(params) if params is not None else list(SCOPED_PARAMS)
    return [p for p in candidates if is_param_effective(p, core_kind)]


def param_scope_table(core_kind: Optional[str] = None) -> Dict[str, Any]:
    """导出当前拓扑族下的参数状态表（供工具与闸门消息使用）。"""
    family = core_family(core_kind)
    return {
        "core_kind": str(core_kind or ""),
        "family": family,
        "family_label": FAMILY_LABELS.get(family, family),
        "params": {p: param_status(p, core_kind) for p in SCOPED_PARAMS},
    }


def implementable_gaps() -> List[Dict[str, Any]]:
    """列出"物理有意义但后端未接通"的参数缺口（待实现的增强项）。

    这些**不是**该被永久禁止的维度，而是网表尚未接线的功能。把它们单独列出，
    是为了避免把 bug 固化成"特性"。
    """
    gaps: List[Dict[str, Any]] = []
    for param, spec in PARAM_SCOPE.items():
        for family, body in (spec.get("families") or {}).items():
            if str(body.get("status")) == "unsupported_by_backend":
                gaps.append({
                    "param": param,
                    "family": family,
                    "family_label": FAMILY_LABELS.get(family, family),
                    "implementable": bool(body.get("implementable")),
                    "reason": body.get("reason", ""),
                    "action": body.get("action", ""),
                })
    return gaps


def legal_template_names() -> List[str]:
    """可 *推荐* 的模板名（不要求可求值）。"""
    return sorted(CORE_KINDS)


def _is_evaluable(meta: Dict[str, Any], mode: str) -> bool:
    """判断某种类在 ``mode``（"analytic" 或 "real"）下是否可求值。

    "evaluable" 表示 *解析镜像* 能运行它；``real`` 标志标记 real ONNArchSimulator
    路径也能构建的子集（dptc_crossbar 以及 Clements/Reck mesh）。
    """
    if not meta.get("evaluable"):
        return False
    if mode == "real":
        return bool(meta.get("real"))
    return True


def evaluable_template_names(mode: str = "real") -> List[str]:
    """当前仿真器在 ``mode`` 下实际可求值的模板名。"""
    return sorted(k for k, v in CORE_KINDS.items() if _is_evaluable(v, mode))


#: ``core_kind`` 候选的**声明顺序**，按拓扑族语义分组（crossbar → mesh → 权重阵列）。
#:
#: 顺序是语义化的，不是字典序：``step_hardware_parameter`` / ``ActionSpace`` 的
#: ±1 步进是**按索引**移动的，因此相邻即"最相近的另一种结构"。按族分组使一次
#: ±1 步进落在同一族内（mesh↔mesh、weight_array↔weight_array），而不是从网格直接
#: 跳到权重阵列。
#:
#: 每个条目都必须是**物理上互不相同**的电路拓扑：两个逐位给出相同网表与指标的
#: core_kind 不得同时出现在这里（``tests/test_topology_duplicates.py`` 会机械校验）。
#:
#: 这是 ``core_kind`` 候选顺序的**唯一**来源：搜索空间 JSON 不再声明候选，
#: 运行时由 :func:`default_core_kind_candidates` 注入本顺序。
CORE_KIND_SEARCH_ORDER: Tuple[str, ...] = (
    # crossbar 家族
    "dptc_crossbar",
    # MZI 网格家族
    "mzi_mesh_clements",
    "mzi_mesh_reck",
    "mzi_mesh_butterfly",
    # WDM 权重阵列家族
    "mrr_weight_bank",
    "pcm_crossbar",
    "wdm_link",
)

#: ``node_kind`` 候选的声明顺序（与上者同理由：±1 步进落在同类节点）。
NODE_KIND_SEARCH_ORDER: Tuple[str, ...] = (
    "dode",
    "tunable_mzi",
    "weight_mrr",
    "pcm_cell",
    "wdm_modulator",
)


def _ordered_candidates(
    registry: Dict[str, Dict[str, Any]],
    order: Tuple[str, ...],
    label: str,
) -> List[str]:
    """按显式顺序返回 ``evaluable`` 的种类，并校验顺序表与注册表一致。

    顺序表中缺失的 kind 追加在末尾（保持向后兼容），但会触发断言——新增拓扑时
    必须显式决定它在步进序列中的位置，否则 ±1 的邻接语义就是随机的。
    """
    evaluable = [k for k, v in registry.items() if v.get("evaluable")]
    ordered = [k for k in order if k in evaluable]
    missing = [k for k in evaluable if k not in order]
    if missing:
        raise AssertionError(
            f"{label}: {missing} are evaluable but missing from the search-order "
            f"table; add them so index-based stepping has a defined adjacency."
        )
    return ordered


def default_core_kind_candidates() -> List[str]:
    """默认搜索空间中暴露的 core 种类（与模式无关，均可求值）。

    这是 core 模板候选网格的单一事实来源：搜索空间 JSON **不再**重复声明这些值，
    ``photon_tuner.core.search_space.load_search_space`` 在加载时注入本函数的返回值。
    不可求值的范式不在此列。

    顺序按 :data:`CORE_KIND_SEARCH_ORDER` 的族分组排列，而非字典序——±1 步进按
    索引移动，顺序即邻接语义。
    """
    return _ordered_candidates(CORE_KINDS, CORE_KIND_SEARCH_ORDER, "core_kind")


def default_node_kind_candidates() -> List[str]:
    """默认搜索空间中暴露的 node 种类（与模式无关，均可求值）。

    顺序按 :data:`NODE_KIND_SEARCH_ORDER` 排列。候选同样由注册表注入搜索空间 JSON。
    """
    return _ordered_candidates(NODE_KINDS, NODE_KIND_SEARCH_ORDER, "node_kind")


def list_topology_templates(mode: str = "real") -> Dict[str, Any]:
    """完整模板注册表与种类耦合，供拓扑/方案 agent 使用。"""
    return {
        "core_kinds": CORE_KINDS,
        "node_kinds": NODE_KINDS,
        "coupling_matrix": {
            core: {"node_kinds": meta.get("node_kinds", [])}
            for core, meta in COUPLING_MATRIX.items()
        },
        "evaluable_templates": evaluable_template_names(mode),
        "legal_templates": legal_template_names(),
    }


def inspect_topology_template(name: str) -> Dict[str, Any]:
    """按名称查看某个模板详情：可为 ``core_kind`` 或 ``node_kind``。

    ``list_topology_templates`` 同时暴露 ``core_kinds`` 与 ``node_kinds``，
    因此查看工具需能应答两者。core 种类返回 core 元数据、兼容的 node 种类与
    所需器件槽位；node 种类返回 node 元数据及其兼容的 core 种类。
    """
    name = str(name)
    if name in CORE_KINDS:
        meta = dict(CORE_KINDS[name])
        meta["kind"] = "core"
        meta["template"] = name
        meta["compatible_node_kinds"] = list(COUPLING_MATRIX.get(name, {}).get("node_kinds", []))
        meta["device_slots"] = {
            node: REQUIRED_DEVICE_SLOTS.get(name, {}).get(node, [])
            for node in meta["compatible_node_kinds"]
        }
        return meta
    if name in NODE_KINDS:
        meta = dict(NODE_KINDS[name])
        meta["kind"] = "node"
        meta["template"] = name
        meta["compatible_core_kinds"] = [
            core
            for core, coupling in COUPLING_MATRIX.items()
            if name in coupling.get("node_kinds", [])
        ]
        return meta
    raise ValueError(
        f"unknown template {name!r}; legal core templates: {legal_template_names()}; "
        f"legal node kinds: {sorted(NODE_KINDS)}"
    )


def validate_coupling(
    hardware: Optional[Dict[str, Any]],
    design_space: Optional[Any] = None,
    mode: str = "real",
) -> List[Dict[str, Any]]:
    """返回硬件设计点的耦合违规列表。

    按顺序检查：
      1. ``core_kind`` 合法且在 ``mode`` 下可求值；
      2. ``node_kind`` 合法、可求值且与 core 兼容；
      3. (core, node) 组合所需的器件槽位齐备；
      4. 提供 :class:`DesignSpace` 时检查数值跨层约束（work_freq 链、精度）。

    ``mode`` 为 "analytic" 或 "real"：real 模式下 ``real`` 标志为 false 的种类会被
    拒绝，从而绝不产出 real ONNArchSimulator 无法构建的设计点。当前 8 个 core 与 5 个
    node 模板的 ``real`` 标志均为 true（全部经 ``.venv-simulator`` 端到端验证）。

    返回 ``{scope, message}`` 违规字典列表（空列表表示一致）。
    """
    violations: List[Dict[str, Any]] = []

    topo = (hardware or {}).get("arch", {}).get("topology", {}) or {}
    core_kind = str(topo.get("core_kind", "dptc_crossbar"))
    node_kind = str(topo.get("node_kind", "dode"))

    if core_kind not in CORE_KINDS:
        violations.append({
            "scope": "core_kind",
            "message": f"unknown core_kind {core_kind!r}; legal: {legal_template_names()}",
        })
    else:
        if not _is_evaluable(CORE_KINDS[core_kind], mode):
            violations.append({
                "scope": "core_kind",
                "message": f"core_kind {core_kind!r} is not evaluable in {mode} mode "
                f"(this kind has no real ONNArchSimulator netlist path)",
            })
        allowed = COUPLING_MATRIX.get(core_kind, {}).get("node_kinds", [])
        if node_kind not in allowed:
            violations.append({
                "scope": "node_kind",
                "message": f"node_kind {node_kind!r} is incompatible with core_kind {core_kind!r}; legal: {allowed}",
            })

    if node_kind not in NODE_KINDS:
        violations.append({
            "scope": "node_kind",
            "message": f"unknown node_kind {node_kind!r}; legal: {sorted(NODE_KINDS)}",
        })
    elif not _is_evaluable(NODE_KINDS[node_kind], mode):
        violations.append({
            "scope": "node_kind",
            "message": f"node_kind {node_kind!r} is not evaluable in {mode} mode "
            f"(this kind has no real ONNArchSimulator netlist path)",
        })

    # (core, node) 组合所需的器件槽位。
    slots = REQUIRED_DEVICE_SLOTS.get(core_kind, {}).get(node_kind, [])
    if slots:
        devices = (hardware or {}).get("devices", {}) or {}
        if core_kind.startswith("mzi_mesh_"):
            # mesh 范式使用独立的器件表（devices.mesh），
            # 因为 mesh 的 core/node 器件集与 crossbar 不同。
            mesh = devices.get("mesh", {}) or {}
            present = set(mesh.get("core", {})) | set(mesh.get("node", {}))
        elif core_kind in WEIGHT_ARRAY_CORE_KINDS:
            # WDM 权重阵列范式使用 devices.weight_array（微梳 + PD + 权重/调制器节点）。
            wa = devices.get("weight_array", {}) or {}
            present = set(wa.get("core", {})) | set(wa.get("node", {}))
        else:
            present = set(devices.get("core", {})) | set(devices.get("node", {}))
        missing = [s for s in slots if s not in present]
        if missing:
            violations.append({
                "scope": "device_slots",
                "message": f"template {core_kind}/{node_kind} is missing device slots: {sorted(missing)}",
            })

    # 数值跨层约束（work_freq <= 器件链、精度兼容）。
    if design_space is not None:
        for v in design_space.violations(hardware):
            violations.append({
                "scope": "numeric",
                "message": v.get("message", str(v)),
            })

    return violations


def _rate_ceiling(device_type: str, cfg: Dict[str, Any]) -> Optional[float]:
    """器件时钟上限，单位 GHz（对应 DesignSpace.min_work_freq_ceiling）。"""
    if device_type == "mzm":
        if "modulation_speed" in cfg:
            testing_bit = float(cfg.get("testing_bit", 1) or 1)
            return float(cfg["modulation_speed"]) / testing_bit
        return None
    if device_type in ("dac", "adc"):
        return cfg.get("sample_rate")
    if device_type == "photodetector":
        return cfg.get("bandwidth")
    return None


def _coupling_compatible(
    device_type: str,
    cfg: Dict[str, Any],
    work_freq: Any,
    in_bit: Any,
    out_bit: Any,
) -> bool:
    """若选择某变体后共享约束仍合法，则称该变体兼容。"""
    if work_freq is not None:
        limit = _rate_ceiling(device_type, cfg)
        if limit is not None and float(work_freq) > float(limit):
            return False
    if device_type == "dac" and "prec" in cfg and in_bit is not None:
        if int(in_bit) > int(cfg["prec"]):
            return False
    if device_type == "adc" and "prec" in cfg and out_bit is not None:
        if int(out_bit) > int(cfg["prec"]):
            return False
    return True


def coupling_compatible_variants(
    catalog: Dict[str, Dict[str, Any]],
    hardware: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """将器件目录过滤为与当前耦合兼容的变体。

    返回 ``{"compatible": {器件类型: [名称]}, "excluded": {器件类型: [名称]}}``。
    当选择某变体会违反 *当前* 设计点的共享约束（``work_freq`` 超过器件速率上限，
    或精度超过转换器分辨率）时，该变体被排除。无速率/精度字段的器件类型
    （如未报告带宽的 TIA）永不过滤——未报告即表示“无不相容证据”，而非伪造上限。
    """
    core = (hardware or {}).get("arch", {}).get("optical_core", {}) or {}
    work_freq = core.get("work_freq")
    precision = core.get("precision", {}) or {}
    in_bit = precision.get("in_bit")
    out_bit = precision.get("out_bit")

    compatible: Dict[str, List[str]] = {}
    excluded: Dict[str, List[str]] = {}
    for device_type, variants in (catalog or {}).items():
        keep: List[str] = []
        drop: List[str] = []
        for name, body in variants.items():
            cfg = (body or {}).get("cfgs", {}) or {} if isinstance(body, dict) else {}
            if _coupling_compatible(device_type, cfg, work_freq, in_bit, out_bit):
                keep.append(name)
            else:
                drop.append(name)
        compatible[device_type] = keep
        if drop:
            excluded[device_type] = drop
    return {"compatible": compatible, "excluded": excluded}
