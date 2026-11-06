"""子架构（网表）装配：把搜索得到的 hardware 字典翻译成 SimPhony 网表。

本模块是**模式无关**的：解析模式（:mod:`photon_tuner.simulators.analytic`）与真实模式
（:mod:`photon_tuner.simulators.simphony` 的 ``_inject_hardware``）共用同一份装配逻辑，
以保证同一个设计点在两种引擎下得到*结构一致*的网表——两种模式的差异只应来自求值引擎，
而不应来自对网表的理解。

装配的输入与 SimPhony 完全相同的 YAML（``cores/dptc.yml``、``nodes/dode.yml``、
``sub_architectures/LT.yml`` 与器件库），再把搜索值覆盖上去，产出与 SimPhony 自身
解析结果同构的 ``sub_arch`` 字典。本模块不做任何数值计算。
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from photon_tuner.core.topology import (
    DEFAULT_LASER_DISTRIBUTION,
    DEFAULT_SHARING_POLICY,
    DEFAULT_TEMPORAL_ACCUM,
    apply_topology,
)
from photon_tuner.simulators.mesh_model import mesh_topology

# --------------------------------------------------------------------------- #
# 子架构命名。
# --------------------------------------------------------------------------- #

#: 架构模板（``LT_hetero.yml``）与层映射表（``architecture_mapping/*.yml``）里
#: 静态写死的子架构名。它**只是 YAML 查表用的键**，不代表任何拓扑——
#: ``dptc_crossbar``、MZI 网格与 WDM 权重阵列都复用它来定位同一个子架构槽位。
LEGACY_SUB_ARCH_NAME = "LT"

def sub_arch_key(hardware: Optional[Dict[str, Any]]) -> str:
    """返回该设计点下子架构字典/报告应使用的键名——即实际拓扑种类。

    搜索空间可以换 ``core_kind``，但 SimPhony 的层映射表是静态的，所有层都指向
    ``LEGACY_SUB_ARCH_NAME``。若沿用那个名字作报告键，report 会声称自己仿真的是
    "LT"（Hamerly crossbar 基线），实际却可能是 MZI 网格，诊断时极易误读。
    因此对外暴露的键统一取 ``core_kind`` 本身。
    """
    topo = ((hardware or {}).get("arch", {}) or {}).get("topology", {}) or {}
    return str(topo.get("core_kind") or "dptc_crossbar")

def first_key(mapping: Any) -> Optional[str]:
    """取报告分节里的子架构键（当前始终单子架构，故即唯一键）。

    历史报告以 ``"LT"`` 为键，新报告以 ``core_kind`` 为键；本函数对两者都成立，
    使读取方无需写死任何一个名字。
    """
    if not isinstance(mapping, dict):
        return None
    return next(iter(mapping), None)

# --------------------------------------------------------------------------- #
# 器件库加载。
# --------------------------------------------------------------------------- #

def load_device_catalog(devicelib_root: str) -> Dict[str, Dict[str, Any]]:
    """把所有器件 YAML 变体加载到 ``{device_type: {name: body}}`` 中。"""
    catalog: Dict[str, Dict[str, Any]] = {}
    import glob
    import os

    for path in sorted(glob.glob(os.path.join(devicelib_root, "*", "*.yml"))):
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for device_type, variants in data.items():
            bucket = catalog.setdefault(str(device_type), {})
            for name, body in variants.items():
                if isinstance(body, dict):
                    bucket[str(name)] = body
    return catalog


# --------------------------------------------------------------------------- #
# 子架构装配（架构参数 + 核心/节点网表 + 器件选择）。
# --------------------------------------------------------------------------- #

def _load_yaml_versioned(path: str, version: str = "v1") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    name = next(iter(data))
    versions = data[name]
    if version not in versions:
        raise ValueError(f"Version {version!r} not found in {path}")
    body = versions[version]
    body.setdefault("name", name)
    return body


def build_sub_arch(
    hardware: Dict[str, Any],
    simphony_root: str,
    device_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """将光子学 ``hardware`` 字典与 SimPhony 的核心/节点网表合并。

    返回的子架构 ``name`` 取该硬件实际使用的拓扑（``core_kind``），而非 ``LT.yml``
    里的静态名字——后者只是查表键，用作出现在报告里会误导诊断。
    """
    # SimPhony 的 CoreLib/NodeLib 会在仿真前把版本级的 ``devices`` 表移入
    # ``core`` / ``node`` 子字典，这里做同样的处理。
    dptc_body = _load_yaml_versioned(f"{simphony_root}/configs/design/cores/dptc.yml", "v1")
    dode_body = _load_yaml_versioned(f"{simphony_root}/configs/design/nodes/dode.yml", "v1")
    core_cfg = dict(dptc_body["core"])
    core_cfg["devices"] = dict(dptc_body["devices"])
    core_cfg.setdefault("name", dptc_body["name"])
    core_cfg.setdefault("version", "v1")
    node_cfg = dict(dode_body["node"])
    node_cfg["devices"] = dict(dode_body["devices"])
    node_cfg.setdefault("name", dode_body["name"])
    node_cfg.setdefault("version", "v1")

    arch = hardware.get("arch", {})
    system = arch.get("system", {})
    core = arch.get("optical_core", {})
    topology = arch.get("topology", {})
    precision = core.get("precision", {})
    # 从 LT 子架构的内存默认值出发（其中带有真实工作流所需的 CACTI
    # ``config_file_name`` / ``output_file_name``），再叠加搜索得到的缓存/带宽值。
    lt_body = _load_yaml_versioned(
        f"{simphony_root}/configs/design/sub_architectures/LT.yml", "v1"
    )
    memory = dict(lt_body.get("memory", {}) or {})
    for level, cfg in (arch.get("memory") or {}).items():
        if isinstance(cfg, dict):
            memory.setdefault(level, {}).update(cfg)

    core_devices = dict(hardware.get("devices", {}).get("core", {}))
    node_devices = dict(hardware.get("devices", {}).get("node", {}))

    # 把器件选择覆盖应用到网表器件表。
    for device_type, device_name in core_devices.items():
        if device_type in core_cfg.get("devices", {}):
            core_cfg["devices"][device_type] = [device_name]
    for device_type, device_name in node_devices.items():
        if device_type in node_cfg.get("devices", {}):
            node_cfg["devices"][device_type] = [device_name]

    # 实例级器件类型开关（可搜索的“实例选择”）。``laser_source`` 在片上与片外光源
    # 之间切换激光实例 i4；``splitter_kind`` 在波导激光分束器与 1x10 MMI 之间切换
    # 扇出分束器 i6。实例 id、缩放规则与网络位置保持不变，因此该替换是结构性的。
    # 真实 ONNArchSimulator 的面积工作流通过 ``netlist.ports_num``（cal_MMI_area）
    # 对含 “mmi” 的器件名做特殊处理，所以 mmi 场景也必须填充该字典。
    netlist = core_cfg.setdefault("netlist", {})
    instances = netlist.setdefault("instances", {})
    laser_source = str(core_devices.get("laser_source", "on_chip"))
    splitter_kind = str(core_devices.get("splitter_kind", "laser_splitter"))
    i4 = instances.get("i4")
    if i4 and i4[0] in ("on_chip_laser", "off_chip_laser"):
        instances["i4"] = [
            "off_chip_laser" if laser_source == "off_chip" else "on_chip_laser",
            0,
        ]
    i6 = instances.get("i6")
    if i6 and i6[0] in ("laser_splitter", "mmi"):
        instances["i6"] = ["mmi" if splitter_kind == "mmi" else "laser_splitter", 0]
    if splitter_kind == "mmi":
        ports = netlist.get("ports_num")
        if not isinstance(ports, dict):
            ports = {}
            netlist["ports_num"] = ports
        ports["i6"] = 10  # 与 Customized_1x10_MMI 的 cfgs.ports_num 一致

    core_cfg["height"] = int(core.get("height", core_cfg.get("height", 12)))
    core_cfg["width"] = int(core.get("width", core_cfg.get("width", 12)))
    core_cfg["num_wavelength"] = int(
        core.get("num_wavelength", core_cfg.get("num_wavelength", 12))
    )
    core_cfg["work_freq"] = float(core.get("work_freq", core_cfg.get("work_freq", 5)))
    core_cfg["precision"] = {
        "in_bit": int(precision.get("in_bit", 4)),
        "w_bit": int(precision.get("w_bit", 4)),
        "out_bit": int(precision.get("out_bit", 4)),
    }
    core_cfg["node"] = node_cfg

    # L4 拓扑/电路覆盖（共享策略、时间累积、激光功率分配）作为受控值映射到核心网表。
    apply_topology(
        core_cfg.setdefault("netlist", {}),
        str(topology.get("sharing_policy", DEFAULT_SHARING_POLICY)),
        str(topology.get("temporal_accum", DEFAULT_TEMPORAL_ACCUM)),
        str(topology.get("laser_distribution", DEFAULT_LASER_DISTRIBUTION)),
    )

    sub_arch = {
        "name": sub_arch_key(hardware),
        "version": "v1",
        "tiles": int(system.get("tiles", 4)),
        "cores_per_tile": int(system.get("cores_per_tile", 2)),
        "dataflow": str(system.get("dataflow", "output_stationary")),
        "op_type": str(system.get("op_type", "MVM")),
        "core": core_cfg,
        "memory": memory,
    }
    return sub_arch


# --------------------------------------------------------------------------- #
# MZI 网格子架构（网格范式的解析镜像）。
# --------------------------------------------------------------------------- #

def _tunable_mzi_node(n_mzi: int, coupler: str, phase_shifter: str) -> Dict[str, Any]:
    """程序化构造的可调 MZI 节点网表（2 个耦合器 + 2 个移相器）。

    每个 2x2 MZI 分解为两个 50:50 定向耦合器加两个移相器（内部 theta、外部 phi）。
    ``n_mzi`` 以整数字面量乘以 ``R*C`` 的形式写入每个实例的缩放规则，使节点的器件
    数量复现已发表的 MZI 数量。``coupler`` / ``phase_shifter`` 为已有的库变体；
    这里不捏造任何新的器件参数。
    """
    netlist = {
        "instances": {
            "i0": ["coupler", 0],
            "i1": ["phase_shifter", 0],
            "i2": ["coupler", 0],
            "i3": ["phase_shifter", 0],
        },
        "scaling_rules": {
            "i0": f"{n_mzi}*R*C",
            "i1": f"{n_mzi}*R*C",
            "i2": f"{n_mzi}*R*C",
            "i3": f"{n_mzi}*R*C",
        },
        "nets": {
            "n0": ["i0", "i1"],
            "n1": ["i1", "i2"],
            "n2": ["i2", "i3"],
        },
        "replication": {},
        "temporal_accum_factor": {},
        "ports_num": {},
        "operand_1": None,
        "operand_2": None,
        "insertion_loss_final_device": "phase_shifter",
        "chip_mapping": {},
    }
    return {
        "netlist": netlist,
        "devices": {"coupler": [coupler], "phase_shifter": [phase_shifter]},
        # MZI 网格权重静态保持：移相器 *持续* 耗散 P_pi（热光保持），而非按运算计
        # （不同于 DODE 节点动态调谐的相位微调器）。initialize_device_power 读取此
        # 标志，把 P_pi 归入 static_power 而非 dynamic_power。
        "weight_stationary_phase_shifter": True,
        "device_spacing_flag": True,
        "node_spacing_flag": True,
        "device_spacing": [5, 5, 5, 5],
        "node_spacing": [5, 5, 5, 5],
        "longest_path": None,
        "bounding_box_area": None,
        "bounding_box_estimator": None,
        "device_counter_function": None,
    }


def build_mesh_sub_arch(
    hardware: Dict[str, Any],
    simphony_root: str,
    catalog: Dict[str, Any],
    kind: str,
) -> Dict[str, Any]:
    """以与 :func:`build_sub_arch` 相同的结构构建网格子架构。

    仅核心/节点网表与器件集合不同；内存层次、数据流以及所有下游工作流（能耗/面积/
    插入损耗）均共享。``kind`` 为 ``mzi_mesh_clements`` / ``mzi_mesh_reck`` /
    ``mzi_mesh_butterfly``；网格布局通过去掉前缀得到。
    """
    arch = hardware.get("arch", {})
    system = arch.get("system", {})
    core = arch.get("optical_core", {})
    precision = core.get("precision", {}) or {}
    topology = arch.get("topology", {}) or {}

    layout = str(kind).replace("mzi_mesh_", "")
    # 酉矩阵秩 N：优先使用拓扑层 ``mesh_rank``；否则回退到 optical_core.height/width，
    # 以保证不含 ``mesh_rank`` 的硬件字典行为不变。
    mesh_rank = topology.get("mesh_rank")
    if mesh_rank is not None:
        n_modes = int(mesh_rank)
    else:
        n_modes = int(core.get("height", core.get("width", 4)))
    if n_modes < 2:
        n_modes = 2
    topo = mesh_topology(layout, n_modes)

    # 节点/核心器件选择来自网格器件表（hardware["devices"]["mesh"]），该表由
    # default_hardware 用来源明确的变体填充；缺失时回退到相同的来源变体。
    mesh_devices = dict(hardware.get("devices", {}).get("mesh", {}) or {})
    mesh_core = dict(mesh_devices.get("core", {}) or {})
    mesh_node = dict(mesh_devices.get("node", {}) or {})
    coupler = str(mesh_node.get("coupler", "LT_DC_2x2"))
    phase_shifter = str(mesh_node.get("phase_shifter", "IHO_PS"))
    node_cfg = _tunable_mzi_node(topo["n_mzi"], coupler, phase_shifter)

    core_cfg: Dict[str, Any] = {
        "name": kind,
        "version": "v1",
        "type": "MZI_mesh",
        "coherent": True,
        "universal": topo["universal"],
        "forward": "direct",
        "ref": topo["ref"],
        "range": {"weight": "full", "input": "full", "output": "full"},
        # 对于网格核心，“height” 表示 MZI 数量而 “width” 为 1，因此共享节点包围盒
        # 计数（height*width*R*C）等于 n_mzi*R*C。
        "height": int(topo["n_mzi"]),
        "width": 1,
        # 酉变换为 n_modes x n_modes；真实模拟器把工作负载矩阵平铺到此维度，而非
        # MZI 数量的包围盒。
        "matrix_height": n_modes,
        "matrix_width": n_modes,
        "num_wavelength": 1,
        "range_forwards": int(topo["range_forwards"]),
        # 论文重配置模型：TO 热调谐 ~10 µs/次；权重数 = 网格 MZI 总数。
        # cycle penalty = weight_count × reconfig_time_ns × work_freq（GHz）。
        "weight_count": int(topo["n_mzi"]),
        "reconfig_time_ns": 10000.0,
        "work_freq": float(core.get("work_freq", 5)),
        "precision": {
            "in_bit": int(precision.get("in_bit", 4)),
            "w_bit": int(precision.get("w_bit", 4)),
            "out_bit": int(precision.get("out_bit", 4)),
        },
        "devices": {
            "off_chip_laser": [str(mesh_core.get("off_chip_laser", "LT_Laser"))],
            "photodetector": [str(mesh_core.get("photodetector", "LT_PD"))],
            "adc": [str(mesh_core.get("adc", "ADC_SAR_1"))],
            "tia": [str(mesh_core.get("tia", "TIA_1"))],
            "dac": [str(mesh_core.get("dac", "DAC_2"))],
            "waveguide": [str(mesh_core.get("waveguide", "WG_Single_Strip"))],
        },
        "netlist": {
            "instances": {
                "i_laser": ["off_chip_laser", 0],
                "i_mzi": ["node", 0],
                "i_pd": ["photodetector", 0],
                "i_tia": ["tia", 0],
                "i_adc": ["adc", 0],
                "i_dac": ["dac", 0],
            },
            "scaling_rules": {
                "i_laser": "R+C",
                "i_mzi": "R*C",
                "i_pd": f"{n_modes}*R*C",
                "i_tia": f"{n_modes}*R*C",
                "i_adc": f"{n_modes}*R*C",
                "i_dac": f"{n_modes}*R*C",
            },
            "nets": {
                "n0": ["i_laser", "i_mzi(s)"],
                "n1": ["i_mzi(e)", "i_pd"],
                "n2": ["i_pd", "i_tia"],
                "n3": ["i_tia", "i_adc"],
                "n4": ["i_dac", "i_laser"],
            },
            "replication": {"i_mzi": int(topo["depth"])},
            # 与真实工作流的交叉开关对齐：内存工作流从网格核心的 temporal_accum_factor
            # 读取 `duration` 与 `memory`（裸 {} 会触发 KeyError）。网格在 DAC/ADC 对中
            # 累积，与 DPTC 交叉开关一致。
            "temporal_accum_factor": {
                "duration": 3,
                "devices": ["i_dac", "i_adc"],
                "memory": "output",
            },
            "ports_num": {},
            "operand_1": ["i_dac"],
            "operand_2": ["i_laser"],
            "laser_power_distribution": 1,
            "chip_mapping": {
                "RF_EIC": ["i_dac", "i_adc", "i_tia"],
                "PIC_1": ["i_mzi", "i_pd"],
                "PIC_2": ["i_laser"],
            },
            "insertion_loss_final_device": "photodetector",
        },
        "node": node_cfg,
        "device_spacing_flag": None,
        "node_spacing_flag": None,
        "device_spacing": None,
        "node_spacing": None,
        "bounding_box_estimator": None,
        "device_counter_function": None,
    }

    # crossing-bearing 布局（Reck / butterfly）：把波导交叉计入关键路径插损与面积。
    # Clements 无交叉（crossing_free=True），不注入 crossing 实例，保持公平对比。
    if not topo["crossing_free"]:
        n_cross = int(topo["n_crossing"])
        core_cfg["devices"]["crossing"] = [str(mesh_core.get("crossing", "WG_Crossing_Parabolic"))]
        nl = core_cfg["netlist"]
        nl["instances"]["i_cross"] = ["crossing", 0]
        # 面积/数量按总交叉数缩放；插损链按深度复制（每个深度层都遇交叉）。
        nl["scaling_rules"]["i_cross"] = f"{n_cross}*R*C"
        nl["replication"]["i_cross"] = int(topo["depth"])
        nl["nets"]["n1"] = ["i_mzi(e)", "i_cross(s)"]
        nl["nets"]["n1b"] = ["i_cross(e)", "i_pd"]
        nl["chip_mapping"]["PIC_1"].append("i_cross")

    # L4 电路级控制项（共享策略 / 时间累积 / 激光分配）也必须写入网格核心网表，
    # 否则网格会静默忽略所有 L4 拓扑改动。
    apply_topology(
        core_cfg.setdefault("netlist", {}),
        str(topology.get("sharing_policy", DEFAULT_SHARING_POLICY)),
        str(topology.get("temporal_accum", DEFAULT_TEMPORAL_ACCUM)),
        str(topology.get("laser_distribution", DEFAULT_LASER_DISTRIBUTION)),
    )

    lt_body = _load_yaml_versioned(
        f"{simphony_root}/configs/design/sub_architectures/LT.yml", "v1"
    )
    memory = dict(lt_body.get("memory", {}) or {})
    for level, cfg in (arch.get("memory") or {}).items():
        if isinstance(cfg, dict):
            memory.setdefault(level, {}).update(cfg)

    return {
        "name": kind,
        "version": "v1",
        "tiles": int(system.get("tiles", 4)),
        "cores_per_tile": int(system.get("cores_per_tile", 2)),
        # 静态 MZI 网格是 weight-stationary PTC（热调谐 µs–ms 级、无法动态换权重），
        # 论文将其限定在 weight-stationary 数据流；故忽略 system.dataflow。
        "dataflow": "weight_stationary",
        "op_type": str(system.get("op_type", "MVM")),
        "core": core_cfg,
        "memory": memory,
    }


# --------------------------------------------------------------------------- #
# WDM 权重阵列子架构（MRR 权重组 / PCM 交叉开关 / 单 WDM 链路）。
# --------------------------------------------------------------------------- #

_WEIGHT_ARRAY_SPECS: Dict[str, Dict[str, Any]] = {
    "mrr_weight_bank": {
        "node_device": "weight_mrr",
        "node_default": "WG_MRR_WEIGHT_TAIT",
        "range_forwards": 2,
        "reconfig_time_ns": 10000.0,
        "coherent": False,
        "ref": "https://doi.org/10.1038/s41598-017-07754-z",
        "type": "MRR_weight_bank",
        "operand_1": ["i_dac"],
        "operand_2": ["i_laser"],
        # 权重 R+（微环 drop 只能正），差分检测实现全值域 → I=2。
        "range": {"weight": "positive", "input": "full", "output": "full"},
    },
    "pcm_crossbar": {
        "node_device": "pcm_cell",
        "node_default": "PCM_CELL_FELDMANN",
        "range_forwards": 4,
        "reconfig_time_ns": 100.0,
        "coherent": False,
        "ref": "https://doi.org/10.1038/s41586-020-03070-1",
        "type": "PCM_crossbar",
        "operand_1": ["i_dac"],
        "operand_2": ["i_laser"],
        # 权重与输入均 R+（GST 吸收态 0-1），四段分解实现全值域 → I=4。
        "range": {"weight": "positive", "input": "positive", "output": "full"},
    },
    "wdm_link": {
        "node_device": "mzm",
        "node_default": "LT_MZM",
        "range_forwards": 1,
        "reconfig_time_ns": 0.0,
        "coherent": True,
        "ref": "https://doi.org/10.1038/s41586-020-03063-0",
        "type": "WDM_link",
        "operand_1": ["i_comb"],
        "operand_2": ["i_dac"],
        # 相干 WDM：权重（微梳相位/强度）与输入（宽带调制）均可全值域 → I=1。
        "range": {"weight": "full", "input": "full", "output": "full"},
    },
}


def _weight_element_node(device_type: str, device_name: str, n_weight: int) -> Dict[str, Any]:
    """单器件权重/调制节点（MRR 权重组 / PCM 单元 / WDM 调制器）。

    权重阵列是 *并行* 结构：光一次通过一个单元，节点网表只含单个器件实例、
    无内部连线；``n_weight`` 通过缩放规则计入器件数量（面积/能耗），而不进入
    插损关键路径（与串行的 MZI 网格不同）。
    """
    netlist = {
        "instances": {"i0": [device_type, 0]},
        "scaling_rules": {"i0": f"{n_weight}*R*C"},
        "nets": {},
        "replication": {},
        "temporal_accum_factor": {},
        "ports_num": {},
        "operand_1": None,
        "operand_2": None,
        "insertion_loss_final_device": device_type,
        "chip_mapping": {},
    }
    return {
        "netlist": netlist,
        "devices": {device_type: [device_name]},
        # 权重静态保持（热调 MRR / 非易失 PCM）；该标志只影响 phase_shifter，置真以保持语义一致。
        "weight_stationary_phase_shifter": True,
        "device_spacing_flag": True,
        "node_spacing_flag": True,
        "device_spacing": [5, 5, 5, 5],
        "node_spacing": [5, 5, 5, 5],
        "longest_path": None,
        "bounding_box_area": None,
        "bounding_box_estimator": None,
        "device_counter_function": None,
    }


def build_weight_array_sub_arch(
    hardware: Dict[str, Any],
    simphony_root: str,
    catalog: Dict[str, Any],
    kind: str,
) -> Dict[str, Any]:
    """以与 :func:`build_mesh_sub_arch` 相同的结构构建 WDM 权重阵列子架构。

    覆盖三类论文拓扑，统一为“泵浦激光器 → 微梳 → 并行权重阵列 → PD → TIA → ADC”：
      * ``mrr_weight_bank`` — Tait 2017 硅光权重组（微环 drop-port 权重，I=2，TO 10µs）；
      * ``pcm_crossbar``    — Feldmann 2021 PCM 张量核（GST 吸收权重，I=4，写入 100ns）；
      * ``wdm_link``        — Xu 2021 单 WDM 链路（微梳谱线权重 × 宽带调制器输入，I=1）。
    """
    spec = _WEIGHT_ARRAY_SPECS[kind]
    arch = hardware.get("arch", {})
    system = arch.get("system", {})
    core = arch.get("optical_core", {})
    precision = core.get("precision", {}) or {}
    topology = arch.get("topology", {}) or {}

    mesh_rank = topology.get("mesh_rank")
    if mesh_rank is not None:
        n_weight = int(mesh_rank)
    else:
        n_weight = int(core.get("height", core.get("width", 8)))
    if n_weight < 2:
        n_weight = 2

    wa = dict(hardware.get("devices", {}).get("weight_array", {}) or {})
    wa_core = dict(wa.get("core", {}) or {})
    wa_node = dict(wa.get("node", {}) or {})
    node_device = spec["node_device"]
    node_name = str(wa_node.get(node_device, spec["node_default"]))
    node_cfg = _weight_element_node(node_device, node_name, n_weight)

    core_cfg: Dict[str, Any] = {
        "name": kind,
        "version": "v1",
        "type": spec["type"],
        "coherent": spec["coherent"],
        "universal": True,
        "forward": "direct",
        "ref": spec["ref"],
        "range": dict(spec["range"]),
        # “height” 表示并行权重数而 “width” 为 1（与 mesh 的 height=n_mzi 一致）；
        # 节点缩放规则 n_weight*R*C 使器件数量等于并行权重数。
        "height": int(n_weight),
        "width": 1,
        "matrix_height": 1,
        "matrix_width": int(n_weight),
        "num_wavelength": int(n_weight),
        "range_forwards": int(spec["range_forwards"]),
        "weight_count": int(n_weight),
        "reconfig_time_ns": float(spec["reconfig_time_ns"]),
        "work_freq": float(core.get("work_freq", 5)),
        "precision": {
            "in_bit": int(precision.get("in_bit", 4)),
            "w_bit": int(precision.get("w_bit", 4)),
            "out_bit": int(precision.get("out_bit", 4)),
        },
        "devices": {
            "off_chip_laser": [str(wa_core.get("off_chip_laser", "LT_Laser"))],
            "micro_comb": [str(wa_core.get("micro_comb", "MC_Si3N4_DKS_100GHz"))],
            "photodetector": [str(wa_core.get("photodetector", "LT_PD"))],
            "adc": [str(wa_core.get("adc", "ADC_SAR_1"))],
            "tia": [str(wa_core.get("tia", "TIA_1"))],
            "dac": [str(wa_core.get("dac", "DAC_2"))],
            "waveguide": [str(wa_core.get("waveguide", "WG_Single_Strip"))],
        },
        "netlist": {
            "instances": {
                "i_laser": ["off_chip_laser", 0],
                "i_comb": ["micro_comb", 0],
                "i_weight": ["node", 0],
                "i_pd": ["photodetector", 0],
                "i_tia": ["tia", 0],
                "i_adc": ["adc", 0],
                "i_dac": ["dac", 0],
            },
            "scaling_rules": {
                "i_laser": "R*C",
                "i_comb": "R*C",
                "i_weight": "R*C",
                "i_pd": "R*C",
                "i_tia": "R*C",
                "i_adc": "R*C",
                "i_dac": f"{n_weight}*R*C",
            },
            "nets": {
                "n0": ["i_laser", "i_comb"],
                "n1": ["i_comb", "i_weight(s)"],
                "n2": ["i_weight(e)", "i_pd"],
                "n3": ["i_pd", "i_tia"],
                "n4": ["i_tia", "i_adc"],
                "n5": ["i_dac", "i_weight(s)"],
            },
            "replication": {},
            "temporal_accum_factor": {
                "duration": 3,
                "devices": ["i_dac", "i_adc"],
                "memory": "output",
            },
            "ports_num": {},
            "operand_1": list(spec["operand_1"]),
            "operand_2": list(spec["operand_2"]),
            "laser_power_distribution": 1,
            "chip_mapping": {
                "RF_EIC": ["i_dac", "i_adc", "i_tia"],
                "PIC_1": ["i_weight", "i_pd"],
                "PIC_2": ["i_laser", "i_comb"],
            },
            "insertion_loss_final_device": "photodetector",
        },
        "node": node_cfg,
        "device_spacing_flag": None,
        "node_spacing_flag": None,
        "device_spacing": None,
        "node_spacing": None,
        "bounding_box_estimator": None,
        "device_counter_function": None,
    }

    apply_topology(
        core_cfg.setdefault("netlist", {}),
        str(topology.get("sharing_policy", DEFAULT_SHARING_POLICY)),
        str(topology.get("temporal_accum", DEFAULT_TEMPORAL_ACCUM)),
        str(topology.get("laser_distribution", DEFAULT_LASER_DISTRIBUTION)),
    )
    # 权重阵列是单输出点积（1 个 PD/TIA/ADC 汇总所有波长），ADC/TIA 共享策略不适用；
    # 撤销 apply_sharing_policy 对 i_adc/i_tia 的 H*W 缩放覆盖。
    core_cfg["netlist"]["scaling_rules"]["i_adc"] = "R*C"
    core_cfg["netlist"]["scaling_rules"]["i_tia"] = "R*C"

    lt_body = _load_yaml_versioned(
        f"{simphony_root}/configs/design/sub_architectures/LT.yml", "v1"
    )
    memory = dict(lt_body.get("memory", {}) or {})
    for level, cfg in (arch.get("memory") or {}).items():
        if isinstance(cfg, dict):
            memory.setdefault(level, {}).update(cfg)

    return {
        "name": kind,
        "version": "v1",
        "tiles": int(system.get("tiles", 4)),
        "cores_per_tile": int(system.get("cores_per_tile", 2)),
        # 权重阵列是 weight-stationary PTC（权重由 MRR/PCM 静态保持或微梳静态编码）。
        "dataflow": "weight_stationary",
        "op_type": str(system.get("op_type", "MVM")),
        "core": core_cfg,
        "memory": memory,
    }

