"""面向离线解析光子评估器的命名可加载工作负载。

本模块将工作负载抽象为 **命名、可加载** 对象，使多 agent 循环能诊断真实模型。

支持的工作负载类型：

``literal``
    显式的 ``cycles`` / ``layer_sizes`` 表（由配置直接给出）。

``vit``
    参数化的 ViT / BERT 编码器，对应 ``simphony/models/quantized_BERT_base.py``。
    其逐层 cycle 元组用与 SimPhony 在 ``onnarchsim.workflow.dataflow.cycles``
    中 **相同的算术** 计算——此处以纯 Python 实现，无需 torch / torchonn / numpy。

由于 cycle 算术依赖架构（``num_wavelength``、``miniblock``、``dim_map``），
``vit`` 工作负载会响应硬件修改：改变波长数会改变 ``iter_D``，进而改变
agent 读取的逐层 cycle 划分。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_DIM_MAP",
    "DEFAULT_MINIBLOCK",
    "DEFAULT_WORKLOAD",
    "WorkloadError",
    "available_workloads",
    "build_workload",
    "load_workload_configs",
]

# --------------------------------------------------------------------------- #
# 架构上下文默认值（对应随附的 SimPhony 映射配置）。
# --------------------------------------------------------------------------- #

#: ``configs/architecture_mapping/simple_lt_*.yml`` 将 M/N/D 映射到核心。
DEFAULT_DIM_MAP: Dict[str, str] = {"M": "width", "N": "height", "D": "wavelength"}

#: ``configs/onn_mapping/simple_*.yml`` 对每一层使用相同的 miniblock。
DEFAULT_MINIBLOCK: List[int] = [2, 2, 4, 4]

DEFAULT_WORKLOAD = "vit_tiny"

CONFIG_RELPATH = Path("configs") / "workloads.json"


class WorkloadError(ValueError):
    """当工作负载名称或规格不可用时抛出。"""


def total_macs(partition_cycles: Mapping[str, Any]) -> int:
    """各层有效 MAC 总数 = ``N * D * M`` 之和。

    Cycles 是 SimPhony 元组 ``(iter_N, iter_D, iter_M, N, D, M,
    forward_factor_w, forward_factor_x)``，由 ``simu_partition_cycles`` 产生。
    有效运算数为下标 [3][4][5] 处的 ``N x D x M`` 矩阵乘/卷积规模
    （内置工作负载的 forward factor 均为 1）。
    """
    total = 0
    for cycle in partition_cycles.values():
        if isinstance(cycle, (list, tuple)) and len(cycle) >= 6:
            total += int(cycle[3]) * int(cycle[4]) * int(cycle[5])
    return total


# --------------------------------------------------------------------------- #
# 可移植的 cycle 算术（对应 onnarchsim/workflow/dataflow.py）。
# --------------------------------------------------------------------------- #

def _prod(values: Iterable[int]) -> int:
    return int(math.prod([int(v) for v in values]))


def _as_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise WorkloadError(f"expected a 2-element pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def im2col_shapes(
    op: str,
    weight_shape: Optional[Sequence[int]] = None,
    input_shape: Optional[Sequence[int]] = None,
    stride: Any = 1,
    padding: Any = 0,
    layer_shape: Optional[Sequence[Any]] = None,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """将一层重塑为 2-D 矩阵对，对应 ``im2col_2d_shape``。

    返回 ``(reshaped_input, reshaped_weight)``——与 SimPhony 构建 ``matrices``
    列表的顺序一致（``dataflow.generate_layer_sizes``），也是
    :func:`cycle_tuple` 期望的 ``matrices[0]`` / ``matrices[1]`` 顺序。
    输入为 ``(D, M)``，权重为 ``(N, D)``。
    """

    if op in ("Linear", "QLinear"):
        if weight_shape is None or input_shape is None:
            return None, None
        # Linear 权重为 (out_features, in_features) == (N, D)。
        out_features, in_features = _as_pair(weight_shape)
        dims = int(input_shape[-1])
        tokens = _prod(input_shape[:-1])
        return (dims, tokens), (out_features, in_features)

    if op in ("Conv", "Conv1d", "Conv2d", "Conv3d", "QConv2d"):
        if weight_shape is None or input_shape is None:
            return None, None
        n_filters, d_filter, *filter_size = [int(v) for v in weight_shape]
        n_x, d_x, *spatial = [int(v) for v in input_shape]
        strides = stride if isinstance(stride, (list, tuple)) else [stride] * len(filter_size)
        paddings = padding if isinstance(padding, (list, tuple)) else [padding] * len(filter_size)
        if len(strides) != len(filter_size):
            strides = [strides[0]] * len(filter_size)
        if len(paddings) != len(filter_size):
            paddings = [paddings[0]] * len(filter_size)
        out_size = [
            (size - f + 2 * p) // s + 1
            for size, f, s, p in zip(spatial, filter_size, strides, paddings)
        ]
        reshaped_input = (d_x * _prod(filter_size), n_x * _prod(out_size))
        reshaped_weight = (n_filters, _prod([d_filter] + filter_size))
        return reshaped_input, reshaped_weight

    if op in ("MatMul", "QMatMul"):
        if layer_shape is None:
            return None, None
        input_size_1 = [int(v) for v in layer_shape[0]]
        input_size_2 = [int(v) for v in layer_shape[2]]
        last_dim = input_size_1[-1]
        reshaped_input = (last_dim, _prod(input_size_1[:-1]))
        reshaped_weight = (_prod(input_size_2) // last_dim, last_dim)
        return reshaped_input, reshaped_weight

    raise WorkloadError(f"unsupported layer op {op!r}")


def loading_factor_calculator(
    weight_rep: str = "full",
    input_rep: str = "full",
    output_rep: str = "full",
) -> Tuple[int, int]:
    """``dataflow.loading_factor_calculator`` 的移植 -> (forward_factor_x, forward_factor_w)。"""
    reps = {"full", "complex", "positive"}
    for label, rep in (("weight", weight_rep), ("input", input_rep), ("output", output_rep)):
        if rep not in reps:
            raise WorkloadError(f"unsupported {label} representation {rep!r}")

    known = {
        ("positive", "positive", "full"): (2, 2),
        ("positive", "positive", "positive"): (1, 1),
        ("positive", "full", "full"): (1, 2),
        ("full", "positive", "full"): (2, 1),
        ("full", "full", "full"): (1, 1),
    }
    key = (weight_rep, input_rep, output_rep)
    if key not in known:
        raise WorkloadError(f"unsupported representation combination {key!r}")
    return known[key]


def calculate_iterations(
    miniblock: Sequence[int],
    dim_map: Dict[str, Optional[str]],
    multi_wavelength: int,
    D: int,
    N: int,
    M: int,
) -> Tuple[int, int, int, Dict[str, int], Dict[str, int]]:
    """``dataflow.calculate_iterations`` 的移植。"""
    if len(miniblock) != 4:
        raise WorkloadError(f"miniblock must have 4 entries, got {list(miniblock)!r}")
    arch_dims = {
        "width": int(miniblock[3]) * int(miniblock[1]),
        "height": int(miniblock[2]) * int(miniblock[0]),
        "wavelength": int(multi_wavelength),
    }
    core_dims = {
        "width": int(miniblock[3]),
        "height": int(miniblock[2]),
        "wavelength": int(multi_wavelength),
    }
    for key in dim_map:
        if key not in {"M", "N", "D"}:
            raise WorkloadError(f"invalid dim_map key {key!r}")
    if any(v is not None and v not in {"width", "height", "wavelength"} for v in dim_map.values()):
        raise WorkloadError(f"invalid dim_map value in {dim_map!r}")

    def _iters(size: int, key: str) -> int:
        target = dim_map.get(key)
        if target is None:
            return int(size)
        return math.ceil(int(size) / arch_dims[target])

    iter_D = _iters(D, "D")
    iter_N = _iters(N, "N")
    iter_M = _iters(M, "M")
    return iter_N, iter_M, iter_D, core_dims, arch_dims


def cycle_tuple(
    miniblock: Sequence[int],
    matrices: Tuple[Tuple[int, int], Tuple[int, int]],
    dim_map: Optional[Dict[str, Optional[str]]] = None,
    multi_wavelength: int = 1,
    weight_rep: str = "full",
    input_rep: str = "full",
    output_rep: str = "full",
) -> Tuple[int, int, int, int, int, int, int, int]:
    """``dataflow.cycles`` 的移植，返回报告所用的 8 元组。

    SimPhony 返回 9 元组，其最后一个元素是解析后的 ``arch_dims`` 字典；
    解析报告仅存储前八个整数。
    """
    dim_map = dict(dim_map or DEFAULT_DIM_MAP)
    reshaped_input, reshaped_weight = matrices
    N, D1 = reshaped_weight  # 权重形状 (N, D)
    D2, M = reshaped_input  # 输入形状 (D, M)
    if D1 != D2:
        raise WorkloadError(f"inconsistent matrix dimensions: weight D={D1}, input D={D2}")

    forward_factor_x, forward_factor_w = loading_factor_calculator(
        weight_rep=weight_rep, input_rep=input_rep, output_rep=output_rep
    )
    iter_N, iter_M, iter_D, _core_dims, _arch_dims = calculate_iterations(
        miniblock=miniblock,
        dim_map=dim_map,
        multi_wavelength=multi_wavelength,
        D=D1,
        N=N,
        M=M,
    )
    return (iter_N, iter_D, iter_M, N, D1, M, forward_factor_w, forward_factor_x)


# --------------------------------------------------------------------------- #
# 工作负载构建器。
# --------------------------------------------------------------------------- #

def _sizes_from_dense(op: str, weight_shape: Sequence[int], input_shape: Sequence[int],
                      stride: Any = 1, padding: Any = 0) -> Dict[str, Any]:
    """根据层形状推导的解析存储模型元素数。"""
    weight_elems = _prod(weight_shape)
    input_elems = _prod(input_shape)
    if op in ("Conv", "Conv1d", "Conv2d", "Conv3d", "QConv2d"):
        n_filters, _d_filter, *filter_size = [int(v) for v in weight_shape]
        n_x, _d_x, *spatial = [int(v) for v in input_shape]
        strides = stride if isinstance(stride, (list, tuple)) else [stride] * len(filter_size)
        paddings = padding if isinstance(padding, (list, tuple)) else [padding] * len(filter_size)
        if len(strides) != len(filter_size):
            strides = [strides[0]] * len(filter_size)
        if len(paddings) != len(filter_size):
            paddings = [paddings[0]] * len(filter_size)
        out_size = [
            (size - f + 2 * p) // s + 1
            for size, f, s, p in zip(spatial, filter_size, strides, paddings)
        ]
        output_elems = n_x * n_filters * _prod(out_size)
    else:
        out_features = int(weight_shape[0])
        tokens = _prod(input_shape[:-1])
        output_elems = out_features * tokens
    return {
        "input_size": [input_elems, 4],
        "output_size": [output_elems, 4],
        "weight_size": [weight_elems, 4],
        "total_size": input_elems + output_elems + weight_elems,
    }


def _vit_layers(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将 ViT/BERT 编码器规格展开为具体层描述符。"""
    embed = int(spec["embed_dim"])
    depth = int(spec["depth"])
    heads = int(spec["num_heads"])
    mlp_dim = int(spec.get("mlp_dim") or round(embed * float(spec.get("mlp_ratio", 4))))
    img = int(spec.get("img_size", 224))
    patch = int(spec.get("patch_size", 16))
    in_chans = int(spec.get("in_chans", 3))
    batch = int(spec.get("batch", 1))
    if embed % heads:
        raise WorkloadError("embed_dim must be divisible by num_heads")
    head_dim = embed // heads
    if img % patch:
        raise WorkloadError("img_size must be divisible by patch_size")
    grid = img // patch
    tokens = grid * grid + 1  # patch 令牌 + 分类令牌

    layers: List[Dict[str, Any]] = []

    # Patch 嵌入：Conv2d(in_chans -> embed, kernel=stride=patch)。
    layers.append({
        "name": "patch_embed",
        "op": "Conv2d",
        "weight_shape": [embed, in_chans, patch, patch],
        "input_shape": [batch, in_chans, img, img],
        "stride": patch,
        "padding": 0,
    })

    for index in range(depth):
        prefix = f"layers.{index}"
        # 融合 QKV 投影：Linear(embed -> 3*embed)。
        layers.append({
            "name": f"{prefix}.qkv",
            "op": "Linear",
            "weight_shape": [3 * embed, embed],
            "input_shape": [tokens, embed],
        })
        # 注意力分数 Q @ K^T：(heads, tokens, head_dim) x (heads, head_dim, tokens)。
        layers.append({
            "name": f"{prefix}.attn.qk",
            "op": "MatMul",
            "layer_shape": [
                [heads, tokens, head_dim],
                [heads, head_dim, tokens],
                [heads, head_dim, tokens],
            ],
        })
        # 注意力输出 scores @ V。
        layers.append({
            "name": f"{prefix}.attn.av",
            "op": "MatMul",
            "layer_shape": [
                [heads, tokens, tokens],
                [heads, tokens, tokens],
                [heads, tokens, head_dim],
            ],
        })
        # 输出投影：Linear(embed -> embed)。
        layers.append({
            "name": f"{prefix}.out_proj",
            "op": "Linear",
            "weight_shape": [embed, embed],
            "input_shape": [tokens, embed],
        })
        # 前馈网络。
        layers.append({
            "name": f"{prefix}.ffn.fc1",
            "op": "Linear",
            "weight_shape": [mlp_dim, embed],
            "input_shape": [tokens, embed],
        })
        layers.append({
            "name": f"{prefix}.ffn.fc2",
            "op": "Linear",
            "weight_shape": [embed, mlp_dim],
            "input_shape": [tokens, mlp_dim],
        })
    return layers


def _build_vit(spec: Dict[str, Any], arch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    miniblock = list(arch.get("miniblock") or DEFAULT_MINIBLOCK)
    dim_map = dict(arch.get("dim_map") or DEFAULT_DIM_MAP)
    wavelength = int(arch.get("num_wavelength", 1) or 1)
    weight_rep = str(arch.get("weight_representation", "full"))
    input_rep = str(arch.get("input_representation", "full"))
    output_rep = str(arch.get("output_representation", "full"))

    cycles: Dict[str, Any] = {}
    sizes: Dict[str, Any] = {}
    for layer in _vit_layers(spec):
        name = layer["name"]
        op = layer["op"]
        if op == "MatMul":
            matrices = im2col_shapes(op, layer_shape=layer["layer_shape"])
            weight_shape = layer["layer_shape"][0]
            input_shape = layer["layer_shape"][0]
        else:
            weight_shape = layer["weight_shape"]
            input_shape = layer["input_shape"]
            stride = layer.get("stride", 1)
            padding = layer.get("padding", 0)
            matrices = im2col_shapes(op, weight_shape=weight_shape, input_shape=input_shape,
                                     stride=stride, padding=padding)
        if matrices[0] is None or matrices[1] is None:
            raise WorkloadError(f"could not reshape layer {name!r} ({op})")
        cycles[name] = cycle_tuple(
            miniblock=miniblock,
            matrices=matrices,
            dim_map=dim_map,
            multi_wavelength=wavelength,
            weight_rep=weight_rep,
            input_rep=input_rep,
            output_rep=output_rep,
        )
        if op == "MatMul":
            matrices_x, matrices_w = matrices
            sizes[name] = {
                "input_size": [matrices_x[0] * matrices_x[1], 4],
                "output_size": [matrices_w[0] * matrices_x[1], 4],
                "weight_size": [matrices_w[0] * matrices_w[1], 4],
                "total_size": matrices_x[0] * matrices_x[1] + matrices_w[0] * matrices_w[1],
            }
        else:
            sizes[name] = _sizes_from_dense(op, weight_shape, input_shape,
                                            layer.get("stride", 1), layer.get("padding", 0))
    return cycles, sizes


_BUILDERS = {
    "literal": lambda spec, arch: (
        {str(k): tuple(int(v) for v in val) for k, val in (spec.get("cycles") or {}).items()},
        {str(k): dict(v) for k, v in (spec.get("layer_sizes") or {}).items()},
    ),
    "vit": _build_vit,
}


# --------------------------------------------------------------------------- #
# 配置加载。
# --------------------------------------------------------------------------- #

def load_workload_configs(
    simphony_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """加载 ``configs/workloads.json``（在若干合理根目录中搜索）。"""
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    if simphony_root:
        root = Path(simphony_root)
        candidates.append(root.parent / CONFIG_RELPATH)
        candidates.append(root / CONFIG_RELPATH)
    candidates.append(Path(__file__).resolve().parents[2] / CONFIG_RELPATH)
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle) or {}
    raise WorkloadError(
        "configs/workloads.json not found; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def available_workloads(
    simphony_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> List[str]:
    config = load_workload_configs(simphony_root=simphony_root, config_path=config_path)
    return sorted((config.get("workloads") or {}).keys())


def build_workload(
    name: Optional[str] = None,
    arch: Optional[Dict[str, Any]] = None,
    simphony_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """将工作负载名称解析为 ``{name, kind, cycles, layer_sizes, spec}``。

    ``arch`` 提供 cycle 算术所需的架构相关输入：``num_wavelength``、``miniblock``、
    ``dim_map`` 与操作数表示标志。
    """
    config = load_workload_configs(simphony_root=simphony_root, config_path=config_path)
    workloads = config.get("workloads") or {}
    resolved = name or config.get("default") or DEFAULT_WORKLOAD

    if resolved not in workloads:
        raise WorkloadError(
            f"unknown workload {resolved!r}; available: {sorted(workloads)}"
        )
    spec = dict(workloads[resolved] or {})
    kind = str(spec.get("kind", "literal"))
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise WorkloadError(f"workload {resolved!r} has unsupported kind {kind!r}")

    cycles, sizes = builder(spec, arch or {})
    if not cycles:
        raise WorkloadError(f"workload {resolved!r} produced no layers")
    return {
        "name": resolved,
        "kind": kind,
        "cycles": cycles,
        "layer_sizes": sizes,
        "spec": spec,
    }
