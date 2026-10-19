"""Named, loadable workloads for the offline analytic photonic evaluator.

The analytic mirror originally evaluated one hand-written three-layer workload
(``synthetic_workload``).  That made every per-layer / per-operator-group signal
constant, because the *demand* side of the diagnosis never changed.  This module
generalises the workload to a **named, loadable** object so the multi-agent loop
can diagnose a realistic model instead.

Two workload kinds are supported:

``literal``
    Explicit ``cycles`` / ``layer_sizes`` tables.  Used by the legacy
    ``synthetic`` baseline so the existing reference numbers (and tests) are
    untouched.

``vit``
    A parameterised ViT / BERT encoder matching
    ``simphony/models/quantized_BERT_base.py``.  Its per-layer cycle tuples are
    computed with the **same arithmetic** SimPhony uses in
    ``onnarchsim.workflow.dataflow.cycles`` -- ported here as pure Python so it
    runs without torch / torchonn / numpy.

Because the cycle arithmetic depends on the architecture
(``num_wavelength``, ``miniblock``, ``dim_map``), a ``vit`` workload responds to
hardware edits: changing the wavelength count moves ``iter_D`` and therefore the
per-layer cycle split the agents read.
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
    "synthetic_cycles",
    "synthetic_layer_sizes",
]

# --------------------------------------------------------------------------- #
# Architecture context defaults (mirror the shipped SimPhony mapping configs).
# --------------------------------------------------------------------------- #

#: ``configs/architecture_mapping/simple_lt_*.yml`` maps M/N/D onto the core.
DEFAULT_DIM_MAP: Dict[str, str] = {"M": "width", "N": "height", "D": "wavelength"}

#: ``configs/onn_mapping/simple_*.yml`` uses the same miniblock for every layer.
DEFAULT_MINIBLOCK: List[int] = [2, 2, 4, 4]

DEFAULT_WORKLOAD = "synthetic"

CONFIG_RELPATH = Path("configs") / "workloads.json"


class WorkloadError(ValueError):
    """Raised when a workload name or spec is unusable."""


# --------------------------------------------------------------------------- #
# Legacy literal baseline (kept byte-identical to the original constants).
# --------------------------------------------------------------------------- #

def synthetic_cycles() -> Dict[str, Tuple[int, int, int, int, int, int, int, int]]:
    """The original hand-written three-layer workload.

    Tuples follow ``(iter_N, iter_D, iter_M, N, D, M, forward_factor_w,
    forward_factor_x)`` produced by SimPhony's ``simu_partition_cycles``.
    """
    return {
        "conv1": (1, 3, 75, 8, 27, 1800, 1, 1),
        "conv2": (1, 6, 66, 8, 72, 1568, 1, 1),
        "linear": (1, 17, 1, 10, 200, 2, 1, 1),
    }


def synthetic_layer_sizes() -> Dict[str, Dict[str, Any]]:
    """Approximate layer I/O sizes (element counts) for the analytic memory model."""
    return {
        "conv1": {"input_size": [1536, 4], "output_size": [288, 4], "weight_size": [72, 4], "total_size": 1896},
        "conv2": {"input_size": [288, 4], "output_size": [256, 4], "weight_size": [144, 4], "total_size": 688},
        "linear": {"input_size": [200, 4], "output_size": [2, 4], "weight_size": [400, 4], "total_size": 602},
    }


def total_macs(partition_cycles: Mapping[str, Any]) -> int:
    """Total useful MACs across layers = sum of ``N * D * M``.

    Cycles are SimPhony tuples ``(iter_N, iter_D, iter_M, N, D, M,
    forward_factor_w, forward_factor_x)`` produced by ``simu_partition_cycles``.
    The useful operation count is the ``N x D x M`` matmul/conv volume at
    indices [3][4][5] (the forward factors are 1 for the built-in workloads).
    """
    total = 0
    for cycle in partition_cycles.values():
        if isinstance(cycle, (list, tuple)) and len(cycle) >= 6:
            total += int(cycle[3]) * int(cycle[4]) * int(cycle[5])
    return total


# --------------------------------------------------------------------------- #
# Portable cycle arithmetic (ported from onnarchsim/workflow/dataflow.py).
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
    """Reshape a layer into a 2-D matrix pair, mirroring ``im2col_2d_shape``.

    Returns ``(reshaped_input, reshaped_weight)`` -- the same ordering SimPhony
    builds its ``matrices`` list in (``dataflow.generate_layer_sizes``) and the
    ordering :func:`cycle_tuple` expects as ``matrices[0]`` / ``matrices[1]``.
    The input is ``(D, M)`` and the weight is ``(N, D)``.
    """

    if op in ("Linear", "QLinear"):
        if weight_shape is None or input_shape is None:
            return None, None
        # A Linear weight is (out_features, in_features) == (N, D).
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
    """Port of ``dataflow.loading_factor_calculator`` -> (forward_factor_x, forward_factor_w)."""
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
    """Port of ``dataflow.calculate_iterations``."""
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
    """Port of ``dataflow.cycles`` returning the 8-tuple the report uses.

    SimPhony returns a 9-tuple whose last element is the resolved ``arch_dims``
    dict; the analytic report only stores the eight integers.
    """
    dim_map = dict(dim_map or DEFAULT_DIM_MAP)
    reshaped_input, reshaped_weight = matrices
    N, D1 = reshaped_weight  # weight shape (N, D)
    D2, M = reshaped_input  # input shape (D, M)
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
# Workload builders.
# --------------------------------------------------------------------------- #

def _sizes_from_dense(op: str, weight_shape: Sequence[int], input_shape: Sequence[int],
                      stride: Any = 1, padding: Any = 0) -> Dict[str, Any]:
    """Element counts for the analytic memory model, derived from layer shapes."""
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
    """Expand a ViT/BERT encoder spec into concrete layer descriptors."""
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
    tokens = grid * grid + 1  # patch tokens + classification token

    layers: List[Dict[str, Any]] = []

    # Patch embedding: Conv2d(in_chans -> embed, kernel=stride=patch).
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
        # Fused QKV projection: Linear(embed -> 3*embed).
        layers.append({
            "name": f"{prefix}.qkv",
            "op": "Linear",
            "weight_shape": [3 * embed, embed],
            "input_shape": [tokens, embed],
        })
        # Attention scores Q @ K^T: (heads, tokens, head_dim) x (heads, head_dim, tokens).
        layers.append({
            "name": f"{prefix}.attn.qk",
            "op": "MatMul",
            "layer_shape": [
                [heads, tokens, head_dim],
                [heads, head_dim, tokens],
                [heads, head_dim, tokens],
            ],
        })
        # Attention output scores @ V.
        layers.append({
            "name": f"{prefix}.attn.av",
            "op": "MatMul",
            "layer_shape": [
                [heads, tokens, tokens],
                [heads, tokens, tokens],
                [heads, tokens, head_dim],
            ],
        })
        # Output projection: Linear(embed -> embed).
        layers.append({
            "name": f"{prefix}.out_proj",
            "op": "Linear",
            "weight_shape": [embed, embed],
            "input_shape": [tokens, embed],
        })
        # Feed-forward network.
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
# Config loading.
# --------------------------------------------------------------------------- #

def load_workload_configs(
    simphony_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load ``configs/workloads.json`` (searching a couple of sensible roots)."""
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
    """Resolve a workload name into ``{name, kind, cycles, layer_sizes, spec}``.

    ``arch`` supplies the architecture-dependent inputs the cycle arithmetic
    needs: ``num_wavelength``, ``miniblock``, ``dim_map`` and the operand
    representation flags.
    """
    config = load_workload_configs(simphony_root=simphony_root, config_path=config_path)
    workloads = config.get("workloads") or {}
    resolved = name or config.get("default") or DEFAULT_WORKLOAD

    if resolved == "synthetic" and resolved not in workloads:
        # Keep the legacy baseline available even if the config omits it.
        return {
            "name": "synthetic",
            "kind": "literal",
            "cycles": synthetic_cycles(),
            "layer_sizes": synthetic_layer_sizes(),
            "spec": {},
        }

    if resolved not in workloads:
        raise WorkloadError(
            f"unknown workload {resolved!r}; available: {sorted(workloads) or ['synthetic']}"
        )
    spec = dict(workloads[resolved] or {})
    kind = str(spec.get("kind", "literal"))
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise WorkloadError(f"workload {resolved!r} has unsupported kind {kind!r}")

    if kind == "literal" and resolved == "synthetic" and not spec.get("cycles"):
        # Fall back to the built-in literal constants.
        cycles, sizes = synthetic_cycles(), synthetic_layer_sizes()
    else:
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
