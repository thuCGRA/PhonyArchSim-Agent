"""面向 SimPhony real 模式的 CPU 模型工厂。

每个工作负载映射到一个 ``module:function`` 构建器，DSE CLI 通过
``--nn-model-factory tests.factory:build_<workload>`` 选择。所有模型都放在 CPU 上，
使用相同的量化层类型；模型 TYPE 决定 NN/arch 映射：``transformer``
（QConv2d/QLinear/QMatMul）或 ``cnn``（QConv2d/QLinear）。

工作负载（来自 configs/workloads.json + 模型库）：

    build_vit_tiny        ViT-Tiny 编码器   (dim 192, depth 4,  heads 3)
    build_vit_small       ViT-Small 编码器  (dim 384, depth 6,  heads 6)
    build_vit_base        ViT-Base 编码器   (dim 768, depth 12, heads 12)
    build_mobilevit_xxs   MobileViT-XXS     (224x224, dims [64, 80, 96])
    build_mobilevit_xs    MobileViT-XS      (224x224, dims [96, 120, 144])
    build_mobilevit       MobileViT-S       (224x224, dims [144, 192, 240])
    build_cnn_tiny        VGG 风格 CNN      (5 conv + 1 fc)
    build_cnn_small       VGG 风格 CNN      (8 conv + 1 fc)
    build_cnn_base        VGG 风格 CNN      (13 conv + 1 fc)
    build_model           build_vit_base 的别名
"""

from __future__ import annotations

import torch

_QCFG = {
    "conv_cfg": {"type": "QConv2d", "w_bit": 4, "in_bit": 4, "out_bit": 4},
    "linear_cfg": {"type": "QLinear", "w_bit": 4, "in_bit": 4, "out_bit": 4},
    "matmul_cfg": {"type": "QMatMul", "w_bit": 4, "in_bit": 4, "out_bit": 4},
}


def _vit(embed_dim: int, depth: int, num_heads: int):
    from models.quantized_BERT_base import QBertViTBase

    return QBertViTBase(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        conv_cfg=_QCFG["conv_cfg"],
        linear_cfg=_QCFG["linear_cfg"],
        matmul_cfg=_QCFG["matmul_cfg"],
    ).to("cpu")


def build_vit_tiny():
    return _vit(192, 4, 3)


def build_vit_small():
    return _vit(384, 6, 6)


def build_vit_base():
    return _vit(768, 12, 12)


def _mobilevit(dim, depth, channels, expansion):
    """构建一个 QMobileViT 变体；(dim, depth, channels, expansion) 元组遵循官方
    MobileViT XXS/XS/S 配置（apple/ml-cvnets），映射到内置 SimPhony QMobileViT 的
    11 元素通道布局。"""
    from models.quantized_mobileViT import QMobileViT

    return QMobileViT(
        dim=dim,
        depth=depth,
        channels=channels,
        expansion=expansion,
        image_height=224,
        image_width=224,
        conv_cfg=_QCFG["conv_cfg"],
        linear_cfg=_QCFG["linear_cfg"],
        matmul_cfg=_QCFG["matmul_cfg"],
    ).to("cpu")


def build_mobilevit_xxs():
    return _mobilevit([64, 80, 96], [2, 4, 3],
                      [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320], expansion=2)


def build_mobilevit_xs():
    return _mobilevit([96, 120, 144], [2, 4, 3],
                      [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], expansion=4)


def build_mobilevit():
    return _mobilevit([144, 192, 240], [2, 4, 3],
                      [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], expansion=4)


def _cnn(channels, blocks):
    from tests.cnn_models import QVGG

    return QVGG(
        conv_channels=channels,
        conv_blocks=blocks,
        num_classes=1000,
        conv_cfg=_QCFG["conv_cfg"],
        linear_cfg=_QCFG["linear_cfg"],
    ).to("cpu")


def build_cnn_tiny():
    return _cnn([32, 64, 128], [1, 2, 2])


def build_cnn_small():
    return _cnn([64, 128, 256, 256], [2, 2, 2, 2])


def build_cnn_base():
    return _cnn([64, 128, 256, 512, 512], [2, 2, 3, 3, 3])


# build_model 为 build_vit_base 的别名。
build_model = build_vit_base


if __name__ == "__main__":
    for name, fn in (
        ("vit_tiny", build_vit_tiny),
        ("vit_small", build_vit_small),
        ("vit_base", build_vit_base),
        ("mobilevit_xxs", build_mobilevit_xxs),
        ("mobilevit_xs", build_mobilevit_xs),
        ("mobilevit", build_mobilevit),
        ("cnn_tiny", build_cnn_tiny),
        ("cnn_small", build_cnn_small),
        ("cnn_base", build_cnn_base),
    ):
        model = fn()
        print(f"{name}: {type(model).__name__} on {next(model.parameters()).device}")
