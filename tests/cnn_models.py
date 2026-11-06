"""面向 SimPhony real 模式的量化 VGG 风格纯 CNN 模型。

仅使用 QConv2d + QLinear 层（无 QMatMul），匹配 ``--model-type cnn`` 所选的
``simple_cnn`` NN/arch 映射。下采样通过 stride-2 卷积完成（全卷积 VGG 变体），
因此每个可学习层都是量化层，池化/BN 层不会泄漏进 ONN 层列表。
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from models.quantized_base import ConvBlock, LinearBlock, QBaseModel


class QVGG(QBaseModel):
    """参数化的 VGG 风格堆叠：卷积阶段 + 单个线性头。

    ``conv_channels`` 给出每个阶段的输出通道数；``conv_blocks`` 给出每个阶段的
    3x3 卷积数。除第一阶段外，每个阶段的首个卷积用 stride 2 下采样。全局平均池化 +
    一个线性层输出 logits。
    """

    def __init__(
        self,
        conv_channels: List[int],
        conv_blocks: List[int],
        num_classes: int = 1000,
        conv_cfg: dict = dict(type="QConv2d"),
        linear_cfg: dict = dict(type="QLinear"),
        matmul_cfg: dict | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            conv_cfg=conv_cfg, linear_cfg=linear_cfg, matmul_cfg=matmul_cfg
        )
        if len(conv_channels) != len(conv_blocks):
            raise ValueError("conv_channels and conv_blocks must be aligned")

        features: List[nn.Module] = []
        in_ch = 3
        for stage, (out_ch, n) in enumerate(zip(conv_channels, conv_blocks)):
            for i in range(n):
                # 每个阶段的首个卷积（除阶段 0 外）按 2 倍下采样。
                stride = 2 if (stage > 0 and i == 0) else 1
                features.append(
                    ConvBlock(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        conv_cfg=dict(conv_cfg),
                        device=device,
                    )
                )
                in_ch = out_ch

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearBlock(
            in_ch,
            num_classes,
            linear_cfg=dict(linear_cfg),
            act_cfg=None,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
