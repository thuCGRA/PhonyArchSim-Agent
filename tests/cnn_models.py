"""Quantized VGG-style pure-CNN model for SimPhony real mode.

Uses only QConv2d + QLinear layers (no QMatMul), matching the ``simple_cnn``
NN/arch mapping selected by ``--model-type cnn``. Downsampling is done with
stride-2 convolutions (an all-convolutional VGG variant), so every learnable
layer is a quantized layer and none of the pooling/BN layers leak into the
ONN layer list.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from models.quantized_base import ConvBlock, LinearBlock, QBaseModel


class QVGG(QBaseModel):
    """Parameterized VGG-style stack: conv stages + a single linear head.

    ``conv_channels`` gives each stage's output channels; ``conv_blocks`` gives
    the number of 3x3 convs per stage. The first conv of every stage except the
    first uses stride 2 to downsample. A global average pool + one linear layer
    produces the logits.
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
                # First conv of each stage (except stage 0) downsamples by 2.
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
