# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import OptMultiConfig, ConfigType
from mmengine.model import BaseModule

from mmyolo.registry import MODELS

# multiple CBAM attention variants.


class ChannelAttention_ECA_ActivateFirst(BaseModule):
    """ChannelAttention.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self, channels, b=1, gamma=2, act_cfg: ConfigType = dict(type='HSigmoid')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = MODELS.build(act_cfg_)

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        avgpool_out = self.activate(y)

        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        maxpool_out = self.activate(y)

        out = (avgpool_out + maxpool_out).expand_as(x)

        return out


class ChannelAttention_ECA_ActivateLast(BaseModule):
    """ChannelAttention.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self, channels, b=1, gamma=2, act_cfg: ConfigType = dict(type='HSigmoid')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = MODELS.build(act_cfg_)

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        avgpool_out = self.activate(y)

        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        maxpool_out = self.activate(y)

        # Multi-scale information fusion
        out = self.activate(avgpool_out + maxpool_out)

        return out.expand_as(x)


class ChannelAttention(BaseModule):
    """ChannelAttention. Default.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self,
                 channels: int,
                 reduce_ratio: int = 16,
                 act_cfg: dict = dict(type='ReLU')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = MODELS.build(act_cfg_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avgpool_out = self.fc(self.avg_pool(x))
        maxpool_out = self.fc(self.max_pool(x))
        out = self.activate(avgpool_out + maxpool_out)
        return out


class ChannelAttention_EffectiveSE(BaseModule):
    """ChannelAttention. Default.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self,
                 channels: int,
                 act_cfg: dict = dict(type='ReLU')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        assert isinstance(act_cfg, dict)
        self.fc = ConvModule(channels, channels, 1, act_cfg=act_cfg)

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = MODELS.build(act_cfg_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avgpool_out = self.fc(self.avg_pool(x))
        maxpool_out = self.fc(self.max_pool(x))
        out = self.activate(avgpool_out + maxpool_out)
        return out


class SpatialAttention(BaseModule):
    """SpatialAttention
    Args:
         kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=dict(type='Sigmoid')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out


@MODELS.register_module()
class CBAM(BaseModule):
    """Convolutional Block Attention Module. arxiv link:
    https://arxiv.org/abs/1807.06521v2.

    Args:
        in_channels (int): The input (and output) channels of the CBAM.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 reduce_ratio: int = 16,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.channel_attention = ChannelAttention(
            channels=channels, reduce_ratio=reduce_ratio, act_cfg=act_cfg)

        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


@MODELS.register_module()
class CBAM_ECA(BaseModule):
    """Convolutional Block Attention Module. arxiv link:
    https://arxiv.org/abs/1807.06521v2.

    Args:
        channels (int): The input (and output) channels of the CBAM.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 7,
                 b=1,
                 gamma=2,
                 act_cfg: ConfigType = dict(type='HSigmoid'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.channel_attention = ChannelAttention_ECA_ActivateLast(
            channels=channels, b=b, gamma=gamma, act_cfg=act_cfg)

        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


@MODELS.register_module()
class CBAM_EffectiveSE(BaseModule):
    """Convolutional Block Attention Module. arxiv link:
    https://arxiv.org/abs/1807.06521v2.

    Args:
        channels (int): The input (and output) channels of the CBAM.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.channel_attention = ChannelAttention_EffectiveSE(
            channels=channels, act_cfg=act_cfg)

        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out