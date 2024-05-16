import math
import torch
import torch.nn as nn

from mmyolo.registry import MODELS
from mmengine.logging import print_log

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


@MODELS.register_module()
class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channels, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top16',
                 act_cfg: dict = dict(type='Hardsigmoid')):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.freq_sel_method = freq_sel_method
        self.num_split = int(freq_sel_method[3:])
        self.channels = channels

        self.dct_layer = MultiSpectralDCTLayer(dct_h)

        act_cfg_ = act_cfg.copy()
        self.activate = MODELS.build(act_cfg_)

        self.filters_predicted = nn.Sequential(
            nn.Conv2d(channels, self.num_split, kernel_size=3, stride=2, padding=0, groups=self.num_split),
            self.activate,
            # nn.Conv2d(self.num_split, self.num_split, kernel_size=1, stride=1, padding=0, groups=self.num_split),
            # self.activate,
            nn.Conv2d(self.num_split, self.num_split, kernel_size=3, stride=2, padding=0, groups=self.num_split)
        )
        self.channels_weight_predicted = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            self.activate,
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # # yolo v8 default.
        # proj = torch.arange(self.num_split, dtype=torch.float)

        proj = torch.ones(self.num_split)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        mapper_x, mapper_y = get_freq_indices(self.freq_sel_method)
        mapper_x = torch.tensor([temp_x * (self.dct_h // 7) for temp_x in mapper_x], device=x.device)
        mapper_y = torch.tensor([temp_y * (self.dct_w // 7) for temp_y in mapper_y], device=x.device)
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.

        filters_predicted = self.filters_predicted(x_pooled)
        filters_predicted = filters_predicted.reshape(n, self.num_split, -1)
        filters_predicted = nn.functional.adaptive_avg_pool1d(filters_predicted, (self.num_split,))
        # # yolo v8 default.
        # filters_predicted = filters_predicted.softmax(2).matmul(
        #     self.proj.view([-1, 1])).squeeze(-1)

        # zhang feng added.
        filters_predicted = filters_predicted.matmul(
            self.proj.view([-1, 1])).squeeze(-1)

        filters_predicted = filters_predicted.clamp(min=0, max=self.num_split-1.1)
        # print_log(f'filters_predicted: {filters_predicted.flatten()}')

        filters_predicted_left = filters_predicted.long()
        filters_predicted_right = filters_predicted_left + 1
        batch_index = torch.arange(end=n, dtype=torch.int64, device=x_pooled.device)[..., None]
        filters_predicted_left_flatten_index = filters_predicted_left + batch_index * mapper_x.shape[0]
        filters_predicted_right_flatten_index = filters_predicted_right + batch_index * mapper_x.shape[0]

        mapper_x_left = mapper_x[None, :].repeat(n, 1)
        mapper_x_left = mapper_x_left.view(-1)[filters_predicted_left_flatten_index.view(-1)].view(n, -1)
        mapper_y_left = mapper_y[None, :].repeat(n, 1)
        mapper_y_left = mapper_y_left.view(-1)[filters_predicted_left_flatten_index.view(-1)].view(n, -1)

        dct_filter_left = x.new_full((n, self.channels), 0)
        for i in torch.arange(n):
            dct_filter_left[i] = self.dct_layer(x_pooled[i], self.dct_h, self.dct_w, mapper_x_left[i], mapper_y_left[i], self.channels)

        # update offset weight
        c_part = self.channels // self.num_split
        for i in torch.arange(self.num_split):
            offset = (filters_predicted_right - filters_predicted)
            dct_filter_left[:, i * c_part: (i + 1) * c_part] *= offset[:, i][:, None]

        # right
        mapper_x_right = mapper_x[None, :].repeat(n, 1)
        mapper_x_right = mapper_x_right.view(-1)[filters_predicted_right_flatten_index.view(-1)].view(n, -1)
        mapper_y_right = mapper_y[None, :].repeat(n, 1)
        mapper_y_right = mapper_y_right.view(-1)[filters_predicted_right_flatten_index.view(-1)].view(n, -1)

        dct_filter_right = x.new_full((n, self.channels), 0)
        for i in torch.arange(n):
            dct_filter_right[i] = self.dct_layer(x_pooled[i], self.dct_h, self.dct_w, mapper_x_right[i], mapper_y_right[i], self.channels)

        for i in torch.arange(self.num_split):
            offset = (filters_predicted - filters_predicted_left)
            dct_filter_right[:, i * c_part: (i + 1) * c_part] *= offset[:, i][:, None]

        channels_weight_predicted = self.channels_weight_predicted(dct_filter_left + dct_filter_right)
        channels_weight_predicted = channels_weight_predicted.view(n, c, 1, 1)
        return x * channels_weight_predicted.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height):
        super(MultiSpectralDCTLayer, self).__init__()
        # fixed DCT init
        self.register_buffer('dct_filters', self.build_dct_filter_table(height))

    def forward(self, x, height, width, mapper_x, mapper_y, channels):
        assert len(x.shape) == 3, 'x must been 3 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape
        assert len(mapper_x) == len(mapper_y)
        assert channels % len(mapper_x) == 0

        weight = self.get_dct_filter(height, width, mapper_x, mapper_y, channels)
        x = x * weight
        result = torch.sum(x, dim=[1, 2])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    # def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
    #     dct_filter = mapper_x.new_full((channel, tile_size_x, tile_size_y), 0, dtype=torch.float)
    #
    #     c_part = channel // len(mapper_x)
    #
    #     for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
    #         dct_filter[i * c_part: (i + 1) * c_part, :, :] = self.dct_filters[:, :, None][u_x, :] * self.dct_filters[
    #                                                                                     v_y, :]
    #
    #     return dct_filter

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        c_part = channel // len(mapper_x)

        dct_filters_x = self.dct_filters[mapper_x, :][:, :, None]
        dct_filters_y = self.dct_filters[mapper_y, :][:, None, :]

        dct_filter = dct_filters_x * dct_filters_y

        dct_filter = dct_filter[:, None, ...].repeat(1, c_part, 1, 1).view(-1, tile_size_x, tile_size_y)

        return dct_filter

    def build_dct_filter_table(self, tile_size):
        dct_filter = torch.zeros((tile_size, tile_size), dtype=torch.float)
        for freq in range(tile_size):
            for pos in range(tile_size):
                dct_filter[freq, pos] = self.build_filter(pos, freq, tile_size)

        return dct_filter
