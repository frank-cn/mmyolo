# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple, Union
from ..layers import CSPLayerWithTwoConv, ASFF
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmengine import MessageHub
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess, make_divisible
from .yolov5_head import YOLOv5Head
from mmengine.logging import print_log


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

        # # asff stage123
        # self.asff_1 = ASFF(level=0, multiplier=widen_factor * 0.5)
        # self.asff_2 = ASFF(level=1, multiplier=widen_factor * 0.5)
        # self.asff_3 = ASFF(level=2, multiplier=widen_factor * 0.5)

        # # asff stage234
        # self.asff_1 = ASFF(level=0, multiplier=widen_factor)
        # self.asff_2 = ASFF(level=1, multiplier=widen_factor)
        # self.asff_3 = ASFF(level=2, multiplier=widen_factor)

        # # asff stage1234
        # self.asff_1 = ASFF(level=0, multiplier=widen_factor)
        # self.asff_2 = ASFF(level=1, multiplier=widen_factor)
        # self.asff_3 = ASFF(level=2, multiplier=widen_factor)
        # self.asff_4 = ASFF(level=3, multiplier=widen_factor)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride) ** 2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    CSPLayerWithTwoConv(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        # kernel_size=3,
                        # stride=1,
                        # padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)
                )
            )
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    CSPLayerWithTwoConv(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        # kernel_size=3,
                        # stride=1,
                        # padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1)
                )
            )

        # yolo v8 default
        # proj = torch.arange(self.reg_max, dtype=torch.float)

        proj = torch.ones(self.reg_max)

        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels

        # # asff. stage123 or 234
        # pan_out0 = self.asff_1(x)
        # pan_out1 = self.asff_2(x)
        # pan_out2 = self.asff_3(x)
        # x = (pan_out2, pan_out1, pan_out0)

        # # asff. stage1234
        # pan_out0 = self.asff_1(x)
        # pan_out1 = self.asff_2(x)
        # pan_out2 = self.asff_3(x)
        # pan_out3 = self.asff_4(x)
        # x = (pan_out3, pan_out2, pan_out1, pan_out0)

        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)

            # # yolo v8 default
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(
            #     self.proj.view([-1, 1])).squeeze(-1)

            bbox_preds = bbox_dist_preds.sigmoid().matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    area_rule = [0, 32, 96, 1e5]

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_cls_qfl: ConfigType = dict(
                     type='mmdet.QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_cls_vfl: ConfigType = dict(
                     type='mmdet.VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0),
                 loss_cls_sdwfl: ConfigType = dict(
                     type='QualityVarialFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0,
                     alpha=0.75,
                     gamma=2.0),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 loss_cls_weight: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_cls_qfl: nn.Module = MODELS.build(loss_cls_qfl)
        self.loss_cls_vfl: nn.Module = MODELS.build(loss_cls_vfl)
        self.loss_cls_sdwfl: nn.Module = MODELS.build(loss_cls_sdwfl)
        self.loss_cls_weight = MODELS.build(loss_cls_weight)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.initial_epoch = self.train_cfg['initial_epoch']
            self.initial_assigner = TASK_UTILS.build(
                self.train_cfg.initial_assigner)
            self.warmup_epoch = self.train_cfg['warmup_epoch']
            self.warmup_assigner = TASK_UTILS.build(
                self.train_cfg.warmup_assigner)
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        # get epoch information from message hub
        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')

        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        gt_bboxes_wh = gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]
        gt_bboxes_area = gt_bboxes_wh[..., 0] * gt_bboxes_wh[..., 1]
        gt_bboxes_area = gt_bboxes_area * pad_bbox_flag.squeeze(-1)

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        if current_epoch < self.initial_epoch:
            assigned_result = self.initial_assigner(
                flatten_pred_bboxes.detach(),
                flatten_cls_preds.detach(),
                self.flatten_priors_train, gt_labels,
                gt_bboxes, pad_bbox_flag, gt_bboxes_area, self.area_rule)
            assigned_labels = assigned_result['assigned_labels']
            assigned_labels_weights = assigned_result['assigned_labels_weights']
            assigned_bboxes = assigned_result['assigned_bboxes']
            assigned_scores = assigned_result['assigned_scores']
            fg_mask_pre_prior = assigned_result['fg_mask_inboxes']
        elif current_epoch < self.initial_epoch + self.warmup_epoch:
            pass
        else:
            assigned_result = self.assigner(
                (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
                flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
                gt_labels, gt_bboxes, pad_bbox_flag, gt_bboxes_area, self.area_rule)
            assigned_labels = assigned_result['assigned_labels']
            assigned_labels_weights = assigned_result['assigned_labels_weights']
            assigned_bboxes = assigned_result['assigned_bboxes']
            assigned_scores = assigned_result['assigned_scores']
            fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']


        gt_bboxes_area_distribution = assigned_result['gt_bboxes_area_distribution']
        assigned_labels_bbox_weights = assigned_result['assigned_labels_bbox_weights']
        dynamic_gt_class_weight = assigned_result['dynamic_gt_class_weight']

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos <= 0:
            print_log(f'num of positive anchors: {num_pos}')
            return dict(
                loss_cls=flatten_pred_bboxes.sum() * 0,
                loss_bbox=flatten_pred_bboxes.sum() * 0,
                loss_dfl=flatten_pred_bboxes.sum() * 0
            )

        a = assigned_bboxes[fg_mask_pre_prior]
        b = assigned_scores[fg_mask_pre_prior]
        c = assigned_labels[fg_mask_pre_prior]
        d = flatten_cls_preds[fg_mask_pre_prior]
        e = assigned_labels_weights[fg_mask_pre_prior]
        f = assigned_labels_bbox_weights[fg_mask_pre_prior]

        # Zhang Feng added to enlarge IoUs of positive samples. Remove this code line if original IoUs are selected.
        assigned_scores = assigned_scores * assigned_labels_bbox_weights[..., None]

        assigned_scores_sum = reduce_mean(assigned_scores.sum()).clamp_(min=1).item()

        batch_index, pos_anchor_index = (fg_mask_pre_prior > 0).nonzero(as_tuple=True)
        pred_sigmoid = flatten_cls_preds.sigmoid()
        scale_factor = pred_sigmoid.clone()
        scale_factor[batch_index, pos_anchor_index] = \
            assigned_scores[fg_mask_pre_prior] - pred_sigmoid[fg_mask_pre_prior]

        # # Solution 2. An update solution by following the dynamic weights of QFL. Zhang Feng @2023/08/06
        # loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores, scale_factor.pow(2))
        # loss_cls = loss_cls * assigned_labels_weights
        # loss_cls = loss_cls.sum()
        # loss_cls /= assigned_scores_sum
        # loss_cls_2 = loss_cls

        # # Solution 3.
        # loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
        # loss_cls = loss_cls * assigned_labels_weights
        # loss_cls = loss_cls.sum()
        # loss_cls /= assigned_scores_sum
        # loss_cls_3 = loss_cls

        # Solution 4. YOLO V8 Original. BCE
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum
        loss_cls_4 = loss_cls

        # BCE with dynamic class weights.
        self.loss_cls_weight.class_weight = dynamic_gt_class_weight
        loss_cls_weight = self.loss_cls_weight(flatten_cls_preds, assigned_scores).sum()
        loss_cls_weight /= assigned_scores_sum

        # Same with solution #3.
        # self.loss_cls_weight.class_weight = (1, 1, 1, 1)
        # loss_cls_weight_4C = self.loss_cls_weight(flatten_cls_preds, assigned_scores, assigned_labels_weights).sum()
        # loss_cls_weight_4C /= assigned_scores_sum

        # default vfl.
        loss_cls_vfl = self.loss_cls_vfl(
            flatten_cls_preds, assigned_scores).sum()
        loss_cls_vfl /= assigned_scores_sum
        loss_cls_vfl_1 = loss_cls_vfl
        #

        # default sdwfl
        loss_cls_sdwfl = self.loss_cls_sdwfl(
            flatten_cls_preds.reshape(-1, self.num_classes),
            (assigned_labels, assigned_scores)).sum()
        loss_cls_sdwfl /= assigned_scores_sum
        loss_cls_sdwfl_1 = loss_cls_sdwfl

        # Solution QFL.
        loss_cls_qfl = self.loss_cls_qfl(
            flatten_cls_preds.reshape(-1, self.num_classes),
            (assigned_labels.reshape(-1), assigned_scores.sum(-1).reshape(-1)),
            avg_factor=assigned_scores_sum)
        loss_cls_qfl_1 = loss_cls_qfl

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])

            # Solution 6. Original
            bbox_weight = torch.masked_select(
                (assigned_scores.sum(-1)), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum
            loss_bbox_6 = loss_bbox
            # End

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])

            # Solution 1. Original.
            bbox_weight = torch.masked_select(
                (assigned_scores.sum(-1)), fg_mask_pre_prior).unsqueeze(-1)
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
            loss_dfl_1 = loss_dfl
            # End

        _, world_size = get_dist_info()

        gt_bboxes_small_num, gt_bboxes_middle_num, gt_bboxes_large_num = gt_bboxes_area_distribution.sum((1, 2))

        # suitable for task alignment, original v8 backbone and neck
        return dict(
            # loss_cls_4=loss_cls_4 * num_imgs * world_size,  # yolo v8 default
            # cost_cls_qfl_1=loss_cls_qfl_1 * num_imgs * world_size,
            # cost_cls_vfl_1=loss_cls_vfl_1 * num_imgs * world_size,
            loss_cls_sdwfl_1=loss_cls_sdwfl_1 * num_imgs * world_size,  # selected
            # cost_cls_weight=loss_cls_weight * num_imgs * world_size,
            loss_bbox_6=loss_bbox_6 * num_imgs * world_size,  # default and selected
            loss_dfl_1=loss_dfl_1 * num_imgs * world_size,  # default and selected
            gt_bboxes_small_num=gt_bboxes_small_num.float(),
            gt_bboxes_middle_num=gt_bboxes_middle_num.float(),
            gt_bboxes_large_num=gt_bboxes_large_num.float(),
        )
