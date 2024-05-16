# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from mmyolo.registry import MODELS
from mmengine.logging import print_log

# @weighted_loss
# def quality_focal_loss(pred, target, alpha=0.75):
#     r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
#     Qualified and Distributed Bounding Boxes for Dense Object Detection
#     <https://arxiv.org/abs/2006.04388>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted joint representation of classification
#             and quality (IoU) estimation with shape (N, C), C is the number of
#             classes.
#         target (tuple([torch.Tensor])): Target category label with shape (N,)
#             and target quality label with shape (N,).
#
#     Returns:
#         torch.Tensor: Loss tensor with shape (N,).
#     """
#     assert len(target) == 2, """target for QFL must be a tuple of two elements,
#         including category label and quality label, respectively"""
#     # label denotes the category id, score denotes the quality score
#     label, original_score = target
#     label = label.reshape(-1)
#     score_focal_weight = original_score.view(pred.shape)
#
#     # negatives are supervised by 0 quality score
#     pred_sigmoid = pred.sigmoid()
#     # overflow = torch.logical_or(pred_sigmoid < 1e-7, pred_sigmoid > (1 - 1e-7))
#     # if overflow.sum() > 0:
#     #     print_log(f'overflow pre_sigmoid: {pred_sigmoid[overflow]}')
#     pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1 - 1e-7)
#
#     focal_weight_pos = score_focal_weight * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
#                        (1 - alpha) * (torch.exp(pred_sigmoid) - 1) * pred_sigmoid * (score_focal_weight <= 0.0).float()
#
#     focal_weight_neg = (1 - score_focal_weight) * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
#                        alpha * (torch.exp(pred_sigmoid) - 1) * pred_sigmoid * (score_focal_weight <= 0.0).float()
#
#     zerolabel = pred_sigmoid.new_zeros(pred.shape)
#
#     loss = -zerolabel * torch.log(pred_sigmoid) * focal_weight_pos - (1 - zerolabel) * torch.log(1 - pred_sigmoid) * focal_weight_neg
#
#     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
#     bg_class_ind = pred.size(1)
#     pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
#     pos_label = label[pos].long()
#
#     score = original_score.sum(-1).reshape(-1)
#
#     loss[pos, pos_label] = -score[pos] * torch.log(pred_sigmoid[pos, pos_label]) * focal_weight_pos[
#         pos, pos_label] - (1 - score[pos]) * torch.log(
#         1 - pred_sigmoid[pos, pos_label]) * focal_weight_neg[pos, pos_label]
#
#     # loss = loss.sum(dim=1, keepdim=False)
#     return loss

# @weighted_loss
# def quality_focal_loss(pred, target, alpha=0.75):
#     r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
#     Qualified and Distributed Bounding Boxes for Dense Object Detection
#     <https://arxiv.org/abs/2006.04388>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted joint representation of classification
#             and quality (IoU) estimation with shape (N, C), C is the number of
#             classes.
#         target (tuple([torch.Tensor])): Target category label with shape (N,)
#             and target quality label with shape (N,).
#
#     Returns:
#         torch.Tensor: Loss tensor with shape (N,).
#     """
#     assert len(target) == 2, """target for QFL must be a tuple of two elements,
#         including category label and quality label, respectively"""
#     # label denotes the category id, score denotes the quality score
#     label, original_score = target
#     label = label.reshape(-1)
#     score_focal_weight = original_score.view(pred.shape)
#
#     # negatives are supervised by 0 quality score
#     pred_sigmoid = pred.sigmoid()
#     # overflow = torch.logical_or(pred_sigmoid < 1e-7, pred_sigmoid > (1 - 1e-7))
#     # if overflow.sum() > 0:
#     #     print_log(f'overflow pre_sigmoid: {pred_sigmoid[overflow]}')
#     pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1 - 1e-7)
#
#     focal_weight_pos = score_focal_weight * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
#                        (1 - alpha) * (torch.exp(pred_sigmoid) - 1) * pred_sigmoid * (score_focal_weight <= 0.0).float()
#
#     focal_weight_neg = (1 - score_focal_weight) * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
#                        alpha * (torch.exp(pred_sigmoid) - 1) * pred_sigmoid * (score_focal_weight <= 0.0).float()
#
#     zerolabel = pred_sigmoid.new_zeros(pred.shape)
#
#     loss = -zerolabel * torch.log(pred_sigmoid) * focal_weight_pos - (1 - zerolabel) * torch.log(1 - pred_sigmoid) * focal_weight_neg
#
#     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
#     bg_class_ind = pred.size(1)
#     pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
#     pos_label = label[pos].long()
#
#     mmm = score_focal_weight[pos, pos_label]
#     nnn = pred_sigmoid[pos, pos_label]
#
#     loss[pos, pos_label] = -score_focal_weight[pos, pos_label] * torch.log(pred_sigmoid[pos, pos_label]) * focal_weight_pos[
#         pos, pos_label] - (1 - score_focal_weight[pos, pos_label]) * torch.log(
#         1 - pred_sigmoid[pos, pos_label]) * focal_weight_neg[pos, pos_label]
#
#     # loss = loss.sum(dim=1, keepdim=False)
#     return loss

# @weighted_loss
# def quality_focal_loss(pred, target, alpha=0.75):
#     r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
#     Qualified and Distributed Bounding Boxes for Dense Object Detection
#     <https://arxiv.org/abs/2006.04388>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted joint representation of classification
#             and quality (IoU) estimation with shape (N, C), C is the number of
#             classes.
#         target (tuple([torch.Tensor])): Target category label with shape (N,)
#             and target quality label with shape (N,).
#
#     Returns:
#         torch.Tensor: Loss tensor with shape (N,).
#     """
#     assert len(target) == 2, """target for QFL must be a tuple of two elements,
#         including category label and quality label, respectively"""
#     # label denotes the category id, score denotes the quality score
#     label, original_score = target
#     label = label.reshape(-1)
#     score_focal_weight = original_score.view(pred.shape)
#
#     # negatives are supervised by 0 quality score
#     pred_sigmoid = pred.sigmoid()
#     # overflow = torch.logical_or(pred_sigmoid < 1e-7, pred_sigmoid > (1 - 1e-7))
#     # if overflow.sum() > 0:
#     #     print_log(f'overflow pre_sigmoid: {pred_sigmoid[overflow]}')
#     pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1 - 1e-7)
#
#     loss = -score_focal_weight * torch.log(pred_sigmoid) - (1 - score_focal_weight) * torch.log(1 - pred_sigmoid)
#     # loss = loss.sum(dim=1, keepdim=False)
#     return loss


@weighted_loss
def quality_focal_loss(pred, target, alpha=0.75):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, original_score = target
    label = label.reshape(-1)
    score_focal_weight = original_score.view(pred.shape)

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-7, max=1 - 1e-7)

    focal_weight_pos = score_focal_weight * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
                       alpha * (torch.exp(pred_sigmoid) - 1) * pred_sigmoid * (score_focal_weight <= 0.0).float()

    focal_weight_neg = (1 - score_focal_weight) * torch.exp((score_focal_weight - pred_sigmoid).abs()) * (score_focal_weight > 0.0).float() + \
                       alpha * (torch.exp(pred_sigmoid) - 1) * (score_focal_weight <= 0.0).float()

    loss = -score_focal_weight * torch.log(pred_sigmoid) * focal_weight_pos - (1 - score_focal_weight) * torch.log(1 - pred_sigmoid) * focal_weight_neg

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def quality_focal_loss_tensor_target(pred, target, beta=2.0, activated=False):
    """`QualityFocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        activated (bool): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    if activated:
        pred_sigmoid = pred
        loss_function = F.binary_cross_entropy
    else:
        pred_sigmoid = pred.sigmoid()
        loss_function = F.binary_cross_entropy_with_logits

    scale_factor = pred_sigmoid
    target = target.type_as(pred)

    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = loss_function(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    pos = (target != 0)
    scale_factor = target[pos] - pred_sigmoid[pos]
    loss[pos] = loss_function(
        pred[pos], target[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def quality_focal_loss_with_prob(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Different from `quality_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss

@MODELS.register_module()
class QualityVarialFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        activated (bool, optional): Whether the input is activated.
            If True, it means the input has been activated and can be
            treated as probabilities. Else, it should be treated as logits.
            Defaults to False.
    """

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(QualityVarialFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (Union(tuple([torch.Tensor]),Torch.Tensor)): The type is
                tuple, it should be included Target category label with
                shape (N,) and target quality label with shape (N,).The type
                is torch.Tensor, the target should be one-hot form with
                soft weights.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                calculate_loss_func = quality_focal_loss
            if isinstance(target, torch.Tensor):
                # the target shape with (N,C) or (N,C,...), which means
                # the target is one-hot form with soft weights.
                calculate_loss_func = partial(
                    quality_focal_loss_tensor_target, activated=self.activated)

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


