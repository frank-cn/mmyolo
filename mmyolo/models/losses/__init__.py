# Copyright (c) OpenMMLab. All rights reserved.
from .iou_loss import IoULoss, bbox_overlaps
from .oks_loss import OksLoss
from .ce_loss import CrossEntropyLoss

__all__ = ['IoULoss', 'bbox_overlaps', 'OksLoss', 'CrossEntropyLoss']
