_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = './data/glcd/'
class_name = ('Black Point', 'Dust -Nearly line-', 'Dust -Nearly round-', 'White Bar')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])

close_mosaic_epochs = 5

max_epochs = 65
train_batch_size_per_gpu = 4  # default 4.
train_num_workers = 2

load_from = None

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 0.5  # 0.5
loss_qfl_weight = 1.0  # ta is 3000. dsa is 1000
loss_bbox_weight = 7.5  # default is 7.5. ta is 8. dsa is 8.
qfl_beta = 2.0  # beta of QualityFocalLoss
img_scale = (960, 960)  # width, height
# ratio range for random resize
random_resize_ratio_range = (0.5, 1)
# Number of cached images in mosaic
mosaic_max_cached_images = 20
# Number of cached images in mixup
mixup_max_cached_images = 10

# default batch size is 8. default base_lr is 0.01.
# according to GPU situation, modify the base_lr, modification ratio is base_lr_default * (your_bs / default_bs)
base_lr = 0.01 / 2
weight_decay = 0.05
lr_start_factor = 1.0e-5

# The scaling factor that controls the depth of the network structure
deepen_factor = 0.5  # default 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.375  # default 0.5, 0.375

model_test_cfg = dict(
    # The number of boxes before NMS
    nms_pre=3000,
    score_thr=0.4,  # Threshold to filter out boxes.
    nms=dict(type='soft_nms', iou_threshold=0.3),  # NMS type and threshold
)  # Max number of detections of each image

model = dict(
    backbone=dict(
        frozen_stages=-1,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # act_cfg=dict(inplace=True, type='ReLU'),
        attention_cfg=dict(
            type='MultiSpectralAttentionLayer',
            # type='EffectiveSELayer',
            # type='EfficientChannelAttention',
            act_cfg=dict(type='Hardsigmoid', inplace=True),
            # reduction=8,
            # freq_sel_method='top32'
        ),
        # attention_cfg=None,
        spp_cfg=dict(
            type='SPPFBottleneck',
            act_cfg=dict(type='SiLU', inplace=True),
            kernel_sizes=3,
            # use_conv_first=False
        ),
        drop_block_cfg=dict(
            _delete_=True,
            type='mmdet.DropBlock',
            drop_prob=0.1,
            block_size=3,
            warmup_iters=500
        ),
        use_spd_stem_layer=False,
        use_cspnext_block=True
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # act_cfg=dict(inplace=True, type='ReLU'),
        attention_cfg=dict(
            type='MultiSpectralAttentionLayer',
            # type='EffectiveSELayer',
            # type='EfficientChannelAttention',
            act_cfg=dict(type='Hardsigmoid', inplace=True),
            # reduction=8,
            # freq_sel_method='top32'
        ),
        # attention_cfg=None,
        use_cspnext_block=True
    ),
    bbox_head=dict(
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            widen_factor=widen_factor,
            # act_cfg=dict(inplace=True, type='ReLU'),
            reg_max=16),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight
        ),
        loss_cls_weight=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight / 4,  # loss_cls_weight = 0.5
            class_weight=(1, 1, 1, 1),
            # pos_weight=(1, 0.5, 0.5, 1)
        ),
        loss_cls_qfl=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_qfl_weight
        ),
        loss_cls_sdwfl=dict(
            type='QualityVarialFocalLoss',
            use_sigmoid=True,
            loss_weight=loss_qfl_weight,
            reduction='none',
            alpha=0.5,  # default 0.75
        ),
        loss_cls_vfl=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            reduction='none',
            alpha=0.5,  # default 0.75
            gamma=1.5  # default 2.0
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='siou', # default ciou
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        _delete_=True,
        initial_epoch=max_epochs, #initial_epoch is how many epochs the initial assigner has.
        # initial_epoch=0,
        initial_assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            soft_center_radius=2.0,
            iou_weight=3.0,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        warmup_epoch=0,
        warmup_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)
    ),
    test_cfg=model_test_cfg
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         backend_args=_base_.backend_args),
#     dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 这里将 LetterResize 修改成 mmdet.Resize
#     dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline
    )
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(
    ann_file=data_root + 'valid.json',
    classwise=True
)
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    classwise=True
)

default_hooks = dict(
    checkpoint=dict(interval=5, save_best='coco/bbox_mAP_s'),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=300),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 16,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=2, type='EpochBasedTrainLoop')
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
