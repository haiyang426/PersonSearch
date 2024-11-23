_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='NAE',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
            type='NaeRoIHead',
            with_fcs=False,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
                out_channels=1024,
                featmap_strides=[16]),
            bbox_head=dict(
                type='NaeHead',
                with_avg_pool=True,
                with_cls=True,
                roi_feat_size=7,
                in_channels=2048,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=10),
                loss_id=dict(
                    type='OIMLoss', 
                    num_features=256,
                    num_pids=5532,
                    num_cq_size=5000,
                    oim_momentum=0.5,
                    oim_scalar=30,
                    loss_weight=1.0,
                    ),
                shared_head=dict(
                    type='ResLayer',
                    depth=50,
                    stage=3,
                    stride=2,
                    dilation=1,
                    style='pytorch',
                    norm_cfg=norm_cfg,
                    norm_eval=True,
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint='torchvision://resnet50')))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=8),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.1,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            nms_post=300,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=16),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100)))



# dataset settings
dataset_type = 'PRWDataset'
data_root = 'data/CUHK-SYSU/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1500, 900), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1500, 900), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict( 
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_coco/train_cocoformat.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_coco/test_cocoformat.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


test_dataloader = val_dataloader

val_evaluator = dict(
    type='CUHKMetric',
    ann_file=data_root + 'annotations_coco/test_cocoformat.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


# ExtracrFeatureHook
train_cluster_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1500, 900), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_cluster_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_unlabel_cocoformat.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=train_cluster_pipeline,
        backend_args=backend_args))



# training schedule for 20e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=22, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2802),
    dict(
        type='MultiStepLR',
        begin=0,
        end=22,
        by_epoch=True,
        milestones=[16, ],
        gamma=0.1)
]

# optimizer[]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=10, norm_type=2),)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)