from datetime import datetime
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.models import DiceLoss, RTMDetInsSepBNHead
from mmengine.dataset import DefaultSampler
from torch.nn import SyncBatchNorm
from torch.nn.modules.activation import SiLU

from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import FilterAnnotations, LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import (
    CachedMixUp,
    CachedMosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    YOLOXHSVRandomAug,
)
from mmdet.models.backbones.cspnext import CSPNeXt
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.detectors.rtmdet import RTMDet
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.iou_loss import CIoULoss
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import (
    DynamicSoftLabelAssigner,
)
from mmdet.models.task_modules.coders.distance_point_bbox_coder import (
    DistancePointBBoxCoder,
)
from mmdet.models.task_modules.prior_generators.point_generator import (
    MlvlPointGenerator,
)
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.adamw import AdamW
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.runner import LogProcessor

from mmdet.engine.hooks import DetVisualizationHook
from mmdet.visualization import DetLocalVisualizer
from mmengine.visualization import TensorboardVisBackend, WandbVisBackend, NeptuneVisBackend

backend_args = None
data_root = "/home/g/gajdosech2/Hamburg2024/"
train_annotations_file = "coco_annotations_train_data_3_frag_nokp.json"
test_annotations_file = "coco_annotations_val_data_3_frag_nokp.json"
train_images_dir = ""
test_images_dir = ""

max_epochs = 500
lr = 0.01
val_interval = checkpoint_interval = 50
batch_size = 6
num_workers = 8

default_scope = None
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook),
    checkpoint=dict(type=CheckpointHook, interval=checkpoint_interval, max_keep_ckpts=10)
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
vis_backends = [
    #dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project="TransparentObjects")),
    #dict(type='NeptuneVisBackend', init_kwargs=dict(project="l.gajdosech/TransparentObjects")),
    ]
visualizer = dict(type=DetLocalVisualizer, vis_backends=vis_backends, name="visualizer")
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = "INFO"
load_from = None
resume = False


model = dict(
    type=RTMDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        #mean=[103.53, 116.28, 123.675],
        #std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type=CSPNeXt,
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True),
    ),
    neck=dict(
        type=CSPNeXtPAFPN,
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True),
    ),
    bbox_head=dict(
        type=RTMDetInsSepBNHead,
        num_classes=6,
        in_channels=192,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=192,
        act_cfg=dict(type=SiLU, inplace=True),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        anchor_generator=dict(type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss, 
            use_sigmoid=True, 
            beta=1.0, 
            loss_weight=1.0
        ),
        loss_bbox=dict(type=CIoULoss, loss_weight=2.0),
        loss_mask=dict(type=DiceLoss, loss_weight=3.0, eps=5e-6, reduction="mean"),
    ),
    train_cfg=dict(
        assigner=dict(type=DynamicSoftLabelAssigner, topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=100,
        min_bbox_size=0,
        score_thr=0.1,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=10,
        mask_thr_binary=0.5,
    ),
    init_cfg=dict(
            type='Pretrained',
            checkpoint=data_root + "/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth")
)

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(type=Resize, scale=(640, 640), keep_ratio=True),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMosaic, 
        img_scale=(640, 640), 
        pad_val=114.0,
        prob=0.7
    ),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.5, 1.2),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(
        type=RandomCrop,
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type=YOLOXHSVRandomAug, hue_delta=10, saturation_delta=40, value_delta=40),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=CachedMixUp,
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114),
    ),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1, 1)),
    dict(type=PackDetInputs),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=CocoDataset,
        metainfo=dict(
            classes = ("shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"),
            palette = [
                (250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0), (250, 0, 250), (0, 250, 250)
            ]),
        data_root=data_root,
        ann_file=train_annotations_file,
        data_prefix=dict(img=train_images_dir),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(type=Resize, scale=(640, 640), keep_ratio=True),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CocoDataset,
        metainfo=train_dataloader["dataset"]["metainfo"],
        data_root=data_root,
        ann_file=test_annotations_file,
        data_prefix=dict(img=test_images_dir),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + test_annotations_file,
    metric=["bbox", "segm"],
    format_only=False,
    proposal_nums=(100, 1, 10),
    backend_args=backend_args,
)
test_evaluator = val_evaluator

train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=val_interval
)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=True, begin=0, end=50),
    dict(
        type=MultiStepLR,
        by_epoch=True,
        milestones=[100, 200, 400],
        gamma=0.1)
]

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))