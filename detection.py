import mmcv
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmengine import Config
from mmdet.registry import VISUALIZERS, DATASETS
from mmengine.utils import ProgressBar
from mmengine.runner import set_random_seed
from mmdet.structures.bbox import BaseBoxes
from mmdet.models.utils import mask2ndarray
from mmdet.utils import register_all_modules
import cv2
import os.path as osp
import os
os.chdir("/home/g/gajdosech2/Hamburg2024")


def modify_config():
    cfg = Config.fromfile('/home/g/gajdosech2/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py')


    cfg.metainfo = {
        'classes': ("shot_glass", "beer_glass", "wine_glass"),
        'palette': [
            (220, 20, 60), (220, 20, 60), (220, 20, 60)
        ]
    }

    # Modify dataset type and path
    cfg.data_root = "/home/g/gajdosech2/Hamburg2024/"

    cfg.train_dataloader.dataset.ann_file = "coco_annotations.json"
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix.img = ''
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo

    cfg.val_dataloader.dataset.ann_file = "coco_annotations.json"
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = ''
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo

    cfg.test_dataloader = cfg.val_dataloader

    # Modify metric config
    cfg.val_evaluator.ann_file = cfg.data_root+'/'+"coco_annotations.json"
    cfg.test_evaluator = cfg.val_evaluator

    # Modify num classes of the model in box head and mask head
    cfg.model.bbox_head.num_classes = 3

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './work_dirs'


    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 3
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 3

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optim_wrapper.optimizer.lr = 0.02 / 8
    cfg.default_hooks.logger.interval = 10


    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # We can also use tensorboard to log the training process
    cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

    #------------------------------------------------------
    config='custom.py'
    with open(config, 'w') as f:
        f.write(cfg.pretty_text)


cfg = Config.fromfile('config.py')

# register all modules in mmdet into the registries
register_all_modules()

output_dir = "/home/g/gajdosech2/Hamburg2024/work_dirs"
visualizer = VISUALIZERS.build(cfg.visualizer)
dataset = DATASETS.build(cfg.train_dataloader.dataset)

progress_bar = ProgressBar(len(dataset))
for item in dataset:
    img = item['inputs'].permute(1, 2, 0).numpy()
    data_sample = item['data_samples'].numpy()
    gt_instances = data_sample.gt_instances
    img_path = osp.basename(item['data_samples'].img_path)

    out_file = osp.join(
            output_dir,
            osp.basename(img_path)) if output_dir is not None else None

    img = img[..., [2, 1, 0]]  # bgr to rgb
    gt_bboxes = gt_instances.get('bboxes', None)
    if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
        gt_instances.bboxes = gt_bboxes.tensor
    gt_masks = gt_instances.get('masks', None)
    if gt_masks is not None:
        masks = mask2ndarray(gt_masks)
        gt_instances.masks = masks.astype(bool)
    gt_keypoints = gt_instances.get('keypoints', None)
    if gt_keypoints is not None:
        gt_instances.keypoints = gt_keypoints.tensor
    data_sample.gt_instances = gt_instances

    visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            wait_time=0,
            out_file=out_file)
    
    progress_bar.update()
