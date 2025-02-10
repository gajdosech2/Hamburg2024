import mmcv
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from mmdet.registry import VISUALIZERS, DATASETS
from mmengine.runner import Runner
from mmengine.utils import ProgressBar
from mmengine.runner import set_random_seed
from mmdet.structures.bbox import BaseBoxes
from mmdet.models.utils import mask2ndarray
from mmdet.utils import register_all_modules
from mmdet.utils import setup_cache_size_limit_of_dynamo
import cv2
import os.path as osp
import os

ROOT_DIR = "/home/g/gajdosech2/"
#ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")

# Register all modules in mmdet into the registries
register_all_modules()

# Reduce the number of repeated compilations and improve training speed
setup_cache_size_limit_of_dynamo()

def modify_config():
    cfg = Config.fromfile(ROOT_DIR + '/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py')

    cfg.metainfo = {
        'classes': ("shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"),
        'palette': [
            (250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0), (250, 0, 250), (0, 250, 250)
        ]
    }

    # Modify dataset type and path
    cfg.data_root = ROOT_DIR + "/Hamburg2024/"

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
    cfg.model.bbox_head.num_classes = 6

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    cfg.load_from = 'rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'

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

    config='custom.py'
    with open(config, 'w') as f:
        f.write(cfg.pretty_text)

def visualize_gt(cfg):
    output_dir = ROOT_DIR + "/Hamburg2024/work_dirs/debug/"
    visualizer = VISUALIZERS.build(cfg.visualizer)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    progress_bar = ProgressBar(len(dataset))
    for i, item in enumerate(dataset):
        if i > 22:
            break
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        gt_instances = data_sample.gt_instances
        img_path = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
                output_dir,
                osp.basename(img_path)) if output_dir is not None else None

        #img = img[..., [2, 1, 0]]  # bgr to rgb
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
                draw_gt=True,
                draw_pred=False,
                wait_time=0,
                out_file=out_file)
        
        progress_bar.update()


def inference():
    checkpoint_file = ROOT_DIR + '/Hamburg2024/checkpoint_nokp.pth'
    model = init_detector(CONFIG_FILE, checkpoint_file, device='cpu') 
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    for i in range(25):
        print(i)
        image = mmcv.imread(ROOT_DIR + f'/Hamburg2024/dataset/scene_220_transparent/head_frame_img/{i}.png', channel_order='rgb')
        image = cv2.resize(image, (640, 360))
        result = inference_detector(model, image)

        heatmap = False
        if heatmap:
            heatmap = result.pred_instances.heatmap.detach().cpu().numpy()[0]
            heatmap_normalized = cv2.normalize((heatmap * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(f"work_dirs/pred/heatmap{i}.png", heatmap_colored)

            maxima_coords = []
            for j, (x, y, w, h) in enumerate(result.pred_instances.bboxes.detach().cpu().numpy()):
                if result.pred_instances.scores[j] > 0.3:
                    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
                    x_max, y_max = min(x + w, heatmap.shape[1]), min(y + h, heatmap.shape[0])       
                    roi = heatmap[y:y_max, x:x_max]
                    if roi.size > 0:           
                        max_idx = np.unravel_index(np.argmax(roi), roi.shape)
                        max_y, max_x = max_idx  
                        maxima_coords.append((x + max_x, y + max_y))     
            print(maxima_coords)
            
        class_names = ["shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass", "key_point"]
        result.pred_instances.bboxes
        print([class_names[i] for i in result.pred_instances.labels])
        print(result.pred_instances.scores)

        visualizer.add_datasample(
            'result',
            image,
            data_sample=result,
            draw_gt = False,
            wait_time=0,
            pred_score_thr=0.3
        )

        img = visualizer.get_image()
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        cv2.imwrite(f"work_dirs/pred/prediction{i}.png", img)


CONFIG_FILE = 'config.py'
cfg = Config.fromfile(CONFIG_FILE)
cfg.work_dir = "work_dirs/"

#visualize_gt(cfg)

#Runner.from_cfg(cfg).train()

inference()