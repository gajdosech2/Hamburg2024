# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Tuple
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm
from mmcv.ops import batched_nms
from mmengine.model import (BaseModule, bias_init_with_prob, constant_init,
                            normal_init)
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean)
from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, distance2bbox, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from mmdet.models.dense_heads import RTMDetInsHead, RTMDetInsSepBNHead
from mmdet.models.dense_heads.rtmdet_ins_head import MaskFeatModule

from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils.misc import unmap


@MODELS.register_module()
class RTMDetHeatInsSepBNHead(RTMDetInsSepBNHead):

    def _bbox_mask_post_process(
            self,
            results: InstanceData,
            mask_feat,
            cfg: ConfigType,
            rescale: bool = False,
            with_nms: bool = True,
            img_meta: Optional[dict] = None) -> InstanceData:
        """bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        stride = self.prior_generator.strides[0][0]
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        assert with_nms, 'with_nms must be True for RTMDet-Ins'
        if results.bboxes.numel() > 0:
            kernel_size = 15
            k = cv2.getGaussianKernel(kernel_size, 0)
            kernel = np.outer(k, k)

            cid = 6
            proposals = results.bboxes[results.labels == cid]
            scores = results.scores[results.labels == cid].detach().cpu().numpy()
            bounding_boxes = proposals.detach().cpu().numpy()

            image_size = img_meta["ori_shape"]
            heatmap = np.zeros(image_size, dtype=np.float64)

            for i in range(len(bounding_boxes)):
                left_top_x, left_top_y, bottom_right_x, bottom_right_y = bounding_boxes[i]
                # Calculate the center of the bounding box
                center_x = int((left_top_x + bottom_right_x) // 2)
                center_y = int((left_top_y + bottom_right_y) // 2)

                # Calculate slicing indices for placing the Gaussian
                x_start = max(0, center_x - kernel_size // 2)
                y_start = max(0, center_y - kernel_size // 2)
                x_end = min(image_size[1], center_x + kernel_size // 2 + 1)
                y_end = min(image_size[0], center_y + kernel_size // 2 + 1)

                # Determine the part of the kernel to be placed within the heatmap bounds
                kernel_x_start = 0 if x_start == center_x - kernel_size // 2 else kernel_size // 2 - center_x
                kernel_y_start = 0 if y_start == center_y - kernel_size // 2 else kernel_size // 2 - center_y
                kernel_x_end = kernel_size if x_end == center_x + kernel_size // 2 + 1 else kernel_size // 2 + (x_end - center_x) + 1
                kernel_y_end = kernel_size if y_end == center_y + kernel_size // 2 + 1 else kernel_size // 2 + (y_end - center_y) + 1

                try:
                    # Add the Gaussian kernel to the heatmap
                    heatmap[y_start:y_end, x_start:x_end] += kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end] * scores[i]
                except:
                    continue

            heatmap_tensor = torch.zeros(1, heatmap.shape[0], heatmap.shape[1], device=results.bboxes.device)
            heatmap_tensor[0] = torch.from_numpy(heatmap).float().to(results.bboxes.device)
            heatmap_tensor = heatmap_tensor.repeat(len(results.bboxes), 1, 1)
            results.heatmap = heatmap_tensor
            

            results = results[results.labels < cid]

            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(
                mask_feat, results.kernels, results.priors)

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')
            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_logits = F.interpolate(
                    mask_logits,
                    size=[
                        math.ceil(mask_logits.shape[-2] * scale_factor[0]),
                        math.ceil(mask_logits.shape[-1] * scale_factor[1])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :ori_h, :ori_w]
            masks = mask_logits.sigmoid().squeeze(0)
            masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta['ori_shape'][:2] if rescale else img_meta[
                'img_shape'][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device)

        return results