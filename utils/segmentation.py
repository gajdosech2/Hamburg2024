from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
from datetime import datetime


import sys
sys.path.append("/home/g/gajdosech2/segment-anything-2")
from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore


def filter_coords_yolo(coords, rgb_image, caps_image, circle=False):
    model = YOLO("yolov8l-worldv2.pt") 
    model.set_classes(["glass", ])
    results = model.predict(rgb_image, conf=0.001)
    results[0].save("work_dirs/debug/debug_yolo_glass.png")

    coords = filter_coords_within_boxes(coords, results)

    if circle:
        #model = YOLO("yolov8l-worldv2.pt") 
        model.set_classes(["green circle", ])
        results = model.predict(caps_image, conf=0.0003)
        results[0].save("work_dirs/debug/debug_yolo_circle.png")

        coords = filter_coords_within_boxes(coords, results)

    return coords



def filter_coords_within_boxes(coords, results):
    coords_copy = np.copy(coords)
    x_coords = coords_copy[:, 0]
    y_coords = coords_copy[:, 1]

    valid_points = np.zeros(len(coords_copy), dtype=bool)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        within_box = (x_coords >= x1) & (x_coords <= x2) & (y_coords >= y1) & (y_coords <= y2)
        valid_points |= within_box

    filtered_coords = coords_copy[valid_points]
    return filtered_coords


def mark_classes(coords, rgb_image, known_names):
    for u, v, index in coords:
        cv2.circle(rgb_image, (u, v), radius=3, color=(0, 255, 0), thickness=2)  
        cv2.putText(rgb_image, known_names[index], (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA, False)
    cv2.imwrite('work_dirs/classes/debug_classes' + datetime.utcnow().strftime('%H:%M:%S.%f') + ".png", rgb_image)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    return mask_image


def show_masks(image, masks, scores, borders=True):
    mask_image = None
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        mask_image = show_mask(mask, plt.gca(), borders=borders)
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        #plt.show()

    return mask_image

        
def segmentation_masks(root_dir, rgb_image, coords, centroids):
    sam2_checkpoint = root_dir + "/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda:1"))
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(rgb_image)

    all_masks = []
    for i, cap in enumerate(coords):
        masks, scores, _ = predictor.predict(
            point_coords=np.array([cap[:2], centroids[i][:2]]),
            point_labels=np.array([1, 1]),
            multimask_output=False,
        )

        all_masks.append(masks[0])
        mask_image = show_masks(rgb_image, masks, scores, borders=True)
        cv2.imwrite("work_dirs/debug/debug_mask" + str(i) + ".png", rgb_image + mask_image[:, :, :3] * 100000)

    return all_masks


def binary_mask_to_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # COCO requires `counts` to be a string
    return rle


def binary_mask_to_polygon_fragmented(binary_mask):
    area = int(np.sum(binary_mask))
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # Minimum 3 points (6 values) to form a polygon
            polygons.append(contour)

    return polygons, area


def binary_mask_to_polygon(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack(contours) 
    hull = cv2.convexHull(all_points)
    area = cv2.contourArea(hull)
    convex_polygon = hull.flatten().tolist()
    if len(convex_polygon) < 6: 
        return []
    return [convex_polygon], area 