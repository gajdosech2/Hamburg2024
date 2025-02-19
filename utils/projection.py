import sys
sys.path.append("/home/g/gajdosech2/Hamburg2024/utils/")
sys.path.append("/home/g/gajdosech2/miniconda3/envs/image_processor/lib/python3.10/dist-packages/")

import cv2
import json
import numpy as np
from nicol_image_processing.base.utils import convert_dict_to_tf
from nicol_image_processing.src. image_processor import ImageProcessor
from nicol_image_processing.base.datatypes import Point

from segmentation import segmentation_masks



def eye_views(coco_json, head_image_path, image_id, coords, plane_centroids, predictor):
    split_path = head_image_path.split("/")
    scene_id = split_path[-3].split("_")[1]
    view_id = split_path[-1].split(".")[0]

    right_eye_calib = '/home/g/gajdosech2/nicol_image_processing/calibration/SEEcam/right_SEEcam_dist_model.yaml'
    left_eye_calib = '/home/g/gajdosech2/nicol_image_processing/calibration/SEEcam/left_SEEcam_dist_model.yaml'
    head_calib = '/home/g/gajdosech2/nicol_image_processing/calibration/realsense/head_realsense_dist_model.yaml'

    camera_tfrm_json = f'/home/g/gajdosech2/Hamburg2024/dataset/scene_{scene_id}_caps/pose_transform_data.json'

    with open(camera_tfrm_json, "r") as jsonFile:
        camera_tfrm_dict = json.load(jsonFile)
    right_camera_tfrm = convert_dict_to_tf(camera_tfrm_dict[f'image_{view_id}']['right_eye_cam'])
    left_camera_tfrm = convert_dict_to_tf(camera_tfrm_dict[f'image_{view_id}']['left_eye_cam'])
    realsense_tfrm = convert_dict_to_tf(camera_tfrm_dict[f'image_{view_id}']['realsense_head_color_optical_frame'])

    left_eye_path = head_image_path.replace("head_frame_img", "left_eye")
    right_eye_path = head_image_path.replace("head_frame_img", "right_eye")

    realsense_img = cv2.imread(head_image_path)
    realsense_img = cv2.resize(realsense_img, (1920, 1080))

    left_image = cv2.imread(left_eye_path)
    if (left_image.shape[0] != 1440 or left_image.shape[1] != 1920):
        left_image = cv2.resize(left_image, (1920, 1440))
        cv2.imwrite(left_eye_path, left_image)

    right_image = cv2.imread(right_eye_path)
    if right_image.shape[0] != 1440 or right_image.shape[1] != 1920:
        right_image = cv2.resize(right_image, (1920, 1440))
        cv2.imwrite(right_eye_path, right_image)

    coco_json["images"].append({
        "id": image_id + 1,
        "file_name": left_eye_path,
        "width": left_image.shape[1],
        "height": left_image.shape[0] 
    })

    coco_json["images"].append({
        "id": image_id + 2,
        "file_name": right_eye_path,
        "width": right_image.shape[1],
        "height": right_image.shape[0] 
    })

    left_image_processor = ImageProcessor(calibration_filename=left_eye_calib)
    right_image_processor = ImageProcessor(calibration_filename=right_eye_calib)
    realsense_image_processor = ImageProcessor(calibration_filename=head_calib)
    left_vis_img = left_image.copy()
    right_vis_img = right_image.copy()

    left_coords = []
    right_coords = []
    for c in coords:
        c = c[:2]
        table_coord, _ = realsense_image_processor.pixel_to_table_coordinate(img=realsense_img, pixels=c * 3, camera_tfrm=realsense_tfrm, verbose=False)
        coordinate = Point(x=table_coord[0], y=table_coord[1], z=table_coord[2])

        left_pixel, _ = left_image_processor.coordinate_to_pixel(img=left_image, coordinate=coordinate, camera_tfrm=left_camera_tfrm, verbose=False)
        cv2.circle(left_vis_img, (int(left_pixel[0]), int(left_pixel[1])), radius=5, color=(0, 255, 0), thickness=4) 
        #cv2.imwrite('/home/g/gajdosech2/Hamburg2024/work_dirs/debug/debug_left_coord.png', left_vis_img)

        right_pixel, _ = right_image_processor.coordinate_to_pixel(img=right_image, coordinate=coordinate, camera_tfrm=right_camera_tfrm, verbose=False)
        cv2.circle(right_vis_img, (int(right_pixel[0]), int(right_pixel[1])), radius=5, color=(0, 255, 0), thickness=4) 
        cv2.imwrite('/home/g/gajdosech2/Hamburg2024/work_dirs/debug/debug_right_coord.png', right_vis_img)

        left_coords.append(np.array((int(left_pixel[0]//4), int(left_pixel[1]//4))))
        right_coords.append(np.array((int(right_pixel[0]//4), int(right_pixel[1]//4))))

    left_centroids = []
    right_centroids = []
    for i, c in enumerate(coords):
        p = np.array(plane_centroids[i][:2]).astype(np.float64)
        table_coord, _ = realsense_image_processor.pixel_to_table_coordinate(img=realsense_img, pixels=p * 3, camera_tfrm=realsense_tfrm, verbose=False)
        
        if np.isnan(table_coord).any():
            table_coord, _ = realsense_image_processor.pixel_to_table_coordinate(img=realsense_img, pixels=c[:2] * 3, camera_tfrm=realsense_tfrm, verbose=False)

        coordinate = Point(x=table_coord[0], y=table_coord[1], z=table_coord[2])

        left_pixel, _ = left_image_processor.coordinate_to_pixel(img=left_image, coordinate=coordinate, camera_tfrm=left_camera_tfrm, verbose=False)
        cv2.circle(left_vis_img, (int(left_pixel[0]), int(left_pixel[1])), radius=5, color=(255, 255, 0), thickness=4) 
        #cv2.imwrite('/home/g/gajdosech2/Hamburg2024/work_dirs/debug/debug_left_keypoint.png', left_vis_img)

        right_pixel, _ = right_image_processor.coordinate_to_pixel(img=right_image, coordinate=coordinate, camera_tfrm=right_camera_tfrm, verbose=False)
        cv2.circle(right_vis_img, (int(right_pixel[0]), int(right_pixel[1])), radius=5, color=(255, 255, 0), thickness=4) 
        cv2.imwrite('/home/g/gajdosech2/Hamburg2024/work_dirs/debug/debug_right_keypoint.png', right_vis_img)

        left_centroids.append(np.array((int(left_pixel[0]//4), int(left_pixel[1]//4))))
        right_centroids.append(np.array((int(right_pixel[0]//4), int(right_pixel[1]//4))))

    left_image = cv2.resize(left_image, (1920//4, 1440//4))
    right_image = cv2.resize(right_image, (1920//4, 1440//4))
    
    left_masks = segmentation_masks(left_image, left_coords, left_centroids, predictor)
    right_masks = segmentation_masks(right_image, right_coords, right_centroids, predictor)

    return left_masks, right_masks, left_centroids, right_centroids
        