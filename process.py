
import open3d as o3d
import numpy as np
import os
import json
import cv2
from PIL import Image

ROOT_DIR = "/home/g/gajdosech2/"
os.chdir(ROOT_DIR + "/Hamburg2024")

from utils.segmentation import *
from utils.geometry import *
from utils.thresholding import *


def add_coco_detections(coco_json, all_masks, image_id, coords, plane_centroids, mask_shape, width_threshold, keypoints):
    global ANNOTATION_ID
    for mask_id, mask in enumerate(all_masks):
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, mask_shape, interpolation = cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

        if w > width_threshold or w < 10: #Bad mask
            continue

        polygons, area = binary_mask_to_polygon(mask)

        coco_json["annotations"].append({
                "id": ANNOTATION_ID,
                "image_id": image_id,
                "category_id": int(coords[mask_id][2] + 1),
                "bbox": [x, y, w, h],
                "segmentation": polygons,
                "area": area,
                "iscrowd": 0
        })

        kp_x = plane_centroids[mask_id][0] * 2
        kp_y = plane_centroids[mask_id][1] * 2
        kp_segm = [kp_x - 15, kp_y - 15, 
                    kp_x + 15, kp_y - 15, 
                    kp_x + 15, kp_y + 15, 
                    kp_x - 15, kp_y + 15, ]
        
        if keypoints:
            coco_json["annotations"].append({
                    "id": 1000000 + ANNOTATION_ID,
                    "image_id": image_id,
                    "category_id": len(KNOWN_NAMES) + 1,
                    "bbox": [kp_x - 15, kp_y - 15, 30, 30],
                    "segmentation": [kp_segm],
                    "area": 30 * 30,
                    "iscrowd": 0
            })

        ANNOTATION_ID += 1


def process_dataset(json_name):
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "shot_glass"}, 
                       {"id": 2, "name": "whisky_glass"}, 
                       {"id": 3, "name": "water_glass"}, 
                       {"id": 4, "name": "beer_glass"}, 
                       {"id": 5, "name": "wine_glass"},
                       {"id": 6, "name": "high_glass"},
                       {"id": 7, "name": "key_point"}] 
    }

    cam_matrix = np.array([[FX, 0, CX], [0, FY, CY], [0,  0,  1]])
    cap_color = WOOD_CAP
    dist_threshold = 150
    circle = False

    for image_id, (rgb_image_path, depth_image_path, caps_image_path) in enumerate(zip(RGB_PATHS, DEPTH_PATHS, CAPS_PATHS)):
        print(rgb_image_path)
        rgb_image = Image.open(rgb_image_path)
        rgb_image = np.array(rgb_image)
        rgb_image = cv2.resize(rgb_image, (640, 360))
        if not ("scene_1_" in rgb_image_path or "scene_2_" in rgb_image_path):
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
        #cv2.imwrite("work_dirs/debug/debug_rgb.png", rgb_image)

        caps_image = Image.open(caps_image_path)
        caps_image = np.array(caps_image)
        caps_image = cv2.resize(caps_image, (640, 360))
        if not ("scene_1_" in caps_image_path or "scene_2_" in caps_image_path):
            caps_image = cv2.cvtColor(caps_image, cv2.COLOR_RGB2BGR) 
        #cv2.imwrite("work_dirs/debug/debug_caps.png", caps_image)

        depth_array = np.load(depth_image_path)
        depth_array = cv2.resize(depth_array, (640, 360), interpolation = cv2.INTER_NEAREST)
        #depth_normalized = (depth_array-np.min(depth_array))/(np.max(depth_array)-np.min(depth_array))
        #cv2.imwrite("work_dirs/debug/debug_depth.png", depth_normalized * 255)

        coco_json["images"].append({
            "id": image_id,
            "file_name": rgb_image_path,
            "width": rgb_image.shape[1] * 2,
            "height": rgb_image.shape[0] * 2 
        })

        pc = depth_2_pc(FX, FY, CX, CY, depth_array)
        pc = np.swapaxes(pc, 0, 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        a, b, c, d, inliers = fit_plane(pcd)
        #r = find_rotation(a, b, c)
        #pcd.rotate(r, center=(0, 0, 0))

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([0, 0, 1]) 

        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        labels, blob_heights = threshold_blobs(outlier_cloud, a, b, c, d)
        if labels.shape[0] == 0:
            continue

        scene_number = int(caps_image_path.split("/")[-3].split("_")[1])
        if scene_number > 100:
            cap_color = PLASTIC_CAP
            dist_threshold = 45
            circle = True

        if scene_number > 300:
            circle = False

        coords = pixel_coordinates(outlier_cloud, labels, blob_heights, caps_image.copy(), cap_color, dist_threshold, cam_matrix, KNOWN_HEIGHTS)
        if len(coords) == 0:
            continue

        coords = filter_coords_yolo(coords, rgb_image, caps_image, circle)
        if len(coords) == 0:
            continue

        for i, (u, v, index, _, _, _) in enumerate(coords):
            print(f"Blob {i}: Pixel coordinates: ({u}, {v}), Name: {KNOWN_NAMES[int(index)]}")

        # CREATE JSON CONTAINING EYES AND LEFT AND RIGHT VIEWS, based on reprojecting coords, it should contain the following:
        # coco_json["images"].append info about the new rgb frams
        # all_masks = segmentation_masks(ROOT_DIR, new_rgb_image, new_coords, new_centroids) use same coords for centroids
        # add_coco_detections(all_masks, image_id, coords, plane_centroids, coco_json, (???, ???), ???, False) eye frames will use different resolution

        plane_centroids = project_points_to_plane([a, b, c, d], coords[:, 3:], rgb_image.copy(), cam_matrix)
        transformed_meshes = place_models(plane_centroids, a, b, c)
        all_masks = segmentation_masks(ROOT_DIR, rgb_image, coords, plane_centroids)

        axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axis_gizmo] + transformed_meshes)
        
        add_coco_detections(coco_json, all_masks, image_id, coords, plane_centroids, (1280, 720), 340, KEYPOINTS)
        #mark_classes(coords[:, :3].astype(int), rgb_image, KNOWN_NAMES)

    with open(json_name, 'w') as f:
        json.dump(coco_json, f)


    
KNOWN_HEIGHTS = [5.0, 9.0, 11.0, 13.0, 17.5, 22.0]
KNOWN_NAMES = ["shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"]
WOOD_CAP = [130, 160, 190]
PLASTIC_CAP = [60, 190, 80]

FX = 964.276709 / 2.0
FY = 968.782545 / 2.0
CX = 629.718065 / 2.0
CY = 393.747072 / 2.0

FX = 914.0937 / 2.0
FY = 914.0947 / 2.0
CX = 649.8485 / 2.0
CY = 370.4816 / 2.0

DEPTH_PATHS, RGB_PATHS, CAPS_PATHS = [], [], []
ANNOTATION_ID = 1
VAL_SCENES = [30, 145, 220, 325, 335]
SCENES_COUNT = 341

KEYPOINTS = True

scenes_count = 0
for j in range(SCENES_COUNT):
    if not os.path.isdir(f"dataset/scene_{j+1}_caps/"):
        continue
    if j+1 == 6: # Shifted scene
        continue
    if j+1 in VAL_SCENES:
        continue
    scenes_count += 1 
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j+1}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j+1}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j+1}_caps/head_frame_img/{i}.png")

print(f"Scene Count: {scenes_count + len(VAL_SCENES)}")
process_dataset('coco_annotations_train_data_3_frag_wkp.json')

DEPTH_PATHS, RGB_PATHS, CAPS_PATHS = [], [], []

for j in VAL_SCENES:
    if not os.path.isdir(f"dataset/scene_{j}_caps/"):
        continue
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j}_caps/head_frame_img/{i}.png")

process_dataset('coco_annotations_val_data_3_frag_wkp.json')


############################################################################################################
DEPTH_PATHS, RGB_PATHS, CAPS_PATHS = [], [], []
ANNOTATION_ID = 1
KEYPOINTS = False

for j in range(SCENES_COUNT):
    if not os.path.isdir(f"dataset/scene_{j+1}_caps/"):
        continue
    if j+1 == 6: # Shifted scene
        continue
    if j+1 in VAL_SCENES:
        continue
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j+1}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j+1}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j+1}_caps/head_frame_img/{i}.png")

process_dataset('coco_annotations_train_data_3_frag_nokp.json')

DEPTH_PATHS, RGB_PATHS, CAPS_PATHS = [], [], []

for j in VAL_SCENES:
    if not os.path.isdir(f"dataset/scene_{j}_caps/"):
        continue
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j}_caps/head_frame_img/{i}.png")

process_dataset('coco_annotations_val_data_3_frag_nokp.json')