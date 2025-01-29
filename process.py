
import open3d as o3d
import numpy as np
import os
import json
import cv2
from PIL import Image

ROOT_DIR = "/home/g/gajdosech2/"
#ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")

from utils.segmentation import *
from utils.geometry import *
from utils.thresholding import *


def add_coco_detections(all_masks, image_id, coords, plane_centroids, coco_json):
    global ANNOTATION_ID
    for mask_id, mask in enumerate(all_masks):
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (1280, 720), interpolation = cv2.INTER_NEAREST)
        # Create bounding box from mask
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

        if w > 350: #bad mask
            continue

        polygons = binary_mask_to_polygon(mask)

        # Add annotation data for the object
        coco_json["annotations"].append({
                "id": ANNOTATION_ID,
                "image_id": image_id,
                "category_id": int(coords[mask_id][2] + 1),
                "bbox": [x, y, w, h],
                "segmentation": polygons,
                "area": int(np.sum(mask)),  # Area is the number of pixels in the mask
                "iscrowd": 0
        })

        kp_x = plane_centroids[mask_id][0] * 2
        kp_y = plane_centroids[mask_id][1] * 2
        kp_segm = [kp_x - 15, kp_y - 15, 
                    kp_x + 15, kp_y - 15, 
                    kp_x + 15, kp_y + 15, 
                    kp_x - 15, kp_y + 15, ]
        # Add keypoint annotations for the object
        #coco_json["annotations"].append({
        #        "id": 1000000 + ANNOTATION_ID,
        #        "image_id": image_id,
        #        "category_id": len(KNOWN_NAMES) + 1,
        #        "bbox": [kp_x - 15, kp_y - 15, 30, 30],
        #        "segmentation": [kp_segm],
        #        "area": 30 * 30,
        #        "iscrowd": 0
        #})

        ANNOTATION_ID += 1


def process_dataset():
    # Initialize COCO JSON structure
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
    green_circle = False

    for image_id, (rgb_image_path, depth_image_path, caps_image_path) in enumerate(zip(RGB_PATHS, DEPTH_PATHS, CAPS_PATHS)):
        # Load RGB and depth images
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
        cv2.imwrite("work_dirs/debug/debug_caps.png", caps_image)

        depth_array = np.load(depth_image_path)
        depth_array = cv2.resize(depth_array, (640, 360), interpolation = cv2.INTER_NEAREST)
        depth_normalized = (depth_array-np.min(depth_array))/(np.max(depth_array)-np.min(depth_array))
        #cv2.imwrite("work_dirs/debug/debug_depth.png", depth_normalized * 255)

        # Add image metadata to COCO JSON
        coco_json["images"].append({
            "id": image_id,
            "file_name": rgb_image_path,  # Store the file name, not the full path
            "width": rgb_image.shape[1] * 2,  # Image width
            "height": rgb_image.shape[0] * 2  # Image height
        })

        pc = depth_2_pc(FX, FY, CX, CY, depth_array)
        pc = np.swapaxes(pc, 0, 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        a, b, c, d, inliers = fit_plane(pcd)
        r = find_rotation(a, b, c)
        #pcd.rotate(r, center=(0, 0, 0))

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([0, 0, 1]) 

        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        labels, blob_heights = threshold_blobs(outlier_cloud, a, b, c, d)
        if labels.shape[0] == 0:
            continue
 
        if "scene_121_" in caps_image_path:
            cap_color = PLASTIC_CAP
            dist_threshold = 80
            green_circle = True

        coords = pixel_coordinates(outlier_cloud, labels, blob_heights, caps_image.copy(), cap_color, dist_threshold, cam_matrix, KNOWN_HEIGHTS)
        if len(coords) == 0:
            continue

        coords = filter_coords_yolo(coords, rgb_image, caps_image, green_circle)
        if len(coords) == 0:
            continue

        for i, (u, v, index, _, _, _) in enumerate(coords):
            print(f"Blob {i}: Pixel coordinates: ({u}, {v}), Name: {KNOWN_NAMES[int(index)]}")

        plane_centroids = project_points_to_plane([a, b, c, d], coords[:, 3:], rgb_image.copy(), cam_matrix)
        transformed_meshes = place_models(plane_centroids, a, b, c)
        all_masks = segmentation_masks(ROOT_DIR, rgb_image, coords, plane_centroids)

        axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axis_gizmo] + transformed_meshes)
        
        add_coco_detections(all_masks, image_id, coords, plane_centroids, coco_json)
        mark_classes(coords[:, :3].astype(int), rgb_image, KNOWN_NAMES)

    with open('coco_annotations_small.json', 'w') as f:
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

#DEPTH_PATHS = ["data/11.npy"]
#RGB_PATHS = ["data/11.png"]
DEPTH_PATHS = []
RGB_PATHS = []
CAPS_PATHS = []

SCENES_COUNT = 218
ANNOTATION_ID = 1

for j in range(1):
    if not os.path.isdir(f"dataset/scene_{j+1}_caps/"):
        continue
    if j+1 == 6: #shift scene
        continue
    if j+1 == 30: #val scene
        continue
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j+1}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j+1}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j+1}_caps/head_frame_img/{i}.png")

process_dataset()