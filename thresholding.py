import torch
from ultralytics import YOLO
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_util
import os
import json
import cv2
from datetime import datetime

ROOT_DIR = "/home/g/gajdosech2/"
#ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")

import sys
sys.path.append(ROOT_DIR + "/segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def depth_2_pc_distortion(fx, fy, cx, cy, d, depth_array):
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    h, w = depth_array.shape
    
    # Generate pixel grid
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([i, j], axis=-1).reshape(-1, 2).astype(np.float32)

    # Undistort the pixel coordinates
    undistorted_pixels = cv2.undistortPoints(pixels, k, d, None, k).reshape(h, w, 2)
    
    # Convert undistorted pixels to normalized coordinates
    x_normalized = (undistorted_pixels[..., 0] - k[0, 2]) / k[0, 0]
    y_normalized = (undistorted_pixels[..., 1] - k[1, 2]) / k[1, 1]
    
    # Convert to 3D points
    z = depth_array / 1000.0
    x = x_normalized * z
    y = y_normalized * z
    
    # Stack into an Nx3 point cloud
    point_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    return point_cloud


def depth_2_pc(fx, fy, cx, cy, depth_array):
    x_points, y_points, z_points = [], [], []
    height, width = depth_array.shape
    for v in range(height):
        for u in range(width):
            z = depth_array[v, u] / 1000.0  # Convert to meters
            if z == 0:  # Skip invalid points
                continue
            if z > 1:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
    return np.array([np.array(x_points), np.array(y_points), np.array(z_points)])


def fit_plane(pcd):
    # Plane fitting using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)

    # Extract the plane model coefficients
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return a, b, c, d, inliers


def find_rotation(a, b, c):
    # Normal of the detected plane
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)  # Normalize the vector

    # Target normal (Z-axis)
    z_axis = np.array([0, 0, 1])

    # Calculate rotation axis (cross product) and angle (dot product)
    rotation_axis = np.cross(plane_normal, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the axis

    angle = np.arccos(np.dot(plane_normal, z_axis))

    # Construct the rotation matrix using Rodrigues' rotation formula
    return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)


def threshold_blobs(outlier_cloud, a, b, c, d):
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=500, print_progress=True))

    if labels.shape[0] == 0:
        return np.array([]), np.array([])

    # Number of clusters
    max_label = labels.max()

    # Visualize the clustered point cloud (each cluster will have a different color)
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Assign black to noise points (label = -1)
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    blob_heights = []

    # Iterate over each cluster (blob)
    for i in range(max_label + 1):
        # Select points belonging to the current cluster
        cluster_cloud = outlier_cloud.select_by_index(np.where(labels == i)[0])
        
        # Fit a plane to the blob (cluster) points
        blob_plane_model, _ = cluster_cloud.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        [_, _, _, d_blob] = blob_plane_model
        
        # Compute the distance between the table plane and the blob plane
        # Distance between two parallel planes: |d2 - d1| / sqrt(a^2 + b^2 + c^2)
        distance = abs(d_blob - d) / np.sqrt(a**2 + b**2 + c**2)
        blob_heights.append((i, distance))

    # Print blob heights
    for i, height in blob_heights:
        print(f"Blob {i} height from table plane: {height*100:.4f} centimeters")

    return labels, blob_heights


def pixel_coordinates(outlier_cloud, labels, blob_heights, r, caps_image):
    k = np.array([[FX, 0, CX], [0, FY, CY], [0,  0,  1]])

    #r_inv = np.linalg.inv(r)
    coordinates = []

    for i, height in blob_heights:
        height = height * 100
        index = -1
        best = None
        for j, known_height in enumerate(KNOWN_HEIGHTS):
            current = abs(known_height - height)
            if current < 2: 
                if best == None or current < best:
                    best = current
                    index = j

        if index == -1:
            continue

        # Get the points belonging to this blob
        cluster_points = np.asarray(outlier_cloud.select_by_index(np.where(labels == i)[0]).points)
        
        if len(cluster_points) == 0:
            continue
        
        # Calculate the centroid of the blob (center of gravity)
        centroid = np.mean(cluster_points, axis=0)
        #original_centroid = np.dot(r_inv, centroid)
        original_centroid = centroid
        
        # Extract the 3D coordinates of the original centroid (X, Y, Z)
        x, y, z = original_centroid
        
        # Project the 3D centroid to 2D pixel coordinates using the intrinsic matrix
        u = (k[0, 0] * x / z) + k[0, 2]
        v = (k[1, 1] * y / z) + k[1, 2]

        # Check the color
        pixel_color = caps_image[int(v), int(u)]
        color_distance = np.linalg.norm(CAP_COLOR - pixel_color)

        cv2.circle(caps_image, (int(u), int(v)), radius=3, color=(0, 255, 0), thickness=2)  # Green circles with radius 3
        cv2.imwrite('work_dirs/debug/debug_circles.png', caps_image)

        if color_distance > 150:
            continue

        if pixel_color[0] > pixel_color[1] or pixel_color[1] > pixel_color[2]:
            continue
        
        # Store the pixel coordinates (u, v) and the corresponding 3D centroid
        coordinates.append((int(u), int(v), index))

    for i, (u, v, index) in enumerate(coordinates):
        print(f"Blob {i}: Pixel coordinates: ({u}, {v}), Name: {KNOWN_NAMES[index]}")

    return np.array(coordinates)


import numpy as np

def filter_coords_within_boxes(coords, rgb_image):
    if len(coords) == 0:
        return coords

    model = YOLO("yolov8l-worldv2.pt")  # or choose yolov8m/l-world.pt
    model.set_classes(["glass", ])
    results = model.predict(rgb_image, conf=0.001)
    results[0].save("work_dirs/debug/debug_yolo.png")

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
    # Iterate through the list of pixel coordinates (u, v) and draw circles on the image
    for i, (u, v, index) in enumerate(coords):
        # Draw a circle on the RGB image at (u, v)
        cv2.circle(rgb_image, (u, v), radius=3, color=(0, 255, 0), thickness=2)  # Green circles with radius 3
        cv2.putText(rgb_image, known_names[index], (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA, False)

    cv2.imwrite('work_dirs/debug/debug_classes' + datetime.utcnow().strftime('%H:%M:%S.%f') + ".png", rgb_image)


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
        # plt.show()

    return mask_image


def segmentation_masks(rgb_image, coords):
    sam2_checkpoint = ROOT_DIR + "/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"))
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(rgb_image)

    all_masks = []
    for i, single in enumerate(coords):
        masks, scores, _ = predictor.predict(
            point_coords=np.array([single[:2] - [3, 3], ]),
            point_labels=np.array([1, ]),
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


def binary_mask_to_polygon(binary_mask):
    # Ensure the mask is uint8 (required for findContours)
    binary_mask = binary_mask.astype(np.uint8)
    # Find contours in the mask (external contours only)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Flatten the contour array and convert it to a list of x, y coordinates
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # Minimum 3 points (6 values) to form a polygon
            polygons.append(contour)

    return polygons

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
                       {"id": 6, "name": "high_glass"}]  # Add more categories if needed
    }

    annotation_id = 1

    for image_id, (rgb_image_path, depth_image_path, caps_image_path) in enumerate(zip(RGB_PATHS, DEPTH_PATHS, CAPS_PATHS)):
        # Load RGB and depth images
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

        axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axis_gizmo])

        coords = pixel_coordinates(outlier_cloud, labels, blob_heights, r, caps_image.copy())

        # TO-DO DISCARD NON-GLASS BLOBS BASED ON ZERO-SHOT NETWORK
        coords = filter_coords_within_boxes(coords, rgb_image)

        # TO-DO MULTIPLE PIXELS PER BLOB FOR BETTER SEGMENTATION
        all_masks = segmentation_masks(rgb_image, coords)

        for mask_id, mask in enumerate(all_masks):
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (1280, 720), interpolation = cv2.INTER_NEAREST)
            # Create bounding box from mask
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

            if w > 350: #bad mask
                continue

            # Convert mask to format for COCO
            # rle_mask = binary_mask_to_rle(mask)
            polygons = binary_mask_to_polygon(mask)

            # Add annotation data for the object
            coco_json["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(coords[mask_id][2] + 1),
                "bbox": [x, y, w, h],
                "segmentation": polygons,
                "area": int(np.sum(mask)),  # Area is the number of pixels in the mask
                "iscrowd": 0
            })

            # TO-DO ADD ANNOTATION FOR THE TOP CAP (POUR AFFORDANCE CLASS)
            # Increment annotation ID
            annotation_id += 1

        mark_classes(coords, rgb_image, KNOWN_NAMES)

    # Save COCO JSON to file
    with open('coco_annotations_new.json', 'w') as f:
        json.dump(coco_json, f)


    
KNOWN_HEIGHTS = [6.0, 9.0, 11.0, 13.0, 17.5, 22.0]
KNOWN_NAMES = ["shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"]
CAP_COLOR = [130, 160, 190]

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

SCENES_COUNT = 35

for j in range(SCENES_COUNT):
    if j+1 == 6: #shift scene
        continue
    if j+1 == 30: #val scene
        continue
    for i in range(25):
        DEPTH_PATHS.append(f"dataset/scene_{j+1}_caps/head_depth_img/{i}.npy")
        RGB_PATHS.append(f"dataset/scene_{j+1}_transparent/head_frame_img/{i}.png")
        CAPS_PATHS.append(f"dataset/scene_{j+1}_caps/head_frame_img/{i}.png")

process_dataset()