import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_util
import os
import json
import cv2
os.chdir("/export/home/gajdosec/Hamburg2024")

import sys
sys.path.append("/export/home/gajdosec/segment-anything-2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


def threshold_blobs(outlier_cloud):
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=1000, print_progress=True))

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


def pixel_coordinates(outlier_cloud, blob_heights, known_heights, known_names):
    K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0,  0,  1]])

    R_inv = np.linalg.inv(r)
    coordinates = []

    for i, height in blob_heights:
        height = height * 100
        index = -1
        for j, known_height in enumerate(known_heights):
            if height > known_height - 1.3 and height < known_height + 1.3:
                index = j
                break

        if index == -1:
            continue

        # Get the points belonging to this blob
        cluster_points = np.asarray(outlier_cloud.select_by_index(np.where(labels == i)[0]).points)
        
        if len(cluster_points) == 0:
            continue
        
        # Calculate the centroid of the blob (center of gravity)
        centroid = np.mean(cluster_points, axis=0)
        original_centroid = np.dot(R_inv, centroid)
        
        # Extract the 3D coordinates of the original centroid (X, Y, Z)
        X, Y, Z = original_centroid
        
        # Project the 3D centroid to 2D pixel coordinates using the intrinsic matrix
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]
        
        # Store the pixel coordinates (u, v) and the corresponding 3D centroid
        coordinates.append((int(u), int(v)-50, index))

    for i, (u, v, index) in enumerate(coordinates):
        print(f"Blob {i}: Pixel coordinates: ({u}, {v}), Name: {known_names[index]}")

    return np.array(coordinates)


def mark_classes(coords, rgb_image, known_names):
    # Iterate through the list of pixel coordinates (u, v) and draw circles on the image
    for i, (u, v, index) in enumerate(coords):
        # Draw a circle on the RGB image at (u, v)
        # Parameters: image, center_coordinates, radius, color (BGR format), thickness
        cv2.circle(rgb_image, (u, v), radius=5, color=(0, 255, 0), thickness=2)  # Green circles with radius 5
        cv2.putText(rgb_image, known_names[index], (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA, False)

    # Save the modified image to a file (replace 'output_image_path' with your desired output path)
    cv2.imwrite('modified_rgb_image.png', rgb_image)


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
        # plt.imshow(image)
        mask_image = show_mask(mask, plt.gca(), borders=borders)
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        # plt.show()

    return mask_image


def segmentation_masks(rgb_image, coords):
    sam2_checkpoint = "/export/home/gajdosec/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"))
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(rgb_image)

    all_masks = []
    for i, single in enumerate(coords):
        masks, scores, _ = predictor.predict(
            point_coords=np.array([single[:2]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

        all_masks.append(masks[0])
        mask_image = show_masks(rgb_image, masks, scores, borders=True)
        cv2.imwrite("mask" + str(i) + ".png", rgb_image + mask_image[:, :, :3] * 10000)

    return all_masks


def binary_mask_to_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # COCO requires `counts` to be a string
    return rle

    
KNOWN_HEIGHTS = [8.0, 13.5, 19.0]
KNOWN_NAMES = ["shot_glass", "beer_glass", "wine_glass"]
FX = FY = 525.0  # Focal length 
CX = 319.5       # Principal point (x-coordinate)
CY = 239.5       # Principal point (y-coordinate)

# Initialize COCO JSON structure
coco_json = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "shot_glass"}, {"id": 2, "name": "beer_glass"}, {"id": 3, "name": "wine_glass"}]  # Add more categories if needed
}

# Placeholder for annotation ID
annotation_id = 1

depth_paths = ["data/10.npy"]
rgb_paths = ["data/10.png"]


for image_id, (rgb_image_path, depth_image_path) in enumerate(zip(rgb_paths, depth_paths)):
    # Load RGB and depth images

    rgb_image = Image.open(rgb_image_path)
    rgb_image = np.array(rgb_image)
    rgb_image = cv2.resize(rgb_image, (640, 360)) 
    cv2.imwrite("rgb.png", rgb_image)

    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Ensure depth is loaded properly
    depth_array = np.load(depth_image_path)
    depth_normalized = (depth_array-np.min(depth_array))/(np.max(depth_array)-np.min(depth_array))
    cv2.imwrite("depth.png", depth_normalized * 255)

    # Add image metadata to COCO JSON
    coco_json["images"].append({
        "id": image_id,
        "file_name": rgb_image_path,  # Store the file name, not the full path
        "width": rgb_image.shape[1],  # Image width
        "height": rgb_image.shape[0]  # Image height
    })

    pc = depth_2_pc(FX, FY, CX, CY, depth_array)
    pc = np.swapaxes(pc, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    a, b, c, d, inliers = fit_plane(pcd)
    r = find_rotation(a, b, c)

    pcd.rotate(r, center=(0, 0, 0))

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 0, 1]) 

    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    labels, blob_heights = threshold_blobs(outlier_cloud)

    axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, axis_gizmo])

    coords = pixel_coordinates(outlier_cloud, blob_heights, KNOWN_HEIGHTS, KNOWN_NAMES)
    all_masks = segmentation_masks(rgb_image, coords)

    for mask_id, mask in enumerate(all_masks):
        mask = mask.astype(np.uint8)
        # Create bounding box from mask
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

        # Convert mask to RLE format for COCO
        rle_mask = binary_mask_to_rle(mask)

        # Add annotation data for the object
        coco_json["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(coords[mask_id][2] + 1),  # Assuming single category, you can extend for multiple categories
            "bbox": [x, y, w, h],
            "segmentation": rle_mask,
            "area": int(np.sum(mask)),  # Area is the number of pixels in the mask
            "iscrowd": 0
        })

        # Increment annotation ID
        annotation_id += 1

    mark_classes(coords, rgb_image, KNOWN_NAMES)

# Save COCO JSON to file
with open('coco_annotations.json', 'w') as f:
    json.dump(coco_json, f)



   
