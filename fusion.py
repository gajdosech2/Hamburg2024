import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json

import os

ROOT_DIR = "/home/g/gajdosech2/"
ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")


def depth_to_pointcloud(depth_image, intrinsics):
    """Convert depth image to point cloud using camera intrinsics."""
    height, width = depth_image.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    
    # Generate pixel grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # Back-project to 3D space
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def apply_transformation(points, translation, rotation):
    """Apply inverse transformation to point cloud."""
    # Invert the rotation (quaternion inverse is the conjugate)
    r = R.from_quat(rotation).inv().as_matrix()
    
    # Inverse of translation is negating the translation vector
    neg_translation = -np.array(translation) * 1000
    
    # Apply inverse rotation
    rotated_points = points @ r.T
    
    # Apply inverse translation
    transformed_points = rotated_points + neg_translation
    return transformed_points

def create_open3d_pointcloud(points, color):
    """Create an Open3D point cloud with a specific color."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)  # Color the point cloud
    return pcd

def get_transformation_data(transform_data, sensor_name):
    """Extract transformation data for a specific sensor from the JSON."""
    translation = transform_data[sensor_name]["transform"]["translation"]
    rotation = transform_data[sensor_name]["transform"]["rotation"]
    
    translation_vector = [translation["x"], translation["y"], translation["z"]]
    rotation_quaternion = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    
    return translation_vector, rotation_quaternion


# Parse the JSON data
with open('data/transformations.json', 'r') as f:
    transform_data = json.load(f)

# Intrinsics (replace with your camera intrinsics)
intrinsics_left = {"fx": 912.7007, "fy": 913.1103, "cx": 653.6740, "cy": 365.5973}
intrinsics_right = {"fx": 909.7750, "fy": 909.3660, "cx": 648.5572, "cy": 386.9201}
intrinsics_top = {"fx": 914.0937, "fy": 914.0947, "cx": 649.8485, "cy": 370.4816}

# Load your depth images (replace with actual depth image loading)
depth_image_right = np.load('data/right_depth_img.npy')  # Example depth map for the right camera
depth_image_left = np.load('data/left_depth_img.npy')    # Example depth map for the left camera
depth_image_top = np.load('data/10chalded.npy')      # Example depth map for the top camera

# Get transformation data from JSON
translation_right, rotation_right = get_transformation_data(transform_data, "realsense_right_color_optical_frame")
translation_left, rotation_left = get_transformation_data(transform_data, "realsense_left_color_optical_frame")
translation_top, rotation_top = get_transformation_data(transform_data, "realsense_top_color_optical_frame")

# Convert depth images to point clouds
print("depth_to_pointclou")
pcd_right = depth_to_pointcloud(depth_image_right, intrinsics_right)
pcd_left = depth_to_pointcloud(depth_image_left, intrinsics_left)
pcd_top = depth_to_pointcloud(depth_image_top, intrinsics_top)

# Apply transformations
print("apply_transformation")
pcd_right_transformed = apply_transformation(pcd_right, translation_right, rotation_right)
pcd_left_transformed = apply_transformation(pcd_left, translation_left, rotation_left)
pcd_top_transformed = apply_transformation(pcd_top, translation_top, rotation_top)

# Create Open3D point clouds with different colors
print("create_open3d_pointcloud")
pcd_right_o3d = create_open3d_pointcloud(pcd_right_transformed, [1, 0, 0])  # Red for right
pcd_left_o3d = create_open3d_pointcloud(pcd_left_transformed, [0, 1, 0])   # Green for left
pcd_top_o3d = create_open3d_pointcloud(pcd_top_transformed, [0, 0, 1])     # Blue for top

# Combine the point clouds
combined_pcd = pcd_right_o3d + pcd_left_o3d + pcd_top_o3d

# Estimate normals for Poisson surface reconstruction
print("estimate_normals")
#combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Perform Poisson surface reconstruction
print("create_from_point_cloud_poisson")
#mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_pcd, depth=9)

# Visualize the reconstructed mesh
#o3d.visualization.draw_geometries([mesh])

axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])

# Combine and visualize
#o3d.visualization.draw_geometries([pcd_left_o3d, pcd_right_o3d, axis_gizmo])

o3d.visualization.draw_geometries([pcd_top_o3d, axis_gizmo])

