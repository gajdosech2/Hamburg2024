import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import os

ROOT_DIR = "/home/g/gajdosech2/"
ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")

from utils.camera_models import PinholeCameraModel


def create_point_cloud_from_depth_and_rgb(depth_image, rgb_image, intrinsics):
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720, fx=intrinsics["fx"], fy=intrinsics["fy"], cx=intrinsics["cx"], cy=intrinsics["cy"]
    )



    h, w = depth_image.shape
    
    # Generate pixel grid
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([i, j], axis=-1).reshape(-1, 2).astype(np.float32)

    camera_instance = PinholeCameraModel()

    ray_vectors, ray_origin = camera_instance.cast_pixel_rays(pixels)


    points_3D = ray_origin + depth_image * ray_vectors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)

    # Create a point cloud from the depth image
    #pcd = o3d.geometry.PointCloud.create_from_depth_image(
    #    depth_image,
    #    intrinsics,
    #    depth_scale=1000.0, 
    #    depth_trunc=3.0,
    #    stride=1
    #)

    # Get pixel coordinates for each 3D point
    points_2d = np.asarray(pcd.points)

    # Project points back to image coordinates
    width, height = rgb_image.shape[1], rgb_image.shape[0]
    fx, fy = intrinsics.intrinsic_matrix[0, 0], intrinsics.intrinsic_matrix[1, 1]
    cx, cy = intrinsics.intrinsic_matrix[0, 2], intrinsics.intrinsic_matrix[1, 2]

    # Convert 3D point cloud to 2D pixel coordinates
    u = np.round((fx * points_2d[:, 0] / points_2d[:, 2]) + cx).astype(int)
    v = np.round((fy * points_2d[:, 1] / points_2d[:, 2]) + cy).astype(int)

    # Filter valid points within image bounds
    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid_mask], v[valid_mask]
    points_2d = points_2d[valid_mask]

    # Get the corresponding RGB values for each valid point
    colors = rgb_image[v, u] / 255.0  # Normalize RGB values

    # Assign colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points_2d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def apply_transformation(pcd, translation, rotation):
    """Apply inverse transformation to point cloud."""
    r = R.from_quat(rotation).as_matrix()

    T = np.zeros((4, 4))

    T[0, 0] = r[0, 0]
    T[0, 1] = r[0, 1]
    T[0, 2] = r[0, 2]
    T[0, 3] = translation[0]

    T[1, 0] = r[1, 0]
    T[1, 1] = r[1, 1]
    T[1, 2] = r[1, 2]
    T[1, 3] = translation[1]

    T[2, 0] = r[2, 0]
    T[2, 1] = r[2, 1]
    T[2, 2] = r[2, 2]
    T[2, 3] = translation[2]

    T[3, 0] = 0
    T[3, 1] = 0
    T[3, 2] = 0
    T[3, 3] = 1

    return pcd.transform(np.linalg.inv(T))

def get_transformation_data(transform_data, image_id, sensor_name):
    """Extract transformation data for a specific sensor from the JSON."""
    translation = transform_data[image_id][sensor_name]["transform"]["translation"]
    rotation = transform_data[image_id][sensor_name]["transform"]["rotation"]
    
    translation_vector = [translation["x"], translation["y"], translation["z"]]
    rotation_quaternion = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    
    return translation_vector, rotation_quaternion


with open('dataset/scene_1_caps/pose_transform_data.json', 'r') as f:
    transform_data = json.load(f)

intrinsics_left = {"fx": 912.7007, "fy": 913.1103, "cx": 653.6740, "cy": 365.5973}
intrinsics_right = {"fx": 909.7750, "fy": 909.3660, "cx": 648.5572, "cy": 386.9201}
intrinsics_top = {"fx": 914.0937, "fy": 914.0947, "cx": 649.8485, "cy": 370.4816}

image_id = 10

depth_image_right = o3d.geometry.Image(np.load(f'dataset/scene_1_caps/right_depth_img.npy'))
depth_image_left = o3d.geometry.Image(np.load(f'dataset/scene_1_caps/left_depth_img.npy'))
depth_image_top = o3d.geometry.Image(np.load(f'dataset/scene_1_caps/head_depth_img/{image_id}.npy'))

rgb_image_right = np.asarray(o3d.io.read_image("dataset/scene_1_caps/right_frame_img.png"))
rgb_image_left = np.asarray(o3d.io.read_image("dataset/scene_1_caps/left_frame_img.png"))
rgb_image_top = np.asarray(o3d.io.read_image(f"dataset/scene_1_caps/head_frame_img/{image_id}.png"))

print("get_transformation")
translation_right, rotation_right = get_transformation_data(transform_data, f"image_{image_id}", "realsense_right_color_optical_frame")
translation_left, rotation_left = get_transformation_data(transform_data, f"image_{image_id}", "realsense_left_color_optical_frame")
translation_top, rotation_top = get_transformation_data(transform_data, f"image_{image_id}", "realsense_head_color_optical_frame")

print("depth_to_pointcloud")
pcd_right = create_point_cloud_from_depth_and_rgb(depth_image_right, rgb_image_right, intrinsics_right)
pcd_left = create_point_cloud_from_depth_and_rgb(depth_image_left, rgb_image_left, intrinsics_left)
pcd_top = create_point_cloud_from_depth_and_rgb(depth_image_top, rgb_image_top, intrinsics_top)

print("apply_transformation")
pcd_right = apply_transformation(pcd_right, translation_right, rotation_right)
pcd_left = apply_transformation(pcd_left, translation_left, rotation_left)
pcd_top = apply_transformation(pcd_top, translation_top, rotation_top)

combined_pcd = pcd_right + pcd_left + pcd_top

# Estimate normals for Poisson surface reconstruction
print("estimate_normals")
#combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Perform Poisson surface reconstruction
print("create_from_point_cloud_poisson")
#mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_pcd, depth=9)

axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Combine and visualize
o3d.visualization.draw_geometries([pcd_right, pcd_left, pcd_top, axis_gizmo])


