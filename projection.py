import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json
import cv2
import os
import open3d as o3d

ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")


def depth_to_3d(u, v, z, K):
    """Back-project depth image points to 3D coordinates in the depth camera's coordinate system."""
    z = z * 0.001
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z, 1])


def project_to_image_plane(X_world, T_world_to_rgb, K_rgb):
    """Project a 3D point in world coordinates onto a 2D RGB image."""
    X_rgb = T_world_to_rgb @ X_world  # Transform to RGB camera coordinates
    x_rgb, y_rgb, z_rgb = X_rgb[:3] / X_rgb[2]  # Normalize by depth
    pixel_coords = K_rgb @ np.array([x_rgb, y_rgb, 1])  # Project onto 2D image plane
    u_rgb = int(pixel_coords[0])
    v_rgb = int(pixel_coords[1])
    return u_rgb, v_rgb


def get_transformation(transform_data, image_id, sensor_name):
    """Extract transformation data for a specific sensor from the JSON."""
    translation = transform_data[image_id][sensor_name]["transform"]["translation"]
    rotation = transform_data[image_id][sensor_name]["transform"]["rotation"]
    
    translation_vector = [translation["x"], translation["y"], translation["z"]]
    rotation_quaternion = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    
    rotation = R.from_quat(rotation_quaternion).as_matrix()

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation_vector
    matrix[3, 3] = 1

    return matrix

K_depth = np.array([[914.0937, 0, 649.8485], [0, 914.0947, 370.4816], [0, 0, 1]])
K_rgb_top = K_depth
K_rgb_left = np.array([[912.7007, 0, 653.6740], [0, 913.1103, 365.5973], [0, 0, 1]])
K_rgb_right = np.array([[909.7750, 0, 648.5572], [0, 909.3660, 386.9201], [0, 0, 1]])

K_left_eye = np.array([[911.3754, 0, 933.9101], [0, 909.0219, 758.0138], [0, 0, 1]])
K_right_eye = np.array([[909.4356, 0, 965.7616], [0, 909.0565, 711.3480], [0, 0, 1]])


D_depth = [0.074713, -0.133798, 0.005981, -0.008687, 0.000000]
D_rgb_top = D_depth

D_rgb_left = [0.106236, -0.152444, -0.012214, -0.010710, 0.000000]
D_rgb_right = [0.120152, -0.160957, -0.013440, -0.009624, 0.000000]

D_left_eye = [0.0, 0.0, 0.0, 0.0, 0.0]
D_right_eye = [0.0, 0.0, 0.0, 0.0, 0.0]



image_id = 7
with open('dataset/scene_1_caps/pose_transform_data.json', 'r') as f:
    transform_data = json.load(f)

T_world_to_rgb_top = get_transformation(transform_data, f"image_{image_id}", "realsense_head_color_optical_frame")
T_world_to_rgb_left = get_transformation(transform_data, f"image_{image_id}", "realsense_left_color_optical_frame")
T_world_to_eye_left = get_transformation(transform_data, f"image_{image_id}", "left_eye_cam")
T_world_to_eye_right = get_transformation(transform_data, f"image_{image_id}", "right_eye_cam")

# Inverse transformations from world to RGB cameras
T_depth_to_world = np.linalg.inv(T_world_to_rgb_top)
T_left_to_world = np.linalg.inv(T_world_to_rgb_left)
T_left_eye_to_world = np.linalg.inv(T_world_to_eye_left)
T_right_eye_to_world = np.linalg.inv(T_world_to_eye_right)

depth_image_top = np.load(f'dataset/scene_1_caps/head_depth_img/{image_id}.npy')
rgb_image_top = np.asarray(o3d.io.read_image(f"dataset/scene_1_caps/head_frame_img/{image_id}.png"))
rgb_image_left = np.asarray(o3d.io.read_image("dataset/scene_1_caps/left_frame_img.png"))
rgb_eye_left = np.asarray(o3d.io.read_image(f"dataset/scene_1_caps/left_eye/{image_id}.png"))
rgb_eye_left = cv2.resize(rgb_eye_left, (1920, 1440)) 
rgb_eye_left = cv2.cvtColor(rgb_eye_left, cv2.COLOR_BGR2RGB)
rgb_eye_right = np.asarray(o3d.io.read_image(f"dataset/scene_1_caps/right_eye/{image_id}.png"))
rgb_eye_right = cv2.resize(rgb_eye_right, (1920, 1440)) 
rgb_eye_right = cv2.cvtColor(rgb_eye_right, cv2.COLOR_BGR2RGB)

depth_points = [(620, 420, depth_image_top[420, 620])]

# Loop through the depth points
for (u_d, v_d, z_d) in depth_points:
    # Step 1: Back-project to 3D using the depth camera intrinsics
    X_d = depth_to_3d(u_d, v_d, z_d, K_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720, fx=K_depth[0, 0], fy=K_depth[1, 1], cx=K_depth[0, 2], cy=K_depth[1, 2]
    )
    depth_image = o3d.geometry.Image(depth_image_top)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
         depth_image,
         intrinsics,
         depth_scale=1000.0, 
         depth_trunc=3.0,
         stride=1
     )
    axis_gizmo = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.translate(X_d[:3])
    o3d.visualization.draw_geometries([pcd, axis_gizmo, sphere])

    # Step 2: Transform the 3D point from depth camera to world coordinates
    X_world = T_depth_to_world @ X_d 

    print(X_world)

    # Step 3: Project the point to each of the RGB cameras
    u_rgb1, v_rgb1 = project_to_image_plane(X_world, T_world_to_rgb_top, K_rgb_top)

    print(u_rgb1)
    print(v_rgb1)

    u_rgb2, v_rgb2 = project_to_image_plane(X_world, T_world_to_rgb_left, K_rgb_left)

    print(u_rgb2)
    print(v_rgb2)

    u_rgb3, v_rgb3 = project_to_image_plane(X_world, T_world_to_eye_left, K_left_eye)

    print(u_rgb3)
    print(v_rgb3)

    u_rgb4, v_rgb4 = project_to_image_plane(X_world, T_world_to_eye_right, K_right_eye)

    print(u_rgb4)
    print(v_rgb4)

    # Step 4: Visualize the point in each of the RGB images
    cv2.circle(rgb_image_top, (u_rgb1, v_rgb1), 5, (0, 255, 0), -1)
    cv2.circle(rgb_image_left, (u_rgb2, v_rgb2), 5, (0, 255, 0), -1)
    cv2.circle(rgb_eye_left, (u_rgb3, v_rgb3), 5, (0, 255, 0), -1)
    cv2.circle(rgb_eye_right, (u_rgb4, v_rgb4), 5, (0, 255, 0), -1)


# Show the results
cv2.imshow("Img1", rgb_image_top)
cv2.imshow("Img2", rgb_image_left)
cv2.imshow("Img3", rgb_eye_left)
cv2.imshow("Img4", rgb_eye_right)
cv2.imshow("Img5", depth_image_top * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
