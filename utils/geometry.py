import numpy as np
import cv2
import open3d as o3d
import copy


def depth_2_pc_distortion(fx, fy, cx, cy, d, depth_array):
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    h, w = depth_array.shape
    
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([i, j], axis=-1).reshape(-1, 2).astype(np.float32)

    undistorted_pixels = cv2.undistortPoints(pixels, k, d, None, k).reshape(h, w, 2)
    
    x_normalized = (undistorted_pixels[..., 0] - k[0, 2]) / k[0, 0]
    y_normalized = (undistorted_pixels[..., 1] - k[1, 2]) / k[1, 1]
    
    z = depth_array / 1000.0
    x = x_normalized * z
    y = y_normalized * z
    
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
            if z > 1.0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
    return np.array([np.array(x_points), np.array(y_points), np.array(z_points)])


def fit_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return a, b, c, d, inliers


def find_rotation(a, b, c):
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    z_axis = np.array([0, 0, 1]) # Target normal (Z-axis)

    # Calculate rotation axis (cross product) and angle (dot product)
    rotation_axis = np.cross(plane_normal, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis) 

    angle = np.arccos(np.dot(plane_normal, z_axis))

    # Construct the rotation matrix using Rodrigues' rotation formula
    return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)


def place_models(plane_centroids, a, b, c):
    mesh = o3d.io.read_triangle_mesh("data/Cup_Made_By_Tyro_Smith.ply")
    mesh.compute_vertex_normals()

    transformed_meshes = []
    spheres = []
    for kp in plane_centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(kp[2]) 
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        spheres.append(sphere)

        r = find_rotation(-a, -b, -c)
        transformed_mesh = copy.deepcopy(mesh)
        transformed_mesh.scale(1/20, center=(0, 0, 0))
        transformed_mesh.rotate(np.linalg.inv(r))
        transformed_mesh.translate(kp[2] + np.linalg.inv(r).dot(np.array([0, 0, 2.83157/20])), relative=False)
        transformed_meshes.append(transformed_mesh)
    return transformed_meshes