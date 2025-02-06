import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
import cv2


def threshold_blobs(outlier_cloud, a, b, c, d):
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=500, print_progress=True))

    if labels.shape[0] == 0:
        return np.array([]), np.array([])

    max_label = labels.max()

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Black noise points (label = -1)
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    blob_heights = []

    for i in range(max_label + 1):
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

    for i, height in blob_heights:
        print(f"Blob {i} height from table plane: {height*100:.4f} centimeters")

    return labels, blob_heights


def bgr_to_lab(bgr):
    bgr = np.array(bgr, dtype=np.uint8).reshape(1, 1, 3)  
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0  
    lab = rgb2lab(rgb)[0, 0] 
    return lab


def pixel_coordinates(outlier_cloud, labels, blob_heights, caps_image, caps_color, dist_threshold, camera_matrix, known_heights):
    #r_inv = np.linalg.inv(r)
    coordinates = []

    for i, height in blob_heights:
        height = height * 100
        best, index = None,-1
        for j, known_height in enumerate(known_heights):
            current = abs(known_height - height)
            if current < 3 and (best == None or current < best): 
                best, index = current, j

        if index == -1:
            continue

        cluster_points = np.asarray(outlier_cloud.select_by_index(np.where(labels == i)[0]).points)
        if len(cluster_points) == 0:
            continue
        
        centroid = np.mean(cluster_points, axis=0)
        #original_centroid = np.dot(r_inv, centroid)
        
        x, y, z = centroid
        u = (camera_matrix[0, 0] * x / z) + camera_matrix[0, 2]
        v = (camera_matrix[1, 1] * y / z) + camera_matrix[1, 2]

        pixel_color = caps_image[int(v), int(u)]

        if dist_threshold > 100:
            color_distance = np.linalg.norm(caps_color - pixel_color)
            print(f"color distance: {color_distance}")
            if color_distance > dist_threshold:
                continue
            if pixel_color[0] > pixel_color[1] or pixel_color[1] > pixel_color[2]:
                continue
        else:
            lab_pixel = bgr_to_lab(pixel_color)
            lab_target = bgr_to_lab(caps_color)
        
            # Only compare A and B channels (ignoring lightness L)
            color_distance = np.linalg.norm(lab_pixel[1:] - lab_target[1:]) 
            print(f"color distance: {color_distance}")
            if color_distance > dist_threshold:
                continue

        cv2.circle(caps_image, (int(u), int(v)), radius=3, color=(0, 255, 0), thickness=2) 
        cv2.imwrite('work_dirs/debug/debug_circles.png', caps_image)
        coordinates.append((int(u), int(v), index, *centroid))

    return np.array(coordinates)


def project_points_to_plane(plane, centroids, rgb_image, camera_matrix):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm

    projected_points = []
    for point in centroids:
        x, y, z = point
        distance = (a * x + b * y + c * z + d) / normal_norm
        projection = point - distance * normal
        x, y, z = projection

        u = (camera_matrix[0, 0] * x / z) + camera_matrix[0, 2]
        v = (camera_matrix[1, 1] * y / z) + camera_matrix[1, 2]
        cv2.circle(rgb_image, (int(u), int(v)), radius=3, color=(0, 255, 255), thickness=2) 
        cv2.imwrite('work_dirs/debug/debug_keypoints.png', rgb_image)

        projected_points.append((int(u), int(v), projection))

    return projected_points
