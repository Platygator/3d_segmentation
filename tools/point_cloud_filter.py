"""
Created by Jan Schiffeler on 15.04.12021
jan.schiffeler[at]gmail.com

Changed by


Python 3.

"""

import open3d as o3d
import numpy as np
import cv2

from utilities import *
from settings import DATA_PATH, CAM_MAT, DIST_MAT

# rouge filter
nb_neighbors = 5
std_ratio = 0.2


cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.ply")

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=3, origin=[0, 0, 0])

# remove outliers
cloud = remove_statistical_outliers(cloud=cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
# o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply", cloud)
# quit()
# o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Plane filtered",
#                                   lookat=np.array([[0, 0, 2.0]], dtype='float64').T,
#                                   up=np.array([[0, -1.0, 0]], dtype='float64').T,
#                                   front=np.array([[0.2, 0.7, -1.0]], dtype='float64').T,
#                                   zoom=0.6)
# quit()
# plane_model, inliers = cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
#
# inlier_cloud = cloud.select_by_index(inliers)
# outlier_cloud = cloud.select_by_index(inliers, invert=True)
#
# inlier_cloud.paint_uniform_color([1, 0, 0])
# # outlier_cloud.paint_uniform_color([1, 0, 0])
#
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], width=3000, height=1800, window_name="Plane filtered",
#                                   lookat=np.array([[0, 0, 0]], dtype='float64').T,
#                                   up=np.array([[0, -1.0, 0]], dtype='float64').T,
#                                   front=np.array([[0.2, 0.7, -1.0]], dtype='float64').T,
#                                   zoom=0.6)
# quit()
#
# cloud.points = outlier_cloud.points
# cloud.colors = outlier_cloud.colors
#
# cloud = remove_statistical_outliers(cloud=cloud, nb_neighbors=nb_neighbors, std_ratio=nb_neighbors)
# inliers = delete_radius(points=cloud.points, radius=6.5)
# # cloud.paint_uniform_color([1, 0, 0])
#
# inlier_cloud = cloud.select_by_index(inliers)
# outlier_cloud = cloud.select_by_index(inliers, invert=True)
#
# # inlier_cloud.paint_uniform_color([1, 0, 0])
# outlier_cloud.paint_uniform_color([1, 0, 0])
#
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], width=3000, height=1800, window_name="Plane filtered",
#                                   lookat=np.array([[0, 0, 0]], dtype='float64').T,
#                                   up=np.array([[0, -1.0, 0]], dtype='float64').T,
#                                   front=np.array([[0.2, 0.7, -1.0]], dtype='float64').T,
#                                   zoom=0.6)
# quit()

o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Clustered")

cloud.estimate_normals()
cloud_points = np.asarray(cloud.points)
normals = np.asarray(cloud.normals)

normal_reorientation = np.zeros([normals.shape[0], 1])
for image, position, normal_map, name in load_images(depth_path="/normals/"):
    if not normal_map.any():
        continue
    camera_center = np.array([[position[4], position[5], position[6]]])
    trans_mat = np.eye(4)
    R = cloud.get_rotation_matrix_from_quaternion([position[0], position[1], position[2], position[3]])
    trans_mat[:3, :3] = R
    trans_mat[:3, 3] = position[-3:]

    rvec = cv2.Rodrigues(trans_mat[:3, :3])
    cv_projection = cv2.projectPoints(objectPoints=cloud_points, rvec=rvec[0], tvec=trans_mat[:3, 3],
                                      cameraMatrix=CAM_MAT, distCoeffs=DIST_MAT)
    pixels = np.rint(cv_projection[0]).astype('int')

    height, width, _ = normal_map.shape

    pixels = np.roll(pixels, 1, axis=2)[:, 0, :]
    mask_u_0 = pixels[:, 0] < 0
    mask_u_w = pixels[:, 0] >= height
    mask_v_0 = pixels[:, 1] < 0
    mask_v_h = pixels[:, 1] >= width

    mask = mask_u_0 + mask_u_w + mask_v_0 + mask_v_h
    pixels[mask, :] = [0, 0]

    # projected_normal_map = np.zeros_like(normals)
    projected_normal_map = normal_map[pixels[:, 0], pixels[:, 1], :]
    # orient = normals.dot(projected_normal_map.T)
    orient = np.sum(normals * projected_normal_map, axis=1)
    orient[mask] = 0
    orient = orient[:, np.newaxis]
    abs_orient = abs(orient)
    abs_orient[abs_orient == 0] = 1
    normal_reorientation += orient / abs_orient

mask_unsure = normal_reorientation <= 20
abs_norm = np.abs(normal_reorientation)
abs_norm[abs_norm == 0] = -1
normal_reorientation /= abs_norm
normal_reorientation[normal_reorientation == 0] = 1
normals *= (normal_reorientation * -1)
# normals[mask_unsure[:, 0], :] = [0, 0, 1]
cloud.normals = o3d.utility.Vector3dVector(normals)

# save
o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply", cloud)

o3d.visualization.draw_geometries([cloud],
                                  width=3000, height=1800, window_name="Filtered")
