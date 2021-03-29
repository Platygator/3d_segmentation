"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by

Segmentation of boulders using 3D reconstruction point clouds [photogrammetry]

Python 3.8
Library version:


"""

import open3d as o3d
import numpy as np

from utilities import *

# rouge filter
nb_neighbors = 20
std_ratio = 1.0

# reorientation for floor filtering
rotation_initial = [5, 142, 0]
translation_initial = [0.0, 0.0, 4.5]

# cluster pre processing param
normal_direction = np.array([-1, 0, 2])
step = 0.5

# chose cluster method
# cluster_method = "kmeans"
cluster_method = "dbscan"

# kmeans param
k = 53
# dbscan param
epsilon = 0.3

# label generation param
growth_rate = 5
min_number = 50
depth_range = 0.1  # allowed deviation from a point to the depth map

generate_new_filtered = False
generate_new_cluster = True
visualization = True

# FILTER
if generate_new_filtered:
    cloud = o3d.io.read_point_cloud("data/pointclouds/reconstruction_1.ply")

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # reorient for easier floor filtering
    R = rotation_matrix(*[float(k)*np.pi/180 for k in rotation_initial])
    cloud.rotate(R, np.array([0, 0, 0]))
    # cloud.translate(translation_initial)

    # remove floor
    cloud.points = delete_below(points=cloud.points, threshold=5.0)

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # rotate back
    cloud.translate([-k for k in translation_initial])
    cloud.rotate(np.linalg.inv(R), np.array([0, 0, 0]))

    # save
    o3d.io.write_point_cloud("data/pointclouds/filtered_reconstruction_1.ply", cloud)
else:
    # load
    cloud = o3d.io.read_point_cloud("data/pointclouds/filtered_reconstruction_1.ply")
    
# CLUSTER
if generate_new_cluster:
    cloud.estimate_normals()

    reoriented_normals = reorient_normals(normals=cloud.normals, direction=normal_direction)
    cloud.normals = reoriented_normals

    normal_moved = move_along_normals(points=cloud.points, normals=cloud.normals, step=step)
    cloud.points = normal_moved

    if cluster_method == 'kmeans':
        clustered, labels = km_cluster(points=cloud.points, k=k)
    elif cluster_method == 'dbscan':
        clustered, labels = dbscan_cluster(points=cloud.points, epsilon=0.3)
    else:
        raise KeyError("[ERROR] Wrong clustering keyword!")
    cloud.colors = clustered

    normal_moved_back = move_along_normals(points=cloud.points, normals=cloud.normals, step=-step)
    cloud.points = normal_moved_back

    # save
    o3d.io.write_point_cloud(f"data/pointclouds/{cluster_method}_clustered_reconstruction_1.ply", cloud)
    np.save(f"data/pointclouds/{cluster_method}_labels.npy", labels)
else:
    # load
    cloud = o3d.io.read_point_cloud("data/pointclouds/clustered_{cluster_method}_reconstruction_1.ply")
    labels = np.load(f"data/pointclouds/{cluster_method}_labels.npy")

# PROJECTION
trans_mat = np.eye(4)
for image, position, depth_map, name in load_images():
    # generate distance map
    target_point = o3d.geometry.PointCloud()
    target_point.points = o3d.utility.Vector3dVector(position[np.newaxis, :3])
    distance_map = cloud.compute_point_cloud_distance(target=target_point)

    # build transformation matrix
    R = target_point.get_rotation_matrix_from_quaternion(position[3:])
    trans_mat[:3, :3] = R
    trans_mat[:3, 3] = position[:3]

    # project, generate a label and save it as a set of masks
    projection = reproject(points=cloud.points, color=cloud.colors, label=labels,
                           transformation_mat=trans_mat, depth_map=depth_map, depth_range=depth_range,
                           distance_map=distance_map)
    generate_label(projection=projection, original=image, growth_rate=growth_rate,
                   min_number=min_number, name=name)
    # save_label(label_name=name, label=label)

# VISUALIZATION
if visualization:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800)