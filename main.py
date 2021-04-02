"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by

Segmentation of boulders using 3D reconstruction point clouds [photogrammetry]

Python 3.8
Library version:

    TODO Checklist wrong projection:
         - FACT: Frames are located wrongly
         - Checked: Quaternion rotation inline with other tools
    TODO Use DATA_PATH
"""

import open3d as o3d
import numpy as np

from utilities import *

data_set = DATA_SET
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
cluster_method = "kmeans"
# cluster_method = "dbscan"

# kmeans param
k = 55
# dbscan param
epsilon = 0.3

# label generation param
min_number = 5
growth_rate = 10
shrink_rate = 5
largest_only = True
fill = True
refinement_method = "crf"
# refinement_method = "graph"
# TODO depth map not same scale thus range chosen to include all also not same size
depth_range = 1000  # allowed deviation from a point to the depth map

generate_new_filtered = False
generate_new_cluster = False
visualization = False

if visualization:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3, origin=[0, 0, 0])

# FILTER
if generate_new_filtered:
    cloud = o3d.io.read_point_cloud(f"data/{data_set}/pointclouds/point_cloud.ply")

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # # reorient for easier floor filtering
    # R = rotation_matrix(*[float(k)*np.pi/180 for k in rotation_initial])
    # cloud.rotate(R, np.array([0, 0, 0]))
    # cloud.translate(translation_initial)
    #
    # # remove floor
    # cloud.points = delete_below(points=cloud.points, threshold=0.5)
    #
    # # remove outliers
    # cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    #
    # # rotate back
    # cloud.translate([-k for k in translation_initial])
    # cloud.rotate(np.linalg.inv(R), np.array([0, 0, 0]))

    # save
    o3d.io.write_point_cloud(f"data/{data_set}/pointclouds/filtered_point_cloud.ply", cloud)

    if visualization:
        o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Filtered")
else:
    # load
    try:
        cloud = o3d.io.read_point_cloud(f"data/{data_set}/pointclouds/filtered_point_cloud.ply")
    except FileNotFoundError:
        print("[ERROR] No filtered files found")

# o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="PRESENTATION")

# CLUSTER
if generate_new_cluster:
    cloud.estimate_normals()

    reoriented_normals = reorient_normals(normals=cloud.normals, direction=normal_direction)
    cloud.normals = reoriented_normals

    # o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="PRESENTATION")

    normal_moved = move_along_normals(points=cloud.points, normals=cloud.normals, step=step)
    cloud.points = normal_moved

    # o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="PRESENTATION")

    if cluster_method == 'kmeans':
        clustered, labels = km_cluster(points=cloud.points, k=k)
    elif cluster_method == 'dbscan':
        clustered, labels = dbscan_cluster(points=cloud.points, epsilon=epsilon)
    else:
        raise KeyError("[ERROR] Wrong clustering keyword!")
    cloud.colors = clustered

    normal_moved_back = move_along_normals(points=cloud.points, normals=cloud.normals, step=-step)
    cloud.points = normal_moved_back

    # o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="PRESENTATION")

    # save
    o3d.io.write_point_cloud(f"data/{data_set}/pointclouds/{cluster_method}_clustered.ply", cloud)
    np.save(f"data/{data_set}/pointclouds/{cluster_method}_labels.npy", labels)

    if visualization:
        o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Clustered")
else:
    # load
    try:
        cloud = o3d.io.read_point_cloud(f"data/{data_set}/pointclouds/{cluster_method}_clustered.ply")
        # cloud = o3d.io.read_point_cloud(f"data/pointclouds/filtered_reconstruction_{exp_number}.ply")
        labels = np.load(f"data/{data_set}/pointclouds/{cluster_method}_labels.npy")
    except FileNotFoundError:
        print("[ERROR] No clustered files found")

# PROJECTION
trans_mat = np.eye(4)
for image, position, depth_map, name in load_images():
    # generate distance map
    target_point = o3d.geometry.PointCloud()
    target_point.points = o3d.utility.Vector3dVector(position[np.newaxis, -3:])
    distance_map = cloud.compute_point_cloud_distance(target=target_point)
    distance_map = np.asarray(distance_map)

    # build transformation matrix
    # R = target_point.get_rotation_matrix_from_quaternion([1, 0, 0, 0])
    R = target_point.get_rotation_matrix_from_quaternion([position[0], position[1], position[2], position[3]])
    # R = np.linalg.inv(R)
    trans_mat[:3, :3] = R
    trans_mat[:3, 3] = position[-3:]
    print(name)
    print(position)
    print(trans_mat)

    # if visualization:
    #     name_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #         size=2, origin=position[-3:])
    #     name_frame.rotate(R, position[-3:])
    #     o3d.visualization.draw_geometries([cloud, name_frame, origin_frame], width=3000, height=1800,
    #                                       window_name=f"Frame {name}")

    # project, generate a label and save it as a set of masks
    projection = reproject(points=cloud.points, color=cloud.colors, label=labels,
                           transformation_mat=trans_mat, depth_map=depth_map, depth_range=depth_range,
                           distance_map=distance_map, save_img=True, name=name)
    generate_label(projection=projection, original=image, growth_rate=growth_rate, shrink_rate=shrink_rate,
                   min_number=min_number, name=name, refinement_method=refinement_method, fill=fill,
                   largest=largest_only)
    # save_label(label_name=name, label=label)

# VISUALIZATION
if visualization:
    o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Origin Frame")
