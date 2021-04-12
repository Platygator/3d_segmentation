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
from settings import *

data_set = DATA_PATH + "/" + DATA_SET

if visualization:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3, origin=[0, 0, 0])

# FILTER
if generate_new_filtered:
    cloud = o3d.io.read_point_cloud(f"{data_set}/pointclouds/point_cloud.ply")

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # reorient for easier floor filtering
    R = rotation_matrix(*[float(k)*np.pi/180 for k in rotation_initial])
    cloud.rotate(R, np.array([0, 0, 0]))
    # cloud.translate(translation_initial)
    #

    t = 10.6
    # remove floor
    cloud.points = delete_above(points=cloud.points, threshold=t)

    w = 25
    h = 30
    mesh_plane = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=0.001)
    mesh_plane.compute_vertex_normals()
    mesh_plane.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_plane.translate([-w + 3.0, -h/2, t])

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # rotate back
    # cloud.translate([-k for k in translation_initial])
    cloud.rotate(np.linalg.inv(R), np.array([0, 0, 0]))

    # save
    o3d.io.write_point_cloud(f"{data_set}/pointclouds/filtered_point_cloud.ply", cloud)

    if visualization:
        o3d.visualization.draw_geometries([cloud, origin_frame, mesh_plane],
                                          width=3000, height=1800, window_name="Filtered")
else:
    # load
    try:
        cloud = o3d.io.read_point_cloud(f"{data_set}/pointclouds/filtered_point_cloud.ply")
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
    o3d.io.write_point_cloud(f"{data_set}/pointclouds/{cluster_method}_clustered.ply", cloud)
    np.save(f"{data_set}/pointclouds/{cluster_method}_labels.npy", labels)

    if visualization:
        o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Clustered")
        quit()
else:
    # load
    try:
        cloud = o3d.io.read_point_cloud(f"{data_set}/pointclouds/{cluster_method}_clustered.ply")
        # cloud = o3d.io.read_point_cloud(f"data/pointclouds/filtered_reconstruction_{exp_number}.ply")
        labels = np.load(f"{data_set}/pointclouds/{cluster_method}_labels.npy")
    except FileNotFoundError:
        print("[ERROR] No clustered files found")

# PROJECTION
trans_mat = np.eye(4)
for image, position, depth_map, name in load_images():
    # generate distance map
    # target_point = o3d.geometry.PointCloud()
    # target_point.points = o3d.utility.Vector3dVector(position[np.newaxis, -3:])
    # distance_map = cloud.compute_point_cloud_distance(target=target_point)
    # distance_map = np.asarray(distance_map)
    distance_map = np.asarray(cloud.points)

    # build transformation matrix
    R = cloud.get_rotation_matrix_from_quaternion([position[0], position[1], position[2], position[3]])
    trans_mat[:3, :3] = R
    trans_mat[:3, 3] = position[-3:]
    # print("[INFO] Positional information: ")
    # print(name)
    # print(position)
    # print(trans_mat)

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
    generate_masks(projection=projection, original=image, growth_rate=growth_rate, shrink_rate=shrink_rate,
                   min_number=min_number, name=name, refinement_method=refinement_method, fill=fill,
                   largest=largest_only, graph_thresh=graph_mask_thresh, t=t, iter_count=iter_count)

# VISUALIZATION
if visualization:
    o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Origin Frame")
