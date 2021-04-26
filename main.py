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
from settings import *

data_set = DATA_PATH + "/" + DATA_SET

# load
try:
    cloud = o3d.io.read_point_cloud(f"{data_set}/pointclouds/clustered.ply")
    labels = np.load(f"{data_set}/pointclouds/labels.npy")
except FileNotFoundError:
    print("[ERROR] No clustered files found")
    quit()

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
                           distance_map=distance_map, save_img=False, name=name)
    generate_masks(projection=projection, original=image, growth_rate=growth_rate, shrink_rate=shrink_rate,
                   min_number=min_number, name=name, refinement_method=refinement_method, fill=fill,
                   largest=largest_only, graph_thresh=graph_mask_thresh, t=t, iter_count=iter_count)

# VISUALIZATION
if visualization:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Origin Frame")
