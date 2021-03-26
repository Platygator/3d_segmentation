"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:


"""

import open3d as o3d
import numpy as np

from utilities import *

nb_neighbors = 20
std_ratio = 1.0
rotation_initial = [5, 142, 0]
translation_initial = [0.0, 0.0, 4.5]

normal_direction = np.array([-1, 0, 2])
step = 0.5
k = 53

growth_rate = 5
min_number = 50

generate_new_filtered = False
visualization = False

# FILTER
if generate_new_filtered:
    cloud = o3d.io.read_point_cloud("data/point_clouds/reconstruction_1.ply")

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # reorient for easier floor filtering
    R = rotation_matrix(*[float(k)*np.pi/180 for k in rotation_initial])
    cloud.rotate(R, np.array([0, 0, 0]))
    cloud.translate(translation_initial)

    # remove floor
    cloud.points = delete_below(points=cloud.points, threshold=0.5)

    # remove outliers
    cloud = remove_statistical_outliers(cloud=cloud,  nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # rotate back
    cloud.translate([-k for k in translation_initial])
    cloud.rotate(np.linalg.inv(R), np.array([0, 0, 0]))

    # save
    o3d.io.write_point_cloud("data/point_clouds/filtered_reconstruction_1.ply", cloud)
else:
    # load
    cloud = o3d.io.read_point_cloud("data/point_clouds/filtered_reconstruction_1.ply")
    
# CLUSTER
cloud.estimate_normals()

reoriented_normals = reorient_normals(normals=cloud.normals, direction=normal_direction)
cloud.normals = reoriented_normals

normal_moved = move_along_normals(points=cloud.points, normals=cloud.normals, step=step)
cloud.points = normal_moved

clustered, labels = km_cluster(points=cloud.points, k=k)
cloud.colors = clustered

normal_moved_back = move_along_normals(points=cloud.points, normals=cloud.normals, step=-step)
cloud.points = normal_moved_back

# PROJECTION
for image, position, depth_map, name in load_images():
    projection = reproject(points=cloud.points, color=cloud.colors, label=labels,
                           transformation_mat=np.eye(4), depth_map=depth_map)
    label = generate_label(projection=projection, original=image, growth_rate=growth_rate,
                           min_number=min_number, name=name)
    # save_label(label_name=name, label=label)

# VISUALIZATION
if visualization:
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800)