"""
Created by Jan Schiffeler on 15.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import open3d as o3d
from utilities import *
from settings import DATA_PATH


# cluster pre processing param
normal_direction = np.array([0, 0, 1])
step = 0.2

# chose cluster method
cluster_method = "kmeans"
# cluster_method = "dbscan"

# kmeans param
k = 15  # 55
# dbscan param
epsilon = 0.3
new_normals = False

# FILTER
# load
try:
    cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply")
except FileNotFoundError:
    print("[ERROR] No filtered file found")
    quit()

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=3, origin=[0, 0, 0])

if new_normals:
    cloud.estimate_normals()

    reoriented_normals = reorient_normals(normals=cloud.normals, direction=normal_direction)
    cloud.normals = reoriented_normals


# normal_dir = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
# normal_dir.paint_uniform_color([0.9, 0.1, 0.1])
# normal_dir.translate(normal_direction)
# o3d.visualization.draw_geometries([cloud, normal_dir, origin_frame], width=3000, height=1800, window_name="Normals")

normal_moved = move_along_normals(points=cloud.points, normals=cloud.normals, step=step)
cloud.points = normal_moved
o3d.visualization.draw_geometries([cloud, origin_frame], width=3000, height=1800, window_name="Normals")

if cluster_method == 'kmeans':
    clustered, labels = km_cluster(points=cloud.points, k=k)
elif cluster_method == 'dbscan':
    clustered, labels = dbscan_cluster(points=cloud.points, epsilon=epsilon)
else:
    raise KeyError("[ERROR] Wrong clustering keyword!")
cloud.colors = clustered

normal_moved_back = move_along_normals(points=cloud.points, normals=cloud.normals, step=-step)
cloud.points = normal_moved_back

# save
o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/clustered.ply", cloud)
np.save(f"{DATA_PATH}/pointclouds/labels.npy", labels)

o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Clustered")
