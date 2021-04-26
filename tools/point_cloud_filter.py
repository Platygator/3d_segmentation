"""
Created by Jan Schiffeler on 15.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import open3d as o3d
import numpy as np

from utilities import *
from settings import DATA_PATH

# rouge filter
nb_neighbors = 20
std_ratio = 1.0

# reorientation for floor filtering
rotation_initial = [2.5, -5.5, 0]  # [-33, 3, 0]  # [5, 142, 0]
# translation_initial = [0.0, 0.0, 4.5]

cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.ply")

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=3, origin=[0, 0, 0])

# remove outliers
cloud = remove_statistical_outliers(cloud=cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

# reorient for easier floor filtering
R = rotation_matrix(*[float(k) * np.pi / 180 for k in rotation_initial])
cloud.rotate(R, np.array([0, 0, 0]))
# cloud.translate(translation_initial)
#

t = 2.2

w = 25
h = 30
mesh_plane = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=0.001)
mesh_plane.compute_vertex_normals()
mesh_plane.paint_uniform_color([0.9, 0.1, 0.1])
mesh_plane.translate([-w + 3.0, -h / 2, t])


# remove floor
cloud.points = delete_above(points=cloud.points, threshold=t)

# remove radius
cloud.points = delete_radius(points=cloud.points, radius=3)

# remove outliers
cloud = remove_statistical_outliers(cloud=cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

# o3d.visualization.draw_geometries([cloud, mesh_plane, origin_frame],
#                                   width=3000, height=1800, window_name="Filtered")
# quit()

# rotate back
# cloud.translate([-k for k in translation_initial])
cloud.rotate(np.linalg.inv(R), np.array([0, 0, 0]))

# save
o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply", cloud)

o3d.visualization.draw_geometries([cloud],
                                  width=3000, height=1800, window_name="Filtered")
