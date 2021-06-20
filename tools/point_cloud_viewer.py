"""
Created by Jan Schiffeler on 13.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import open3d as o3d
from settings import *

# cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply")
cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.ply")

o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Old")
