"""
Created by Jan Schiffeler on 10.06.12021
jan.schiffeler[at]gmail.com

Changed by

Transform point cloud from ply to pcd to be used in https://app.supervise.ly
Python 3.

"""

import open3d as o3d
from settings import DATA_PATH

cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.ply")
o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.pcd", cloud)
cloud_re = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.pcd")
o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Inspector")
