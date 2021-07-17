"""
Created by Jan Schiffeler on 13.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import open3d as o3d
from settings import *

cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/fused.ply")

o3d.visualization.draw_geometries([cloud], width=3000, height=1800, mesh_show_wireframe=True,
                                  lookat=np.array([[0, 0, 0]], dtype='float64').T,
                                  up=np.array([[0.8, -0.2, -0.8]], dtype='float64').T,
                                  front=np.array([[-0.7, -0.25, -0.2]], dtype='float64').T,
                                  zoom=0.4
                                   )
