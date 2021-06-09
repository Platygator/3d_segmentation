"""
Created by Jan Schiffeler on 09.06.12021
jan.schiffeler[at]gmail.com

Changed by


Simplified version of the open3d example code found here:
http://www.open3d.org/docs/latest/tutorial/Advanced/interactive_visualization.html


Can be used to manually crop a point cloud

Python 3.8

"""

import open3d as o3d
from settings import DATA_PATH


print("Demo for manual geometry cropping")
print(
    "1) Press 'Y' twice to align geometry with negative direction of y-axis"
)
print("2) Press 'K' to lock screen and to switch to selection mode")
print("3) Drag for rectangle selection,")
print("   or use ctrl + left click for polygon selection")
print("4) Press 'C' to get a selected geometry and to save it")
print("5) Press 'F' to switch to freeview mode")
cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/fused.ply")
o3d.visualization.draw_geometries_with_editing([cloud])
