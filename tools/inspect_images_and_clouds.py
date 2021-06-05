"""
Created by Jan Schiffeler on 12.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
from utilities import read_depth_map
from settings import DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2

# label = cv2.imread("/Users/jan/Programming/PycharmProjects/master/3d_segmentation/debug_images/visual_projection_39.png")
# img = cv2.imread(f"{DATA_PATH}/images/39.png")
# both = cv2.addWeighted(img, 0.4, label, 0.8, 0.0)


cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.ply")
o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Inspector")
# name = "02_01_00600"
# normal_map = read_depth_map(DATA_PATH + "/normals/" + name + '.png.geometric.bin')
# np.save("debug_images/normal.npy", normal_map)
# plt.imshow(normal_map)
# plt.show()

print("Done")
