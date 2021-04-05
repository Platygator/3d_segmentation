"""
Created by Jan Schiffeler on 05.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
from camera_parameters import load_camera_param

# Chose camera parameters
# camera = "real"
camera = "simulation"

CAM_MAT, DIST_MAT, HEIGHT, WIDTH = load_camera_param(camera=camera)


# Define path to data and to specific data set
DATA_PATH = "data"
# DATA_SET = "reconstruction_1"
# DATA_SET = "reconstruction_2"
DATA_SET = "simulation_1"


# rouge filter
nb_neighbors = 20
std_ratio = 1.0

# reorientation for floor filtering
rotation_initial = [0, -5, 0]  # [5, 142, 0]
translation_initial = [0.0, 0.0, 4.5]

# cluster pre processing param
normal_direction = np.array([-1, 0, 2])
step = 0.5

# chose cluster method
cluster_method = "kmeans"
# cluster_method = "dbscan"

# kmeans param
k = 23  # 55
# dbscan param
epsilon = 0.3

# label generation param
min_number = 5
growth_rate = 10
shrink_rate = 5
largest_only = True
fill = True
# refinement_method = "crf"
t = 10
refinement_method = "graph"
graph_mask_thresh = 200
iter_count = 5
# TODO depth map not same scale thus range chosen to include all
depth_range = 1000  # allowed deviation from a point to the depth map

generate_new_filtered = False
generate_new_cluster = False
visualization = False
