"""
Created by Jan Schiffeler on 05.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
from camera_parameters import load_camera_param
import os

# Chose camera parameters
# camera = "real"
camera = "simulation"

CAM_MAT, DIST_MAT, HEIGHT, WIDTH = load_camera_param(camera=camera)


# Define path to data and to specific data set
# DATA_PATH = "/Users/jan/Programming/PycharmProjects/master/3d_segmentation/data"
DATA_PATH = "/Volumes/Transfer/master/3d_sets"
# DATA_SET = "reconstruction_1"
# DATA_SET = "reconstruction_2"
DATA_SET = "simulation_3"

DATA_PATH = os.path.join(DATA_PATH, DATA_SET)

# mask generation param
# depth_range = 0.3  # allowed deviation from a point to the depth map
min_number = 5
growth_rate = 10
shrink_rate = 10
blur = 17  # relative to mask size
largest_only = False
fill = False


refinement_method = "crf"
# refinement_method = "graph"

# crf param
times = 7
gsxy = 2
gcompat = 3
bsxy = 10
brgb = 3
bcompat = 15

# graph cut param
graph_mask_thresh = 125
iter_count = 5

# label generation
border_thickness = 3

# unknown parameters
un_max_refinement_loss = 0.5
un_small_tresh = 500

generate_new_cluster = False
visualization = False
