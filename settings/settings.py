"""
Created by Jan Schiffeler on 05.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import os
import numpy as np


def load_camera_param(cam):
    if cam == "real":
        CAM_MAT = np.array([[1577.1159987660135, 0, 676.7292997380368],
                            [0, 1575.223362703865,  512.8101184300463],
                            [0, 0, 1]])

        DIST_MAT = np.array([-0.46465317710098897, 0.2987490394355827, 0.004075959465516531, 0.005311175696501367])
        # k1: -0.46465317710098897, k2: 0.2987490394355827, p1: 0.004075959465516531, p2: 0.005311175696501367
        HEIGHT = 1080
        WIDTH = 1440
    elif cam == "simulation":
        CAM_MAT = np.array([[455, 0, 376],
                            [0, 455, 240],
                            [0.0, 0, 1]])

        DIST_MAT = np.array([0.0, 0.0, 0.0, 0.0])
        HEIGHT = 480
        WIDTH = 752

    return CAM_MAT, DIST_MAT, HEIGHT, WIDTH


# Chose camera parameters
# camera = "real"
camera = "simulation"

CAM_MAT, DIST_MAT, HEIGHT, WIDTH = load_camera_param(cam=camera)


# Define path to data and to specific data set
DATA_PATH = "/Users/jan/Programming/PycharmProjects/master/3d_sets"
# DATA_PATH = "/Volumes/Transfer/master/3d_sets"
# DATA_SET = "reconstruction_1"
# DATA_SET = "reconstruction_2"
DATA_SET = "simulation_8"

try:
    DATA_SET = DATA_SET[-1] + str(os.environ['SIM'])
except KeyError:
    pass

DATA_PATH = os.path.join(DATA_PATH, DATA_SET)

print("[INFO] Starting 3D Label Generator for :", DATA_PATH)
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
dsxy = 15
dddd = 5
dcompat = 8

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
