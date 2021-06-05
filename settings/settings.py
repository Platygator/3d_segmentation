"""
Created by Jan Schiffeler on 05.04.21
jan.schiffeler[at]gmail.com

Settings file for 3D Label Generator
"""
import os
import numpy as np


def load_camera_param(cam):
    if cam == "real":
        cam_mat = np.array([[1577.1159987660135, 0, 676.7292997380368],
                            [0, 1575.223362703865,  512.8101184300463],
                            [0, 0, 1]])

        dist_mat = np.array([-0.46465317710098897, 0.2987490394355827, 0.004075959465516531, 0.005311175696501367])
        # k1: -0.46465317710098897, k2: 0.2987490394355827, p1: 0.004075959465516531, p2: 0.005311175696501367
        # height = 1080
        # width = 1440
        height = 1080
        width = 1440
    elif cam == "simulation":
        cam_mat = np.array([[455, 0, 376],
                            [0, 455, 240],
                            [0.0, 0, 1]])

        dist_mat = np.array([0.0, 0.0, 0.0, 0.0])
        height = 480
        width = 752
    elif cam == "real_small":
        cam_mat = np.array([[823.60502158, 0, 353.4030787530269],
                            [0, 700.09927231, 227.91560819021072],
                            [0.0, 0, 1]])

        dist_mat = np.array([0.0, 0.0, 0.0, 0.0])
        height = 480
        width = 752
    elif cam == "ximea":
        cam_mat = np.array([[386.5767965, 0, 368.22744112],
                            [0, 329.84952469, 236.82126874],
                            [0.0, 0, 1]])

        dist_mat = np.array([0.0, 0.0, 0.0, 0.0])
        height = 480
        width = 752

    return cam_mat, dist_mat, height, width


# Chose camera parameters
camera = "real"
# camera = "real_small"
# camera = "ximea"
# camera = "simulation"

CAM_MAT, DIST_MAT, HEIGHT, WIDTH = load_camera_param(cam=camera)

EXPERIMENT_NAME = "real_1"

# Define path to data and to specific data set
DATA_PATH = "/Users/jan/Programming/PycharmProjects/master/3d_sets"
DATA_SET = "real_1_full_sized"
# DATA_SET = "simulation_3"

try:
    DATA_SET = DATA_SET[-1] + str(os.environ['SIM'])
except KeyError:
    pass

DATA_PATH = os.path.join(DATA_PATH, DATA_SET)

print("[INFO] Starting 3D Label Generator for :", DATA_PATH)


# mask generation param
MIN_NUMBER = 5          # minimum number of instanced of one label
GROWTH_RATE = 5         # number of dilation steps
SHRINK_RATE = 5         # number of erosion steps after dilation
LARGEST_ONLY = True     # use only the largest connected region for mask generation
FILL = True             # fill holes for mask generation
BLUR = 15                # blur applied to regions (region dependent)
BLUR_THRESH = 0       # cutting off region here in binarization step

# CRF PARAM
TIMES = 6               # repetitions of CRF
GSXY = 95                # standard deviation smoothness pixel position
GCOMPAT = 1             # class compatibility gaussian
BSXY = 58               # standard deviation colour ref pixel position
BRGB = 3                # standard deviation colour
BCOMPAT = 75            # class compatibility bilateral colour
DSXY = 74               # standard deviation depth ref pixel position
DDDD = 20                # standard deviation depth
DCOMPAT = 22            # class compatibility gaussian bilateral depth

# LABEL GENERATION
BORDER_THICKNESS = 3    # thickness of border in final label

# UNKNOWN PARAMETERS
UN_MAX_REFINEMENT_LOSS = 0.5     # percentage size change in refinement to be considered a unknown region
UN_SMALL_THRESH = 500            # unknown class threshold for which a mask is considered a small region

VISUALIZATION = False            # Show clustered point cloud in beginning and save all reprojection images
