"""
Created by Jan Schiffeler on 05.04.21
jan.schiffeler[at]gmail.com

Settings file for 3D Label Generator

Running behaviour is changed on the top of this file.
All parameters are set in their respected .json file. An explanation of all parameters is found on the bottom
of this file.
"""
import os
import numpy as np
import json

# SETTINGS
EXPERIMENT_NAME = "real"        # Name of experiment (relevant mainly for automated IoU calculations)
data_set = "real_9"             # Which data set generate labels from
setting = "real_resized"        # Which settings file to use (expected to be save in /settings/
DATA_PATH = "/Users/jan/Programming/PycharmProjects/master/3d_sets"  # Path to parent folder of data sets
VISUALIZATION = True            # Show clustered point cloud in beginning and save all reprojection images
# SETTINGS


DATA_PATH = os.path.join(DATA_PATH, data_set)

print("[INFO] Starting 3D Label Generator for :", DATA_PATH)

with open(f"settings/{setting}.json") as d:
    settings = json.load(d)

try:
    camera_settings = np.load(f"{DATA_PATH}/camera_info.npy", allow_pickle=True).item()
    WIDTH = camera_settings["width"]
    HEIGHT = camera_settings["height"]
    CAM_MAT = camera_settings["cam_mat"]
    DIST_MAT = camera_settings["dist_mat"]
except FileNotFoundError:
    print("[ERROR] If you're not running txt_to_npy.py, there will be an error soon :)")

# mask generation param
label_settings = settings["label_generation"]
MIN_NUMBER = label_settings["min_number"]           # minimum number of instanced of one label
GROWTH_RATE = label_settings["growth_rate"]         # number of dilation steps
SHRINK_RATE = label_settings["shrink_rate"]         # number of erosion steps after dilation
LARGEST_ONLY = label_settings["largest_only"]       # use only the largest connected region for mask generation
FILL = label_settings["fill"]                       # fill holes for mask generation
BLUR = label_settings["blur"]                       # blur applied to regions (region dependent)
BLUR_THRESH = label_settings["blur_thresh"]         # cutting off region here in binarization step

# CRF PARAM
crf_settings = settings["crf"]
TIMES = crf_settings["times"]        # repetitions of CRF
GSXY = crf_settings["gsxy"]          # standard deviation smoothness pixel position
GCOMPAT = crf_settings["gcompat"]    # class compatibility gaussian
BSXY = crf_settings["bsxy"]          # standard deviation colour ref pixel position
BRGB = crf_settings["brgb"]          # standard deviation colour
BCOMPAT = crf_settings["bcompat"]    # class compatibility bilateral colour
DSXY = crf_settings["dsxy"]          # standard deviation depth ref pixel position
DDDD = crf_settings["dddd"]          # standard deviation depth
DCOMPAT = crf_settings["dcompat"]    # class compatibility gaussian bilateral depth

# LABEL GENERATION
BORDER_THICKNESS = settings["border_thickness"]    # thickness of border in final label

# UNKNOWN PARAMETERS
unknown_detector = settings["unknown_detector"]
UN_MAX_REFINEMENT_LOSS = unknown_detector["max_refinement_loss"]   # percentage size change in refinement to be
                                                                   # considered a unknown region
UN_SMALL_THRESH = unknown_detector["small_threshold"]              # unknown class threshold for which a mask is
                                                                   # considered a small region
