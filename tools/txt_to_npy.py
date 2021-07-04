"""
Created by Jan Schiffeler on 29.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import numpy as np
from settings import DATA_PATH

colmap_images_file = f"{DATA_PATH}/images.txt"
colmap_cameras_file = f"{DATA_PATH}/cameras.txt"

image_dict = {}
# endings = ['09_05_' + str(k).zfill(5) for k in range(0, 7020, 20)]
# bl = ['bl_' + str(k).zfill(12) for k in range(0, 7020, 20)]
#
# mapping = {text: folder for text, folder in zip(bl, endings)}


with open(colmap_images_file) as file:
    for line in file.readlines():
        if line.endswith(".png\n"):

            _, rw, rx, ry, rz, tx, ty, tz, _, name = line.split(" ")
            vec = np.array([float(k) for k in [rw, rx, ry, rz, tx, ty, tz]])
            name = name[:-5]
            # name = mapping[name]
            image_dict[name] = vec

np.save(f"{DATA_PATH}/positions.npy", image_dict)

with open(colmap_cameras_file) as file:
    line = [k for k in file.readlines() if k != ""][-1]
    try:
        _, _, width, height, fx, fy, cx, cy = line.split(" ")
    except ValueError:
        _, _, width, height, fx, cx, cy = line.split(" ")
        fy = fx

cam_mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype="float32")

# TODO In order to deal with size variations in depth and normal maps in colmap when using distorted images,
#  the undistorting is now handled before colmap and thus the distortion coefficients are set to 0, but this could be
#  changed here

cam_dict = {"width": int(width), "height": int(height), "cam_mat": cam_mat, "dist_mat": np.array([[0.0, 0, 0, 0]])}
np.save(f"{DATA_PATH}/camera_info.npy", cam_dict)
