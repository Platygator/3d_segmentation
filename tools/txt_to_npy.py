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

save_dic = {}
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
            save_dic[name] = vec

np.save(f"{DATA_PATH}/positions.npy", save_dic)
