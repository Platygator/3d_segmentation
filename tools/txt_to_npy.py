"""
Created by Jan Schiffeler on 29.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import numpy as np

colmap_images_file = "../data/positions/images.txt"

save_dic = {}

with open(colmap_images_file) as file:
    for line in file.readlines():
        if line.endswith(".png\n"):
            _, rw, rx, ry, rz, tx, ty, tz, _, name = line.split(" ")
            vec = np.array([float(k) for k in [rw, rx, ry, rz, tx, ty, tz]])
            name = name[:-5]
            save_dic[name] = vec

np.save("data/positions/reconstruction_2.npy", save_dic)
