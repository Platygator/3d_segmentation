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

name = "02_01_00600"
normal_map = read_depth_map(DATA_PATH + "/normals/" + name + '.png.geometric.bin')
np.save("debug_images/normal.npy", normal_map)
plt.imshow(normal_map)
plt.show()

print("Done")
