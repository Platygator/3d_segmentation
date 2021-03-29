"""
Created by Jan Schiffeler on 29.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:

TODO: - drop all lines not ending on .png
      - find re to get positions
      - save as dictionary

"""

import numpy as np
import re

colmap_images_file = "data/positions/images.txt"

with open(colmap_images_file) as file:
    for line in file.readlines():
        print(line)