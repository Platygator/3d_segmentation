"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import os
import shutil
from settings import DATA_PATH, DATA_SET

directory = DATA_PATH + "/" + DATA_SET + "/"

shutil.rmtree(directory + 'masks')
os.mkdir(directory + "masks")

shutil.rmtree(directory + 'labels')
os.mkdir(directory + "labels")
