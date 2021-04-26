"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import os
import shutil
from settings import DATA_PATH

shutil.rmtree(DATA_PATH + '/masks')
os.mkdir(DATA_PATH + "/masks")

shutil.rmtree(DATA_PATH + '/labels')
os.mkdir(DATA_PATH + "/labels")
