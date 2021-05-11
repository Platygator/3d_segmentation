"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import os

# path = '/Users/jan/Programming/PycharmProjects/3d_segmentation/data/'
path = '/Volumes/Transfer/master/3d_sets/'
name = 'simulation_9'

pathn = os.path.join(path, name)

try:
    os.mkdir(pathn)
    os.mkdir(pathn + "/depth")
    os.mkdir(pathn + "/ground_truth")
    os.mkdir(pathn + "/images")
    os.mkdir(pathn + "/labels")
    # os.mkdir(pathn + "/unknown")
    # os.mkdir(pathn + "/masks")
    os.mkdir(pathn + "/pointclouds")
    os.mkdir(pathn + "/results")
except FileExistsError:
    print("[ERROR] There is already a set named like that!")

