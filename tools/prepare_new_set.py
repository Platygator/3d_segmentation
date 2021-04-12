"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import os

path = '/Users/jan/Programming/PycharmProjects/3d_segmentation/data/'
name = 'simulation_5'

pathn = path + name

try:
    os.mkdir(pathn)
    os.mkdir(pathn + "/depth")
    os.mkdir(pathn + "/ground_truth")
    os.mkdir(pathn + "/images")
    os.mkdir(pathn + "/labels")
    os.mkdir(pathn + "/masks")
    os.mkdir(pathn + "/pointclouds")
    os.mkdir(pathn + "/results")
except FileExistsError:
    print("[ERROR] There is already a set named like that!")

