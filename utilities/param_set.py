"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:


"""
import numpy as np

CAM_MAT = np.array([[1577.1159987660135, 0, 676.7292997380368],
                    [0, 1575.223362703865,  512.8101184300463],
                    [0, 0, 1]])

DIST_MAT = np.array([-0.46465317710098897, 0.2987490394355827, 0.004075959465516531, 0.005311175696501367])
# k1: -0.46465317710098897, k2: 0.2987490394355827, p1: 0.004075959465516531, p2: 0.005311175696501367
HEIGHT = 1080
WIDTH = 1440

DATA_PATH = "/Users/jan/Box/Period 2/Simulation/project/3d_segmentation/data"