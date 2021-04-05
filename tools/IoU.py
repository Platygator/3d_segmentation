"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

from settings import DATA_SET, DATA_PATH
from utilities import IoU
import glob
import os
import cv2
import numpy as np

label_names = [os.path.basename(k) for k in glob.glob(f'{DATA_PATH + "/" + DATA_SET + "/labels/"}*.png')]

global_per_instance = np.zeros(3)
global_mean = 0
count = 0
for label_name in label_names:
    label = cv2.imread(f'{DATA_PATH + "/" + DATA_SET + "/labels/" + label_name}', 0)
    ground_truth = cv2.imread(f'{DATA_PATH + "/" + DATA_SET + "/ground_truth/" + label_name}', 0)

    if label.any():
        count += 1
        per_instance, mean = IoU(label=label, ground_truth=ground_truth)
        global_per_instance += per_instance
        global_mean += mean

global_per_instance /= count
global_mean /= count

print("IoU per instance: ", global_per_instance)
print("mIoU: ", global_mean)
