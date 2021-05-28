"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

from settings import DATA_PATH
from utilities import IoU

import glob
import os
import cv2
import shutil
from datetime import datetime

import numpy as np


experiment_name = "perfect_pc_local"
label_names = [os.path.basename(k) for k in glob.glob(f'{DATA_PATH + "/images/"}*.png')]

global_per_instance = np.zeros(3)
global_mean = 0
count = 0
for label_name in label_names:
    label = cv2.imread(f'{DATA_PATH}/labels/{label_name}', 0)
    ground_truth = cv2.imread(f'{DATA_PATH}/ground_truth/{label_name}', 0)

    if label.any():
        count += 1
        per_instance, mean = IoU(label=label, ground_truth=ground_truth)
        global_per_instance += per_instance
        global_mean += mean

    if mean < 0.6:
        print(mean)
        print(label_name)
global_per_instance /= count
global_mean /= count

IoU_string = f"""
                IoU Background: {global_per_instance[0]}
                IoU Stone:      {global_per_instance[1]}
                IoU Border:     {global_per_instance[2]}
                mIoU:           {global_mean}
             """
print(IoU_string)

now = datetime.now().strftime("%d_%m_%H_%M")

if experiment_name is not None:
    name = experiment_name
else:
    name = now
shutil.copy(f'{os.getcwd()}/settings/settings.py',
            f'{DATA_PATH}/results/{name}.txt')

with open(f'{DATA_PATH}/results/{name}.txt', 'a') as file:
    file.write("\a")
    file.write(IoU_string)

IoU_dirt = {"Background": global_per_instance[0],
            "Stone": global_per_instance[1],
            "Border": global_per_instance[2],
            "Mean": global_mean}
np.save(f'{DATA_PATH}/results/{name}.npy', IoU_dirt)
