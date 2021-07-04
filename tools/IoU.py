"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

from settings import DATA_PATH, EXPERIMENT_NAME
from utilities import IoU

import glob
import os
import cv2
import shutil
from datetime import datetime
import argparse
import numpy as np


def create_argparser():
    parser = argparse.ArgumentParser(description='Chose images main_path')
    parser.add_argument('-l', '--label', required=False, help='path to labels')
    parser.add_argument('-g', '--ground_truth', required=False, help='path to ground_truth')
    parser.add_argument('-e', '--experiment_name', required=False, help='name of experiment')
    args = vars(parser.parse_args())
    return args


arg = create_argparser()

if arg['label']:
    label_path = os.path.join(arg['label'], '')
else:
    label_path = f'{DATA_PATH}/labels/'

if arg['ground_truth']:
    ground_truth_path = os.path.join(arg['ground_truth'], '')
else:
    ground_truth_path = f'{DATA_PATH}/ground_truths/'

if arg['ground_truth']:
    experiment_name = arg['experiment_name']
else:
    experiment_name = EXPERIMENT_NAME


label_names = [os.path.basename(k) for k in glob.glob(f'{label_path}*.png')]

global_per_instance = np.zeros(3)
global_obs = []
global_obs_class = []
count = 0
for label_name in label_names:
    if label_name[:2] == "v_":
        continue
    label = cv2.imread(label_path + label_name, 0)
    ground_truth = cv2.imread(ground_truth_path + label_name, 0)
    if label is None:
        print(label_name)
        continue
    if label.any():
        count += 1
        per_instance, mean = IoU(label=label, ground_truth=ground_truth)
        global_per_instance += per_instance
        global_obs.append(mean * 100)
        global_obs_class.append(per_instance * 100)

global_per_instance /= count
global_mean = sum(global_obs) / count

IoU_string = f"""
                IoU Background: {global_per_instance[0]}
                IoU Stone:      {global_per_instance[1]}
                IoU Border:     {global_per_instance[2]}
                mIoU:           {global_mean}
             """
print(IoU_string)

global_obs_class = list(map(list, zip(*global_obs_class)))
save_dict = {"Background": global_obs_class[0],
             "Stone": global_obs_class[1],
             "Border": global_obs_class[2],
             "Mean": global_obs}
np.save(f'results/{experiment_name}.npy', save_dict)

# now = datetime.now().strftime("%d_%m_%H_%M")
# 
# if EXPERIMENT_NAME is not None:
#     name = EXPERIMENT_NAME
# else:
#     name = now
# shutil.copy(f'{os.getcwd()}/settings/settings.py',
#             f'{DATA_PATH}/results/{name}.txt')
# 
# with open(f'{DATA_PATH}/results/{name}.txt', 'a') as file:
#     file.write("\a")
#     file.write(IoU_string)
# 
# IoU_dirt = {"Background": global_per_instance[0],
#             "Stone": global_per_instance[1],
#             "Border": global_per_instance[2],
#             "Mean": global_mean}
# np.save(f'{DATA_PATH}/results/{name}.npy', IoU_dirt)
