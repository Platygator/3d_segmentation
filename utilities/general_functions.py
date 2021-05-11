"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by

General utility functions, data loader, wrapper, etc

Python 3.8
Library version:


"""
import numpy as np
import open3d as o3d

import cv2
import os
from glob import glob
from settings import DATA_PATH
from .image_utilities import read_depth_map

IMAGES = "/images/"
DEPTH = "/depth/"
POSE_DIC = os.path.join(DATA_PATH, "positions.npy")
POSE_DIC = np.load(POSE_DIC, allow_pickle=True).item()


def turn_ply_to_npy(func):
    """
    wrapper to transform point_cloud data type to numpy
    :param func: function
    :return: wrapped function
    """

    def wrap(**kwargs):
        # check keys
        for key, value in kwargs.items():
            if type(value) == o3d.utility.Vector3dVector:
                    kwargs[key] = np.asarray(value)

        # call function
        result_npy = func(**kwargs)

        # translate back
        # result = o3d.utility.Vector3dVector(result_npy)

        return result_npy

    return wrap


def turn_npy_to_ply(func):
    """
    wrapper to transform numpy to point_cloud data type
    :param func: function
    :return: wrapped function
    """
    def wrap(**kwargs):
        result_npy = func(**kwargs)
        result = o3d.utility.Vector3dVector(result_npy)

        return result

    return wrap


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Calculate the 3D rotation matrix
    :param roll: rot around x
    :param pitch: rot around y
    :param yaw: rot around z
    :return: Rotation matrix of order Z,Y,X
    """
    R_x = np.array([[1,            0,                 0],
                    [0, np.cos(roll), -1 * np.sin(roll)],
                    [0, np.sin(roll),      np.cos(roll)]])

    R_y = np.array([[     np.cos(pitch), 0, np.sin(pitch)],
                    [                 0, 1,             0],
                    [-1 * np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -1 * np.sin(yaw), 0],
                    [np.sin(yaw),      np.cos(yaw), 0],
                    [          0,                0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x)).T


def load_images(data_path: str = DATA_PATH, positions: {str: np.array} = POSE_DIC) -> [np.ndarray, np.ndarray]:
    """
    Generator to load all images and there respective data from the data_path folder
    :param data_path: folder containing depth, images, masks, positions, pointclouds
    :param positions: Colmap provided positions
    :return: yield, image, position, depth_map and name
    """
    instance_names = [os.path.basename(k)[:-4] for k in glob(f'{data_path + IMAGES}*.png')]
    n_img = len(instance_names)
    for i, name in enumerate(instance_names):
        print(f"[INFO] Processing image {i+1} / {n_img}")
        image = cv2.imread(data_path + IMAGES + name + '.png', 1)
        depth_map = read_depth_map(data_path + DEPTH + name + '.png.geometric.bin')
        position = positions[name]
        yield image, position, depth_map, name


def IoU(label: np.ndarray, ground_truth: np.ndarray) -> (np.ndarray, float):
    """
    Calculate Intersection over Union
    :param label: generated label
    :param ground_truth: ground truth to compare to
    :return: IoU per instance and mean IoU
    """
    if ground_truth.max() == 2:
        gt_vals = [0, 1, 2]
    else:
        gt_vals = [0, 128, 255]
    iou_per_instance = np.zeros(3)
    # change ground truth at unknown spots to unknown
    ground_truth[label == 50] = 50
    for i, (instance_lab, instance_gt) in enumerate(zip([0, 128, 255], gt_vals)):
        org_instance = np.zeros_like(ground_truth)
        org_instance[np.where(ground_truth == instance_gt)] = 1
        rec_instance = np.zeros_like(label)
        rec_instance[np.where(label == instance_lab)] = 1

        intersection = np.logical_and(org_instance, rec_instance).astype('uint8')
        union = np.logical_or(org_instance, rec_instance).astype('uint8')
        iou_per_instance[i] = np.sum(intersection) / np.sum(union)

    return iou_per_instance, np.mean(iou_per_instance)
