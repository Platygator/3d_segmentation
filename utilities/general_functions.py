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
from os.path import basename
# from os import mkdir
from glob import glob
from .live_camera_parameters import DATA_PATH, DATA_SET

IMAGES = "/images/"
DEPTH = "/depth/"
POSE_DIC = DATA_PATH + "/" + DATA_SET + "/positions.npy"
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


def load_images(data_path: str = DATA_PATH + "/" + DATA_SET,
                positions: {str : np.array} = POSE_DIC) -> [np.ndarray, np.ndarray]:
    """
    Generator to load all images and there respective data from the data_path folder
    :param data_path: folder containing depth, images, masks, positions, pointclouds
    :return: yield, image, position, depth_map and name
    """
    instance_names = [basename(k)[:-4] for k in glob(f'{data_path + IMAGES}*.png')]
    for name in instance_names:
        image = cv2.imread(data_path + IMAGES + name + '.png', 1)
        depth_map = cv2.imread(data_path + DEPTH + name + '.png', 0)
        position = positions[name]
        yield image, position, depth_map, name


# def save_label(label_name: str, label: np.ndarray, data_path: str = DATA_PATH):
#     cv2.imwrite(data_path + label_name + '.png', label)
