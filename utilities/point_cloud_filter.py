"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:


"""

import numpy as np
from .general_functions import turn_ply_to_npy, turn_npy_to_ply
import open3d as o3d


@turn_ply_to_npy
@turn_npy_to_ply
def move_along_normals(points: np.ndarray, normals: np.ndarray, step: float) -> np.ndarray:
    """
    Move all points allong their respective normals
    :param points:
    :param normals:
    :param step:
    :return:
    """
    filtered = points + normals * step

    return filtered


@turn_ply_to_npy
@turn_npy_to_ply
def reorient_normals(normals: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Align all normals in the same direction as the given direction vector
    :param normals:
    :param direction:
    :return:
    """
    if len(direction.shape):
        direction = direction[:, np.newaxis]
    dot_product = normals.dot(direction)
    normals[np.where(dot_product < 0)[0], :] *= -1
    return normals


@turn_ply_to_npy
@turn_npy_to_ply
def delete_below(points: np.ndarray, threshold: float) -> np.ndarray:
    """
    Delete all points below threshold
    :param points:
    :param threshold:
    :return:
    """
    filtered = points.copy()
    mask = points[:, 2] < threshold
    filtered = np.delete(filtered, mask, 0)

    return filtered


@turn_ply_to_npy
@turn_npy_to_ply
def delete_above(points: np.ndarray, threshold: float) -> np.ndarray:
    """
    Delete all points above threshold
    :param points:
    :param threshold:
    :return:
    """
    filtered = points.copy()
    mask = points[:, 2] > threshold
    filtered = np.delete(filtered, mask, 0)

    return filtered


@turn_ply_to_npy
@turn_npy_to_ply
def delete_radius(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Delete all points outside the radius
    :param points:
    :param radius:
    :return:
    """
    filtered = points.copy()
    distance = np.linalg.norm(points[:, 0:2], axis=1)
    mask = distance > radius
    filtered = np.delete(filtered, mask, 0)

    return filtered


def remove_statistical_outliers(cloud: o3d.geometry.PointCloud,
                                nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    """
    Wrapping around remove_statistical_outliers function
    :param cloud:
    :param nb_neighbors:
    :param std_ratio:
    :return:
    """
    return cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)[0]