"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:


"""
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import KMeans as km

from .general_functions import turn_ply_to_npy, turn_npy_to_ply
from .param_set import *


@turn_ply_to_npy
def km_cluster(points: np.ndarray, k: int) -> [o3d.utility.Vector3dVector, np.ndarray]:
    """
    Using k-means to cluster point cloud
    :param points:
    :param k:
    :return:
    """
    kmeans = km(n_clusters=k)
    kmeans.fit(points)
    label = kmeans.labels_
    rand_col = np.random.random([k, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def reproject(points: np.ndarray, color: np.ndarray, label: np.ndarray,
              transformation_mat: np.ndarray, depth_map: np.ndarray,
              cam_mat: np.ndarray = CAM_MAT, height: int = HEIGHT,
              width: int = WIDTH) -> np.ndarray:
    """
    Project all point cloud points into the image scene pixel points
    :param points:
    :param color:
    :param label:
    :param transformation_mat:
    :param depth_map:
    :param cam_mat:
    :param height:
    :param width:
    :return:
    """

    rvec = cv2.Rodrigues(transformation_mat[:3, :3])
    cv_projection = cv2.projectPoints(objectPoints=points, rvec=rvec[0], tvec=transformation_mat[:3, 3], cameraMatrix=cam_mat,
                                  distCoeffs=DIST_MAT)

    pixels_cv = np.rint(cv_projection[0]).astype('int')

    save_index = np.zeros([height, width], dtype='uint')
    for i, pixel in enumerate(pixels_cv):
        pixel = pixel[0]
        if 0 < pixel[0] < height and 0 < pixel[1] < width:
            save_index[pixel[0], pixel[1]] = i+1


    # based on the closest index, select the respective color
    color = np.concatenate([np.zeros([1, 3]), color], axis=0)
    label = np.concatenate([np.zeros([1]), label+1], axis=0)
    reprojection_visual = color[save_index]
    reprojection = label[save_index]
    cv2.imwrite("label_projection.png", reprojection)
    cv2.imwrite("visual_projection.png", np.floor(reprojection_visual*255).astype('uint8'))
    return reprojection


def generate_label(projection: np.ndarray, original: np.ndarray, growth_rate: int) -> np.ndarray:
    border = cv2.dilate(projection, np.ones((5, 5), 'uint8'), iterations=growth_rate)