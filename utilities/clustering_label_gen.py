"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by

Different clustering and label generation functions

Python 3.8
Library version:


"""

import open3d as o3d
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from .general_functions import turn_ply_to_npy
from settings import CAM_MAT, DIST_MAT, WIDTH, HEIGHT
import numpy as np


@turn_ply_to_npy
def km_cluster(points: np.ndarray, k: int) -> [o3d.utility.Vector3dVector, np.ndarray]:
    """
    Using k-means to cluster point cloud
    :param points:
    :param k:
    :return:
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    label = kmeans.labels_
    np.random.seed(5)
    rand_col = np.random.random([k, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def dbscan_cluster(points: np.ndarray, epsilon: float) -> [o3d.utility.Vector3dVector, np.ndarray]:
    """
    Using dbscan to cluster point cloud
    :param points:
    :return:
    """
    db = DBSCAN(eps=epsilon, min_samples=2)
    print("[INFO] Fitting DBSCAN")
    db.fit(points)
    print("[INFO] Done")
    label = db.labels_
    n_clusters_ = len(set(label)) - (1 if -1 in label else 0)
    rand_col = np.random.random([n_clusters_, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def reproject(points: np.ndarray, color: np.ndarray, label: np.ndarray,
              transformation_mat: np.ndarray, depth_map: np.ndarray,
              name: str,
              cam_mat: np.ndarray = CAM_MAT, dist_mat: np.ndarray = DIST_MAT,
              height: int = HEIGHT, width: int = WIDTH, save_img: bool = False) -> np.ndarray:
    """
    Project all point cloud points into the image scene pixel points
    :param points: 3D point coordinates
    :param color: color vector for each point
    :param label: label for each point
    :param transformation_mat: transformation of the current camera location
    :param depth_map: depth map of image
    :param name: image name
    :param cam_mat: camera matrix
    :param dist_mat: distortion matrix [k1, k2, p1, p2]
    :param height: img height
    :param width: img width
    :param save_img: save label and visual image for debugging
    :return: image where each pixel has a label [0 = background]
    """

    # TODO This projection technique was designed to handle sparse point clouds! When projecting the points back,
    #      occlusion is not directly included so all points in frame will be written to the image plane in random order.
    #      This means previously projected points might be overwritten by newer ones. With sparse point clouds this
    #      basically never happened and could savely be ignored, but the dense point clouds might be problematic.
    #      I suggest to project each label onto its own matrix. This adaptation will not be made as the presentation
    #      of the thesis is already completed and the results should be reproducable. Maybe it is also not a problem
    #      after all, since the morphological operations later on could just mask over it.
    R = transformation_mat[:3, :3]
    camera_center = transformation_mat[:3, 3]
    real_center = -1 * R.T.dot(camera_center.T)

    rvec = cv2.Rodrigues(R)
    cv_projection = cv2.projectPoints(objectPoints=points, rvec=rvec[0], tvec=transformation_mat[:3, 3],
                                      cameraMatrix=cam_mat, distCoeffs=dist_mat)

    pixels = np.rint(cv_projection[0]).astype('int')

    distance_map = points - np.repeat(real_center[np.newaxis, :], points.shape[0], axis=0)
    distance_map = np.linalg.norm(distance_map, axis=1)
    # distance_map = distance_map[:, 2]

    save_distance = np.zeros([height, width])

    save_index = np.zeros([height, width], dtype='uint')

    # TODO vectorize
    for i, pixel in enumerate(pixels):
        y, x = pixel[0]
        if 0 <= x < height and 0 <= y < width:
            dist = distance_map[i]
            save_distance[x, y] = dist
            save_index[x, y] = i+1

    # based on the index select the respective index
    label = np.concatenate([np.zeros([1]), label+1], axis=0)
    reprojection = label[save_index]

    if save_img:
        print("[INFO] Saving debug images")
        cv2.imwrite(f"debug_images/label_projection_{name}.png", reprojection)

        # based on the index select the respective colour
        color = np.concatenate([np.zeros([1, 3]), color], axis=0)
        visual_label_img = color[save_index]

        visual_label_img = cv2.cvtColor(np.floor(visual_label_img*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"debug_images/visual_projection_{name}.png", visual_label_img)

        np.save(f"debug_images/{name}_distance.npy", save_distance)
        np.save(f"debug_images/{name}_depth.npy", depth_map)

    return reprojection, save_distance



