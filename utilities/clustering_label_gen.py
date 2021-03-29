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
from sklearn.mixture import GaussianMixture

from .general_functions import turn_ply_to_npy
from .live_camera_parameters import *

from os import mkdir

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax

try:
    mkdir(DATA_PATH + "/masks")
except FileExistsError:
    pass


def crf_inference_label(img: np.ndarray, mask: np.ndarray, t: int, n_classes: int) -> np.ndarray:
    """
    Based on this dudes code: https://github.com/seth814/Semantic-Shapes/blob/master/CRF%20Cat%20Demo.ipynb
    :param img: original image
    :param mask: label mask
    :param t: UNKNOWN! TODO find out
    :param n_classes: number of classes (actually always 2 here)
    :return: refined label image
    """

    not_mask = cv2.bitwise_not(mask)
    not_mask = np.expand_dims(not_mask, axis=2)
    mask = np.expand_dims(mask, axis=2)
    im_softmax = np.concatenate([not_mask, mask], axis=2)
    im_softmax = im_softmax / 255.0

    feat_first = im_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=10, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(t)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    res *= 255
    return res.astype('uint8')


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
    rand_col = np.random.random([k, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def gmm_cluster(points: np.ndarray):
    # TODO write method
    gausmmix = GaussianMixture(n_components=3, random_state=0)
    gausmmix.fit(points)


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
              depth_range: float, distance_map: np.ndarray,
              cam_mat: np.ndarray = CAM_MAT, dist_mat: np.ndarray = DIST_MAT,
              height: int = HEIGHT, width: int = WIDTH, save_img: bool = False) -> np.ndarray:
    """
    Project all point cloud points into the image scene pixel points
    :param points: 3D point coordinates
    :param color: color vector for each point
    :param label: label for each point
    :param transformation_mat: transformation of the current camera location
    :param depth_map: depth map of image
    :param depth_range: acceptable deviation between distance and depth
    :param distance_map: distance from all points to camera center point
    :param cam_mat: camera matrix
    :param dist_mat: distortion matrix [k1, k2, p1, p2]
    :param height: img height
    :param width: img width
    :param save_img: save label and visual image for debugging
    :return: image where each pixel has a label [0 = background]
    """

    rvec = cv2.Rodrigues(transformation_mat[:3, :3])
    cv_projection = cv2.projectPoints(objectPoints=points, rvec=rvec[0], tvec=transformation_mat[:3, 3],
                                      cameraMatrix=cam_mat, distCoeffs=dist_mat)

    pixels_cv = np.rint(cv_projection[0]).astype('uint8')

    save_index = np.zeros([height, width], dtype='uint8')
    for i, pixel in enumerate(pixels_cv):
        x, y = pixel[0]
        if 0 < x < height and 0 < y < width:
            # TODO check if distance check for occlusion is working
            dist = distance_map[x, y]
            depth = depth_map[x, y]
            if abs(dist - depth) <= depth_range:
                save_index[x, y] = i+1

    # based on the closest index, select the respective color
    color = np.concatenate([np.zeros([1, 3]), color], axis=0)
    label = np.concatenate([np.zeros([1]), label+1], axis=0)
    reprojection = label[save_index]
    if save_img:
        reprojection_visual = color[save_index]
        cv2.imwrite("label_projection.png", reprojection)
        cv2.imwrite("visual_projection.png", np.floor(reprojection_visual*255).astype('uint8'))
    return reprojection


def generate_label(projection: np.ndarray, original: np.ndarray, growth_rate: int, name: str,
                   min_number: int, data_path: str = DATA_PATH):
    """
    Generate masks for each label instance (similar to blender mask output)
    :param projection: image with instance label for each pixel [0 = background]
    :param original: original image
    :param growth_rate: number of dilation steps
    :param name: name of the current image
    :param min_number: minimum number of instanced of one label
    :param data_path: folder containing depth, images, masks, positions, pointclouds
    :return: save mask images to separate folders in masks
    """
    labels_present, counts = np.unique(projection, return_counts=True)
    filter_small = counts < min_number
    labels_present = np.delete(labels_present, filter_small, 0)
    labels_present = np.delete(labels_present, 0)

    for i in labels_present:
        mask_dir = data_path + f"/masks/mask{int(i)}"
        try:
            mkdir(mask_dir)
        except FileExistsError:
            pass
        instance = np.zeros_like(projection)
        instance[np.where(projection == i)] = 255
        instance = cv2.dilate(instance, np.ones((5, 5), 'uint8'), iterations=growth_rate)
        label = crf_inference_label(original, instance, 10, 2)

        cv2.imwrite(mask_dir + name + ".png", label)
