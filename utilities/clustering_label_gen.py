"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by



Python 3.8
Library version:


"""
import open3d as o3d
import cv2
from sklearn.cluster import KMeans as km

from .general_functions import turn_ply_to_npy
from .param_set import *

from os import mkdir

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax


def crf_inference_label(img, mask, t, n_classes):
    """
    Based on this dudes code: https://github.com/seth814/Semantic-Shapes/blob/master/CRF%20Cat%20Demo.ipynb
    :param img:
    :param mask:
    :param t:
    :param n_classes:
    :return:
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
    return res


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
              cam_mat: np.ndarray = CAM_MAT, dist_map: np.ndarray = DIST_MAT,
              height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """
    Project all point cloud points into the image scene pixel points
    :param points:
    :param color:
    :param label:
    :param transformation_mat:
    :param depth_map:
    :param cam_mat:
    :param dist_mat:
    :param height:
    :param width:
    :return:
    """

    rvec = cv2.Rodrigues(transformation_mat[:3, :3])
    cv_projection = cv2.projectPoints(objectPoints=points, rvec=rvec[0], tvec=transformation_mat[:3, 3], cameraMatrix=cam_mat,
                                      distCoeffs=dist_map)

    pixels_cv = np.rint(cv_projection[0]).astype('int')

    save_index = np.zeros([height, width], dtype='uint')
    for i, pixel in enumerate(pixels_cv):
        pixel = pixel[0]
        if 0 < pixel[0] < height and 0 < pixel[1] < width:
            save_index[pixel[0], pixel[1]] = i+1

    # TODO add distance check for occlusion
    # based on the closest index, select the respective color
    color = np.concatenate([np.zeros([1, 3]), color], axis=0)
    label = np.concatenate([np.zeros([1]), label+1], axis=0)
    reprojection_visual = color[save_index]
    reprojection = label[save_index]
    cv2.imwrite("label_projection.png", reprojection)
    cv2.imwrite("visual_projection.png", np.floor(reprojection_visual*255).astype('uint8'))
    return reprojection


def generate_label(projection: np.ndarray, original: np.ndarray, growth_rate: int, name: str,
                   min_number: int, data_path: str = DATA_PATH) -> np.ndarray:
    labels_present, counts = np.unique(projection, return_counts=True)
    filter_small = counts < min_number
    labels_present = np.delete(labels_present, filter_small, 0)
    labels_present = np.delete(labels_present, 0)

    for i in labels_present:
        mask_dir = data_path + f"/masks/mask{i+1}"
        try:
            mkdir(mask_dir)
        except FileExistsError:
            pass

    for i, label in enumerate(labels_present):
        instance = np.zeros_like(projection)
        instance[np.where(projection == label)] = 255
        instance = cv2.dilate(instance, np.ones((5, 5), 'uint8'), iterations=growth_rate)
        label = crf_inference_label(original, instance, 10, 2) * 255
        label = label.astype('uint8')

        mask_dir = data_path + f"/masks/mask{i+1}/"

        cv2.imwrite(mask_dir + name + ".png", label)
