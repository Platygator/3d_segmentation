"""
Created by Jan Schiffeler on 31.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2

from scipy import ndimage

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax


def crf_refinement(img: np.ndarray, mask: np.ndarray, depth: np.ndarray, times: int,
                   gsxy: int, gcompat: int,
                   bsxy: int, brgb: int, bcompat: int,
                   dsxy: int, dddd: int, dcompat: int,
                   n_classes: int) -> np.ndarray:
    """
    Based on this dudes code: https://github.com/seth814/Semantic-Shapes/blob/master/CRF%20Cat%20Demo.ipynb
    :param img: original image
    :param mask: label mask
    :param times: repetitions
    :param n_classes: number of classes (actually always 2 here)
    :return: refined label image
    """

    n_classes += 1
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)

    # TODO play with ground_truth probability
    unary = unary_from_labels(mask, n_classes, 0.7, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=gsxy, compat=gcompat, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=bsxy, srgb=brgb, rgbim=img,
                           compat=bcompat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    # TODO work on this - maybe use normalization and cut of extremes
    depth * 255/depth.max()
    depth = depth.astype('uint8')
    depth = depth[:, :, np.newaxis].repeat(3, axis=2)
    d.addPairwiseBilateral(sxy=dsxy, srgb=dddd, rgbim=depth,
                           compat=dcompat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(times)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res


def graph_cut_refinement(img: np.ndarray, mask: np.ndarray, iter_count: int) -> np.ndarray:
    mask[mask > 0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD
    fg_model = np.zeros((1, 65), dtype="float")
    bg_model = np.zeros((1, 65), dtype="float")
    label, bg_model, fg_model = cv2.grabCut(img, mask, None, bg_model,
                                            fg_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_MASK)

    label[label == 2] = 0
    label[label == 3] = 255
    return label


def largest_region(mask: np.ndarray) -> np.ndarray:
    # TODO maybe differentiate between border rocks and center rocks
    # instance = np.pad(instance, [[1, 1], [1, 1]], constant_values=1)

    if not mask.any():
        return mask
    connected, _ = ndimage.label(mask > 0)
    uni, count = np.unique(connected, return_counts=True)
    uni = np.delete(uni, np.argmax(count))
    count = np.delete(count, np.argmax(count))
    largest_label = uni[np.argmax(count)]

    largest_region = np.zeros_like(mask, dtype='uint8')
    largest_region[np.where(connected == largest_label)] = 255

    # largest_region = largest_region[1: -1, 1: -1]

    return largest_region


def fill_holes(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    # mask = np.pad(mask, [[1, 1], [1, 1]], constant_values=1)
    mask = cv2.bitwise_not(mask)

    connected, _ = ndimage.label(mask > 0)
    uni, count = np.unique(connected, return_counts=True)
    uni = np.delete(uni, np.argmax(count))

    closed_holes = np.zeros_like(mask, dtype='uint8')
    for n in uni:
        closed_holes[np.where(connected == n)] = 255

    # closed_holes = closed_holes[1:-1, 1:-1]

    return closed_holes


def read_depth_map(path: str) -> np.ndarray:
    """
    Copied from colmap scripts
    For copyright see: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_dense.py
    :param path:
    :return:
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()