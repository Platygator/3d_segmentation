"""
Created by Jan Schiffeler on 07.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:

TODO:
    - dilate unknown region
    - get all regions before and after refinement to reimplement refinement loss check
"""

import numpy as np
from scipy import ndimage
import cv2


class UnknownRegister:
    """Class to remember all unknown regions in an image judged by different criteria"""

    def __init__(self, width, height, small_treshold, max_refinement_loss):
        """Constructor for UnknownRegister
        Parameters
        ----------

        """
        self._width = width
        self._height = height
        self._label = np.zeros([height, width], dtype='uint8')
        self._small_treshold = small_treshold
        self._max_refinement_loss = max_refinement_loss
        self._unknown_label = 50

    def register_label_img(self, label: np.ndarray):
        self._label = label

    def new_label(self):
        self._label = np.zeros([self._height, self._width], dtype='uint8')

    def retrieve_label_img(self):
        self._label = cv2.dilate(self._label, np.ones((3, 3), 'uint8'), iterations=1)
        return self._label

    def refinement_lost(self, before: np.ndarray, after: np.ndarray):
        """
        Check if a region is lost during refinement
        :param before:
        :param after:
        :return:
        """
        for label in np.unique(before):
            if label == 0:
                continue
            before_inst = np.zeros_like(before, dtype='float16')
            before_inst[before == label] = 1
            after_inst = np.zeros_like(after, dtype='float16')
            after_inst[after == label] = 1
            diff = abs(before_inst.nonzero()[0].shape[0] - after_inst.nonzero()[0].shape[0])\
                       / before_inst.nonzero()[0].shape[0]
            if diff >= self._max_refinement_loss:
                self._label[before_inst.nonzero()] = self._unknown_label

    def small_region(self, region: np.ndarray):
        if region.nonzero()[0].shape[0] <= self._small_treshold:
            self._label[region.nonzero()] = self._unknown_label

    def disconnected_patches(self, region: np.ndarray):
        if region.any():
            connected, _ = ndimage.label(region > 0)
            uni, count = np.unique(connected, return_counts=True)
            uni = np.delete(uni, np.argmax(count))
            count = np.delete(count, np.argmax(count))
            largest_label = uni[np.argmax(count)]

            unknown_region = np.zeros_like(region, dtype='uint8')
            unknown_region[np.where(connected != largest_label)] = 1
            unknown_region[np.where(connected == 0)] = 0

            self._label[unknown_region == 1] = self._unknown_label

    def holes(self, region: np.ndarray):
        if region.any():
            if region.max() == 1:
                region = region * 255
            mask = np.pad(region, [[1, 1], [1, 1]], constant_values=255)
            mask = cv2.bitwise_not(mask)

            connected, _ = ndimage.label(mask > 0)
            connected = connected[1:-1, 1:-1]
            uni, count = np.unique(connected, return_counts=True)
            uni = np.delete(uni, np.argmax(count))
            count = np.delete(count, np.argmax(count))
            uni = np.delete(uni, np.argmax(count))

            for n in uni:
                self._label[np.where(connected == n)] = self._unknown_label


if __name__ == '__main__':
    import cv2
    ur = UnknownRegister(width=50, height=50, small_treshold=700, max_refinement_loss=0.5)
    before = np.zeros([50, 50], dtype='uint8')
    before[(15, 15, 30, 30), (10, 13, 25, 26)] = 255
    before = cv2.dilate(before, np.ones((5, 5), 'uint8'), iterations=5)

    after = np.zeros([50, 50], dtype='uint8')
    after[(20, 20, 40, 40), (12, 17, 25, 26)] = 255
    after = cv2.dilate(after, np.ones((5, 5), 'uint8'), iterations=4)
    after[10, 10] = 255

    ur.register_label_img(before)
    un = ur.retrieve_label_img()
    ur.small_region(after)
    un = ur.retrieve_label_img()
    ur.disconnected_patches(after)
    un = ur.retrieve_label_img()
    ur.refinement_lost(before, after)
    un = ur.retrieve_label_img()
    print("Done")
