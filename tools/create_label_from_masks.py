"""
Created by Jan Schiffeler on 01.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import cv2
import numpy as np
import glob
import os
from settings import DATA_PATH

thickness = 3

IMAGES = "/images/"
MASKS = "/masks/"
LABELS = "/labels/"


class GroundTruthGenerator:
    def __init__(self, height: int, width: int, path: str, border_thickness: int,
                masks: str = MASKS):
        self.height = height
        self.width = width
        self.path = path + masks
        self.border_thickness = border_thickness
        self.kernel = np.array([[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]])

    def process_image(self, name: str) -> np.ndarray:

        segments_sum = np.zeros((self.height, self.width), dtype='uint8')
        border_sum = np.zeros((self.height, self.width), dtype='uint8')

        mask_directories = [k[0] for k in os.walk(self.path)][1:]

        for directory in mask_directories:
            mask_name = directory + "/" + name
            mask_img = cv2.imread(mask_name, 0)
            if mask_img is not None:
                if mask_img.any():
                    mask_img = np.rint(mask_img / 255).astype('uint8')
                    segments_sum += mask_img

                    border = cv2.filter2D(mask_img, -1, self.kernel)
                    border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=self.border_thickness)
                    # # make sure the border is on top of stone
                    border *= mask_img
                    # turn into 0 and 1
                    border = np.array(border, dtype=bool).astype("uint8")

                    border_sum += border

        # ground_truth_image = 255 * segments_sum - 127 * border_sum
        ground_truth_image = segments_sum + border_sum
        ground_truth_image = ground_truth_image.astype('uint8')
        return ground_truth_image


photo_images = [os.path.basename(k) for k in glob.glob(f'{DATA_PATH + IMAGES}*.png')]
image_number = len(photo_images)

# initialize generator
height, width = cv2.imread(DATA_PATH+ IMAGES + photo_images[0], 0).shape
gtg = GroundTruthGenerator(height=height, width=width, path=DATA_PATH,
                           border_thickness=thickness)

for n, image_name in enumerate(photo_images):
    print(f"Processing image {n + 1}/{image_number}")
    ground_truth = gtg.process_image(name=image_name)
    cv2.imwrite(DATA_PATH + LABELS + image_name, np.rint(ground_truth*127.5))
