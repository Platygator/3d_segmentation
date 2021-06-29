"""
Created by Jan Schiffeler on 14.02.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import json
import cv2
import numpy as np
import os

height = 480
width = 752

kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]])

with open("debug_images/real_2_gt.json") as f:
# with open("debug_images/own_iou_check.json") as f:
    data = json.load(f)

# extract information form json
images = []
for img in data:
    shapes = []
    segments_sum = np.zeros((height, width), dtype='uint8')
    border_sum = np.zeros((height, width), dtype='uint8')
    for i, r in enumerate(data[img]['regions']):
        if r is None:
            continue
        x_points = r['shape_attributes']['all_points_x']
        y_points = r['shape_attributes']['all_points_y']
        mask = np.zeros([height, width])
        mask = cv2.fillPoly(mask, [np.array([x_points, y_points]).T], (1))
        mask = mask.astype('uint8')

        segments_sum[mask != 0] = i

    #     segments_sum += mask
    #
    #     border = cv2.filter2D(mask, -1, kernel)
    #     border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=3)
    #     # make sure the border is on top of stone
    #     border *= mask
    #     # turn into 0 and 1
    #     border = np.array(border, dtype=bool).astype("uint8")
    #
    #     border_sum += border
    #
    # label = np.array(segments_sum, dtype=bool).astype("uint8") + \
    #         np.array(border_sum, dtype=bool).astype("uint8")
    # label = np.rint(label * 127.5).astype('uint8')

    label = segments_sum
    cv2.imwrite(f"debug_images/masks/{img[:-6]}", label)
