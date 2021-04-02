"""
Created by Jan Schiffeler on 30.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2

name = '828'
label_img = cv2.imread(f"debug_images/visual_projection_{name}.png", 1)
original_img = cv2.imread(f"data/reconstruction_2/images/{name}.png", 1)

# label_img = np.rot90(label_img)
# label_img = np.fliplr(label_img)
# label_img = np.flipud(label_img)
result_viz = cv2.addWeighted(label_img, 0.9, original_img, 0.1, 0.0)

cv2.imwrite(f"debug_images/result_viz_{name}.png", result_viz)

