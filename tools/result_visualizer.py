"""
Created by Jan Schiffeler on 30.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2
from settings import DATA_PATH

name = 'real_1_8'
label_img = cv2.imread(f"{DATA_PATH}/labels/{name}.png")
debug_projection = cv2.imread(f"debug_images/visual_projection_{name}.png")
original_img = cv2.imread(f"{DATA_PATH}/images/{name}.png")

result_viz = cv2.addWeighted(label_img, 0.5, original_img, 0.5, 0.0)
# result_viz = cv2.addWeighted(original_img, 0.5, debug_projection, 0.5, 0.0)

cv2.imshow("debug", result_viz)
cv2.waitKey(0)
cv2.destroyWindow("debug")

cv2.imwrite(f"debug_images/result_viz_{name}.png", result_viz)
