"""
Created by Jan Schiffeler on 30.03.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2
from settings import DATA_PATH, MODE
import glob
import os

name = 'pile1_8'
debug_projection = cv2.imread(f"debug_images/visual_projection_{name}.png")

if MODE == "semantic":
    img_names = [os.path.basename(k) for k in glob.glob(f"{DATA_PATH}/labels/*.png")]
else:
    img_names = [os.path.basename(k) for k in glob.glob(f"{DATA_PATH}/masks/*.png")]

len_img = len(img_names)

for i, name in enumerate(img_names):
    print(f"Transforming image {i+1} / {len_img} -> {name}")
    original_img = cv2.imread(f"{DATA_PATH}/images/{name}")
    if name[:2] == "v_":
        continue

    if MODE == "semantic":
        label_img = cv2.imread(f"{DATA_PATH}/labels/{name}", 0)
        col_label = np.zeros([label_img.shape[0], label_img.shape[1], 3], dtype='uint8')
        col_label[label_img == 128] = [65, 189, 245]
        # label[label == 127] = 128
        col_label[label_img == 255] = [60, 90, 255]
        # label[label == 254] = 255
        col_label[label_img == 50] = [0, 0, 0]
        result_viz = cv2.addWeighted(original_img, 0.5, col_label, 0.5, 0.0)
        cv2.imwrite(f"{DATA_PATH}/labels/v_{name}", result_viz)
    else:
        label_img = cv2.imread(f"{DATA_PATH}/masks/{name}", 0)
        labels_present = np.unique(label_img)
        rand_col = np.random.random(
            [2 * len(labels_present), 3])  # produces more colours as sometimes labels get
        rand_col[0, :] = [0, 0, 0]  # deleted and thus len != max_label_number
        mask_show = (rand_col[label_img.astype('uint8')] * 255).astype('uint8')
        result_viz = cv2.addWeighted(original_img, 0.3, mask_show, 0.7, 0.0)
        cv2.imwrite(f"{DATA_PATH}/masks/v_{name}", result_viz)
# color_img = plt.imshow(label_img[:, :, 0] * 10, cmap='Pastel1')
# plt.show()

# result_viz = cv2.addWeighted(debug_projection, 0.5, original_img, 0.5, 0.0)


# cv2.imshow("debug", result_viz)
# cv2.waitKey(0)
# cv2.destroyWindow("debug")

