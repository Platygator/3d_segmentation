"""
Created by Jan Schiffeler on 09.06.21
jan.schiffeler[at]gmail.com

Changed by


Python 3.8

"""
import cv2
import numpy as np
import json
from datetime import datetime
import open3d as o3d

import pydensecrf.densecrf as dcrf
from utilities import *
from settings import DATA_PATH


class Tuner(LabelGenerator):
    """Tool to tune parameters"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/clustered.ply")
        labels = np.load(f"{DATA_PATH}/pointclouds/labels.npy")

        image, position, depth_map, name = next(load_images())
        self.img_original = image
        self.depth_img = depth_map
        self.mask = np.zeros([self.height, self.width])
        self.solo_label = np.zeros([self.height, self.width])

        R = cloud.get_rotation_matrix_from_quaternion([position[0], position[1], position[2], position[3]])
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = R
        trans_mat[:3, 3] = position[-3:]

        self.projection, self.distance_map = reproject(points=cloud.points, color=cloud.colors, label=labels,
                                                       transformation_mat=trans_mat, depth_map=depth_map,
                                                       save_img=False, name="Tuning tool")

        self.labels_present = np.unique(labels)
        self.instance_to_show_id = 1
        self.instance_to_show = self.labels_present[self.instance_to_show_id]

        rand_col = np.random.random([2 * len(self.labels_present), 3])  # produces more colours as sometimes labels get
        rand_col[0, :] = [0, 0, 0]                                      # deleted and thus len != max_label_number
        self.visual_projection = (rand_col[self.projection.astype('uint8')] * 255).astype('uint8')
        self.visual_projection_save = self.visual_projection.copy()

        self.grey_bar = np.ones([self.height, 1, 3], dtype='uint8') * 128
        self.green_box = np.zeros([25, 500, 3])
        self.green_box[:, :, 1] = 255
        self.red_box = np.zeros([25, 500, 3])
        self.red_box[:, :, 2] = 255

    def setup_window(self):
        cv2.namedWindow("Label Generator")
        cv2.imshow("Label Generator", self.green_box)

        cv2.namedWindow("Results")

        # occlusion_check()
        self.update_viewport()

        cv2.createTrackbar('times', "Label Generator", 1, 100, trackbar_function)
        cv2.setTrackbarPos('times', 'Label Generator', self.times)
        cv2.createTrackbar('G_sxy', "Label Generator", 2, 100, trackbar_function)
        cv2.setTrackbarPos('G_sxy', 'Label Generator', self.gsxy)
        cv2.createTrackbar('G_compat', "Label Generator", 1, 100, trackbar_function)
        cv2.setTrackbarPos('G_compat', 'Label Generator', self.gcompat)
        cv2.createTrackbar('B_sxy', "Label Generator", 2, 100, trackbar_function)
        cv2.setTrackbarPos('B_sxy', 'Label Generator', self.bsxy)
        cv2.createTrackbar('B_srgb', "Label Generator", 2, 100, trackbar_function)
        cv2.setTrackbarPos('B_srgb', 'Label Generator', self.brgb)
        cv2.createTrackbar('B_compat', "Label Generator", 1, 100, trackbar_function)
        cv2.setTrackbarPos('B_compat', 'Label Generator', self.bcompat)
        cv2.createTrackbar('D_sxy', "Label Generator", 2, 100, trackbar_function)
        cv2.setTrackbarPos('D_sxy', 'Label Generator', self.dsxy)
        cv2.createTrackbar('D_sddd', "Label Generator", 2, 100, trackbar_function)
        cv2.setTrackbarPos('D_sddd', 'Label Generator', self.dddd)
        cv2.createTrackbar('D_compat', "Label Generator", 1, 100, trackbar_function)
        cv2.setTrackbarPos('D_compat', 'Label Generator', self.dcompat)

        cv2.createTrackbar('blur_thresh', "Label Generator", 0, 255, trackbar_function)
        cv2.setTrackbarPos('blur_thresh', 'Label Generator', self.blur_thresh)
        cv2.createTrackbar('min_number', "Label Generator", 0, 100, trackbar_function)
        cv2.setTrackbarPos('min_number', 'Label Generator', self.min_number)
        cv2.createTrackbar('growth_rate', "Label Generator", 0, 20, trackbar_function)
        cv2.setTrackbarPos('growth_rate', 'Label Generator', self.growth_rate)
        cv2.createTrackbar('shrink_rate', "Label Generator", 0, 20, trackbar_function)
        cv2.setTrackbarPos('shrink_rate', 'Label Generator', self.shrink_rate)
        cv2.createTrackbar('largest', "Label Generator", 0, 1, trackbar_function)
        cv2.setTrackbarPos('largest', 'Label Generator', int(self.largest_only))
        cv2.createTrackbar('fill', "Label Generator", 0, 1, trackbar_function)
        cv2.setTrackbarPos('fill', 'Label Generator', int(self.fill))
        cv2.createTrackbar('blur', "Label Generator", 50, 200, trackbar_function)
        cv2.setTrackbarPos('blur', 'Label Generator', self.blur)

        while True:
            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            elif k == ord("r"):
                print("Refreshing")
                self.progress_box(mode="busy")
                self.read_all_filter_positions()
                self.read_all_crf_positions()
                self.mask_for_one()
                self.update_crf()
                self.update_viewport()
                self.progress_box(mode="free")

            elif k == ord("f"):
                print("Refreshing Filter")
                self.progress_box(mode="busy")
                self.read_all_filter_positions()
                self.mask_for_one()
                self.update_viewport()
                self.progress_box(mode="free")

            elif k == ord("c"):
                print("Refreshing CRF")
                self.progress_box(mode="busy")
                self.read_all_crf_positions()
                self.update_crf()
                self.update_viewport()
                self.progress_box(mode="free")

            elif k == ord("p"):
                self.instance_to_show_id = self.instance_to_show_id + 1 \
                    if self.instance_to_show_id + 1 < len(self.labels_present) else self.instance_to_show_id
                self.instance_to_show = self.labels_present[self.instance_to_show_id]
                print("Instance: ", self.instance_to_show)
            elif k == ord("l"):
                self.instance_to_show_id = self.instance_to_show_id - 1 \
                    if self.instance_to_show_id - 1 > 0 else self.instance_to_show_id
                self.instance_to_show = self.labels_present[self.instance_to_show_id]
                print("Instance: ", self.instance_to_show)

            elif k == ord("a"):
                print("Creating full label")
                self.progress_box(mode="busy")
                self._clear()
                all_masks = self._generate_masks(projection=self.projection, original=self.img_original,
                                                 depth=self.depth_img, distance_map=self.distance_map)
                self._generate_label(all_masks=all_masks)
                rand_col = np.random.random(
                    [2 * len(self.labels_present), 3])  # produces more colours as sometimes labels get
                rand_col[0, :] = [0, 0, 0]  # deleted and thus len != max_label_number
                mask_show = (rand_col[self.label.astype('uint8')] * 255).astype('uint8')
                label_show = cv2.addWeighted(self.img_original, 0.3, mask_show, 0.7, 0.0)

                scale = 1.5
                resized = cv2.resize(label_show, (int(label_show.shape[1] * scale),
                                                  int(label_show.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                cv2.imshow("Full label", resized)
                cv2.waitKey(0)
                cv2.destroyWindow("Full label")
                self.progress_box(mode="free")

            elif k == ord("s"):
                print("Exporting Params")
                self.progress_box(mode="busy")
                self.read_all_filter_positions()
                self.read_all_crf_positions()
                settings_dict = {
                    "label_generation": {
                        "min_number": self.min_number,
                        "growth_rate": self.growth_rate,
                        "shrink_rate": self.shrink_rate,
                        "largest_only": self.largest_only == 1,
                        "fill": self.fill == 1,
                        "blur": self.blur,
                        "blur_thresh": self.blur_thresh
                    },
                    "crf": {
                        "times": self.times,
                        "gsxy": self.gsxy,
                        "gcompat": self.gcompat,
                        "bsxy": self.bsxy,
                        "brgb": self.bsxy,
                        "bcompat": self.bcompat,
                        "dsxy": self.dsxy,
                        "dddd": self.dddd,
                        "dcompat": self.dcompat
                    },
                    "border_thickness": 3,
                    "unknown_detector": {
                        "max_refinement_loss": 0.5,
                        "small_threshold": 500
                    }
                }
                now = datetime.now().strftime("%d_%m_%H_%M")
                with open(f'{now}.json', 'w') as file:
                    json.dump(settings_dict, file)
                self.progress_box(mode="free")
        cv2.destroyAllWindows()

    def read_all_crf_positions(self):
        self.times = cv2.getTrackbarPos("times", "Label Generator")
        self.gsxy = cv2.getTrackbarPos("G_sxy", "Label Generator")
        self.gcompat = cv2.getTrackbarPos("G_compat", "Label Generator")
        self.bsxy = cv2.getTrackbarPos("B_sxy", "Label Generator")
        self.brgb = cv2.getTrackbarPos("B_srgb", "Label Generator")
        self.bcompat = cv2.getTrackbarPos("B_compat", "Label Generator")
        self.dsxy = cv2.getTrackbarPos("D_sxy", "Label Generator")
        self.dddd = cv2.getTrackbarPos("D_sddd", "Label Generator")
        self.dcompat = cv2.getTrackbarPos("D_compat", "Label Generator")

    def read_all_filter_positions(self):
        self.blur_thresh = cv2.getTrackbarPos('blur_thresh', "Label Generator")
        self.min_number = cv2.getTrackbarPos('min_number', "Label Generator")
        self.growth_rate = cv2.getTrackbarPos('growth_rate', "Label Generator")
        self.shrink_rate = cv2.getTrackbarPos('shrink_rate', "Label Generator")
        self.largest_only = cv2.getTrackbarPos('largest', "Label Generator")
        self.fill = cv2.getTrackbarPos('fill', "Label Generator")
        self.blur = cv2.getTrackbarPos('blur', "Label Generator")

        if self.blur % 2 == 0:
            self.blur += 1

    def crf_refinement_single(self) -> np.ndarray:
        """
        Based on this dudes code: https://github.com/seth814/Semantic-Shapes/blob/master/CRF%20Cat%20Demo.ipynb
        :return: refined label image
        """

        not_mask = cv2.bitwise_not(self.mask)
        not_mask = np.expand_dims(not_mask, axis=2)
        mask = np.expand_dims(self.mask, axis=2)
        im_softmax = np.concatenate([not_mask, mask], axis=2)
        im_softmax = im_softmax / 255.0

        feat_first = im_softmax.transpose((2, 0, 1)).reshape((2, -1))
        unary = unary_from_softmax(feat_first)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(self.width, self.height, 2)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=self.gsxy, compat=self.gcompat, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        d.addPairwiseBilateral(sxy=self.bsxy, srgb=self.brgb, rgbim=self.img_original,
                               compat=self.bcompat,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        self.depth_img * 255 / self.depth_img.max()
        depth_img = self.depth_img.astype('uint8')
        depth_img = depth_img[:, :, np.newaxis].repeat(3, axis=2)
        d.addPairwiseBilateral(sxy=self.dsxy, srgb=self.dddd, rgbim=depth_img,
                               compat=self.dcompat,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(self.times)
        res = np.argmax(Q, axis=0).reshape((self.height, self.width))
        res *= 255
        res = res.astype('uint8')
        return res

    def update_crf(self):
        self.read_all_crf_positions()
        self.solo_label = self.crf_refinement_single()
        if self.solo_label.any():
            self.solo_label = largest_region(self.solo_label)
            self.solo_label = fill_holes(self.solo_label)

    def update_viewport(self):
        mask_show = self.mask
        mask_show = np.repeat(mask_show[:, :, np.newaxis], 3, axis=2).astype('uint8')

        # filtered_show = np.repeat(self.visual_projection, 3, axis=2).astype('uint8')
        filtered_show = cv2.addWeighted(self.visual_projection, 0.8, mask_show, 0.3, 0.0)
        reprojection = cv2.addWeighted(self.visual_projection, 0.7, self.img_original, 0.3, 0.0)

        # label_show = np.zeros([self.height, self.width], dtype='uint8')
        # label_show[self.label == self.instance_to_show_id] = 255
        label_show = np.repeat(self.solo_label[:, :, np.newaxis], 3, axis=2).astype('uint8')
        label_show[:, :, 1] = 0
        label_show = cv2.addWeighted(label_show, 0.3, self.img_original, 0.7, 0.0)

        # img = np.concatenate([reprojection,
        #                       self.grey_bar,
        #                       filtered_show,
        img = np.concatenate([filtered_show,
                              self.grey_bar,
                              label_show], axis=1)
        scale = 1.8
        resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Results", resized)

    def mask_for_one(self):
        self.unknown_reg.new_label()

        # FILTER TOO SMALL REGIONS
        labels_present, counts = np.unique(self.projection, return_counts=True)
        filter_small = counts < self.min_number
        labels_present = np.delete(labels_present, filter_small, 0)
        labels_present = np.delete(labels_present, 0, 0)
        labels_present = np.rint(labels_present).astype('uint8')

        if labels_present.shape[0] == 0:
            return None
        # OCCLUSION
        self.masks = np.zeros([labels_present.shape[0], self.height, self.width], dtype='uint8')
        distances = []

        print("[INFO]       Creating masks")
        for i, label in enumerate(labels_present):
            instance = np.zeros_like(self.projection)
            instance[np.where(self.projection == label)] = 255

            instance = cv2.GaussianBlur(instance, (self.blur, self.blur), 0)
            instance = cv2.threshold(instance, self.blur_thresh, 255, cv2.THRESH_BINARY)[1]

            instance = cv2.dilate(instance, np.ones((3, 3), 'uint8'), iterations=self.growth_rate)
            instance = cv2.erode(instance, np.ones((3, 3), 'uint8'), iterations=self.shrink_rate)

            instance = cv2.GaussianBlur(instance, (self.blur, self.blur), 0)
            instance = cv2.threshold(instance, self.blur_thresh, 255, cv2.THRESH_BINARY)[1]

            instance = largest_region(instance) if self.largest_only else instance
            instance = fill_holes(instance) if self.fill else instance

            self.masks[i, :, :] = instance.astype('uint8')
            distances.append(sum(self.distance_map[self.projection == label]) /
                             self.distance_map[self.projection == label].shape[0])

        # Check for occlusion by hierarchy
        distances = np.array(distances)
        sort_order = distances.argsort()
        labels_present = labels_present[sort_order]
        self.masks = self.masks[sort_order, :, :]

        empty_masks = []
        for i in range(labels_present.shape[0]):
            master = self.masks[i, :, :].copy()
            if master.any():
                for j in range(i + 1, labels_present.shape[0]):
                    self.masks[j, :, :] = cv2.bitwise_or(self.masks[j, :, :],  self.masks[j, :, :],
                                                         mask=cv2.bitwise_not(master))
            else:
                empty_masks.append(i)

        self.visual_projection = self.visual_projection_save.copy()
        for i, instance in enumerate(labels_present):
            filter_img = cv2.bitwise_not((self.projection == instance) * cv2.bitwise_not(self.masks[i, :, :])) // 255
            self.visual_projection *= filter_img[:, :, np.newaxis].repeat(3, axis=2)

        self.mask = self.masks[self.instance_to_show, :, :]

    def progress_box(self, mode):
        if mode == "busy":
            cv2.imshow("Label Generator", self.red_box)
            cv2.waitKey(1)
        elif mode == "free":
            cv2.imshow("Label Generator", self.green_box)
            cv2.waitKey(1)


def trackbar_function(value):
    pass


if __name__ == '__main__':
    param_tool = Tuner()
    param_tool.setup_window()


