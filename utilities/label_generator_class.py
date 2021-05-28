"""
Created by Jan Schiffeler on 11.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2
from .image_utilities import crf_refinement, largest_region, fill_holes
from .unknown_class import UnknownRegister
from settings import *


class LabelGenerator:
    """Handle all label instances"""

    def __init__(self, border_thickness: int = BORDER_THICKNESS,
                 growth_rate: int = GROWTH_RATE, shrink_rate: int = SHRINK_RATE,
                 min_number: int = MIN_NUMBER, blur: int = BLUR, blur_thresh: int = BLUR_THRESH,
                 gsxy: int = GSXY, gcompat: int = GCOMPAT,
                 bsxy: int = BSXY, brgb: int = BRGB, bcompat: int = BCOMPAT,
                 dsxy: int = DSXY, dddd: int = DDDD, dcompat: int = DCOMPAT,
                 times: int = TIMES, height: int = HEIGHT, width: int = WIDTH,
                 un_small_thresh: int = UN_SMALL_THRESH, un_max_refinement_loss: float = UN_MAX_REFINEMENT_LOSS,
                 data_path: str = DATA_PATH, **kwargs):
        """
        Constructor for LabelGenerator
        :param border_thickness: thickness of border in final label
        :param growth_rate: number of dilation steps
        :param shrink_rate: number of erosion steps after dilation
        :param min_number: minimum number of instanced of one label
        :param blur: blur applied to regions (region dependent)
        :param blur_thresh: cutting off region here in binarization step
        :param gsxy: standard deviation smoothness pixel position
        :param gcompat: class compatibility gaussian
        :param bsxy: standard deviation colour ref pixel position
        :param brgb: standard deviation colour
        :param bcompat: class compatibility bilateral colour
        :param dsxy: standard deviation depth ref pixel position
        :param dddd: standard deviation depth
        :param dcompat: class compatibility gaussian bilateral depth
        :param times: repetitions of CRF
        :param height: image height
        :param width: image width
        :param un_small_thresh: unknown class threshold for which a mask is considered a small region
        :param un_max_refinement_loss: percentage size change in refinement to be considered a unknown region
        :param data_path: folder containing depth, images, masks, positions, pointclouds
        :param kwargs: largest: bool -> use only the largest connected region
                       fill: bool -> fill holes
        """
        # IMAGE AND LABEL
        self.height = height
        self.width = width
        self.label = np.zeros([height, width])
        self.masks = None

        # MASK GENERATION
        self.min_number = min_number
        self.growth_rate = growth_rate
        self.shrink_rate = shrink_rate
        self.blur = blur
        self.blur_thresh = blur_thresh
        # crf
        self.times = times
        #   Gaussian
        self.gsxy = gsxy
        self.gcompat = gcompat
        #   Bilateral Colour
        self.bsxy = bsxy
        self.brgb = brgb
        self.bcompat = bcompat
        #   Bilateral Depth
        self.dsxy = dsxy
        self.dddd = dddd
        self.dcompat = dcompat

        # REGISTER PARAMETER SETTINGS
        self.largest_only = False
        self.fill = False
        self.graph_mask_thresh = 125
        for argument, value in kwargs.items():
            if argument == "largest" and value:
                self.largest_only = True
            elif argument == "fill" and value:
                self.fill = True
            else:
                print("[ERROR] Unknown keyword argument: ", argument)

        # LABEL GENERATION
        self.kernel = np.array([[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]])
        self.border_thickness = border_thickness
        self.label_path = os.path.join(data_path, "labels")

        # INTO THE UNKNOWN
        self.unknown_reg = UnknownRegister(width=width, height=height,
                                           small_treshold=un_small_thresh, max_refinement_loss=un_max_refinement_loss)
        self.unknown_label = 50

    def main(self, projection: np.ndarray, original: np.ndarray, depth: np.ndarray, distance_map: np.ndarray, name: str):
        self._clear()
        all_masks = self._generate_masks(projection=projection, original=original, depth=depth, distance_map=distance_map)
        self._generate_label(all_masks=all_masks)
        self._apply_unknown()
        self._save(name=name)

    def _clear(self):
        self.label = np.zeros([self.height, self.width])
        self.masks = None
        self.unknown_reg.new_label()

    def _save(self, name: str):
        """
        :param name: name of the current image
        :return:
        """
        cv2.imwrite(self.label_path + "/" + name + ".png", self.label)

    def _generate_masks(self, projection: np.ndarray, original: np.ndarray, depth: np.ndarray, distance_map: np.ndarray):
        """
        Generate masks for each label instance (similar to blender mask output)
        :param projection: image with instance label for each pixel [0 = background]
        :param original: reprojected_cloud image
        :return: save mask images to separate folders in masks
        """

        self.unknown_reg.new_label()

        # FILTER TOO SMALL REGIONS
        labels_present, counts = np.unique(projection, return_counts=True)
        filter_small = counts < self.min_number
        labels_present = np.delete(labels_present, filter_small, 0)
        labels_present = np.delete(labels_present, 0, 0)
        labels_present = np.rint(labels_present).astype('uint8')

        if labels_present.shape[0] == 0:
            return None
        # OCCLUSION
        self.masks = np.zeros([labels_present.shape[0], projection.shape[0], projection.shape[1]], dtype='uint8')
        distances = []

        print("[INFO] Creating occlusion image")
        for i, label in enumerate(labels_present):
            instance = np.zeros_like(projection)
            instance[np.where(projection == label)] = 255
            instance = cv2.dilate(instance, np.ones((5, 5), 'uint8'), iterations=self.growth_rate)
            instance = cv2.erode(instance, np.ones((5, 5), 'uint8'), iterations=self.shrink_rate)

            instance = largest_region(instance) if self.largest_only else instance
            instance = fill_holes(instance) if self.fill else instance

            # instance_before = instance.copy()
            # core = instance.copy()

            blur = int(self.blur / 2500 * instance.nonzero()[0].shape[0])
            blur = blur + 1 if blur % 2 == 0 else blur
            instance = cv2.GaussianBlur(instance, (blur, blur), 0)

            # core = cv2.GaussianBlur(core, (blur, blur), 0)
            # core = cv2.threshold(core, 127, 255, cv2.THRESH_BINARY)[1]
            # instance[core == 255] = 255

            instance = cv2.threshold(instance, 127, 255, cv2.THRESH_BINARY)[1]

            self.masks[i, :, :] = instance.astype('uint8')
            distances.append(sum(distance_map[projection == label]) / distance_map[projection == label].shape[0])

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

        # debug option to filter projection image
        # for i, instance in enumerate(labels_present):
        #     filter_img = cv2.bitwise_not((projection == instance) * cv2.bitwise_not(self.masks[i, :, :])) // 255
        #     projection *= filter_img
        # debug purposes (projection != 0).astype('uint8') * 255

        # delete empty masks
        self.masks = np.delete(self.masks, empty_masks, 0)
        labels_present = np.delete(labels_present, empty_masks, 0)

        all_mask = np.zeros_like(self.masks[0], dtype='int64')
        for i, mask in enumerate(self.masks):
            self.unknown_reg.disconnected_patches(region=mask)
            all_mask += (mask // 255 * (i+1)).astype('int64')
        # import matplotlib.pyplot as plt
        # plt.imshow(all_mask)
        # plt.show()

        # REFINEMENT
        all_masks_refined = crf_refinement(img=original, mask=all_mask, depth=depth,
                                           times=self.times, n_classes=len(labels_present),
                                           gsxy=self.gsxy, gcompat=self.gcompat,
                                           bsxy=self.bsxy, brgb=self.brgb, bcompat=self.bcompat,
                                           dsxy=self.dsxy, dddd=self.dddd, dcompat=self.dcompat)

        # import matplotlib.pyplot as plt
        # plt.imshow(all_masks_refined)
        # plt.show()

        # self.unknown_reg.refinement_lost(before=instance_before, after=self.masks[i, :, :])
        # self.unknown_reg.small_region(region=self.masks[i, :, :])

        print()
        return all_masks_refined

    def _generate_label(self, all_masks):
        if all_masks is None:
            self.label = np.zeros((self.height, self.width), dtype='uint8')
        else:
            segments_sum = np.zeros((self.height, self.width), dtype='uint8')
            border_sum = np.zeros((self.height, self.width), dtype='uint8')

            for instance in np.unique(all_masks):
                if instance == 0:
                    continue
                mask_img = np.zeros_like(all_masks)
                mask_img[all_masks == instance] = 1
                mask_img = mask_img.astype('uint8')

                self.unknown_reg.disconnected_patches(region=mask_img)
                self.unknown_reg.holes(region=mask_img)

                mask_img = largest_region(mask_img)
                mask_img = fill_holes(mask_img)

                if mask_img.any():
                    segments_sum += mask_img

                    border = cv2.filter2D(mask_img, -1, self.kernel)
                    border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=self.border_thickness)
                    # make sure the border is on top of stone
                    border *= mask_img
                    # turn into 0 and 1
                    border = np.array(border, dtype=bool).astype("uint8")

                    border_sum += border

            self.label = np.array(segments_sum, dtype=bool).astype("uint8") + \
                         np.array(border_sum, dtype=bool).astype("uint8")
            self.label = np.rint(self.label * 127.5).astype('uint8')

    def _apply_unknown(self):
        unknown = self.unknown_reg.retrieve_label_img()
        self.label[unknown != 0] = self.unknown_label
