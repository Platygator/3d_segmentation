"""
Created by Jan Schiffeler on 11.05.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import cv2
from .image_utilities import crf_refinement, graph_cut_refinement, largest_region, fill_holes
from .unknown_class import UnknownRegister
from settings import *


class LabelGenerator:
    """Handle all label instances"""

    def __init__(self, border_thickness: int = border_thickness,  growth_rate: int = growth_rate, shrink_rate: int = shrink_rate,
                 min_number: int = min_number, refinement_method: str = refinement_method, blur: int = blur,
                 gsxy: int = gsxy, gcompat: int = gcompat, bsxy: int = bsxy, brgb: int = brgb, bcompat: int = bcompat,
                 times: int = times, iter_count: int = iter_count, height: int = HEIGHT, width: int = WIDTH,
                 data_path: str = DATA_PATH, **kwargs):
        """Constructor for LabelGenerator
        :param growth_rate: number of dilation steps
        :param shrink_rate: number of erosion steps after dilation
        :param min_number: minimum number of instanced of one label
        :param refinement_method: Chose "crf" or "graph"
        :param data_path: folder containing depth, images, masks, positions, pointclouds
        :param kwargs: largest: bool -> use only the largest connected region
                       fill: bool -> fill holes
                       graph_thresh: int -> threshold for gaussian blur for graph cut mask

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
        self.refinement_method = refinement_method
        # graph cut
        self.iter_count = iter_count
        # crf
        self.times = times
        self.gsxy = gsxy
        self.gcompat = gcompat
        self.bsxy = bsxy
        self.brgb = brgb
        self.bcompat = bcompat

        # REGISTER PARAMETER SETTINGS
        self.largest_only = False
        self.fill = False
        self.graph_mask_thresh = 125
        for argument, value in kwargs.items():
            if argument == "largest" and value:
                self.largest_only = True
            elif argument == "fill" and value:
                self.fill = True
            elif argument == "graph_thresh":
                self.graph_mask_thresh = value
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
                                           small_treshold=un_small_tresh, max_refinement_loss=un_max_refinement_loss)
        self.unknown_label = 50

    def main(self, projection: np.ndarray, original: np.ndarray, distance_map: np.ndarray, name: str):
        self._clear()
        self._generate_masks(projection=projection, original=original, distance_map=distance_map)
        self._generate_label()
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

    def _generate_masks(self, projection: np.ndarray, original: np.ndarray, distance_map: np.ndarray):
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
        labels_present = np.delete(labels_present, 0)
        labels_present = np.rint(labels_present).astype('uint8')

        # OCCLUSION
        self.masks = np.zeros([labels_present.shape[0], projection.shape[0], projection.shape[1]], dtype='uint8')
        distances = []

        print("[INFO] Creating occlusion image")
        for i, label in enumerate(labels_present):
            instance = np.zeros_like(projection)
            instance[np.where(projection == label)] = 255
            instance = cv2.dilate(instance, np.ones((5, 5), 'uint8'), iterations=growth_rate)
            instance = cv2.erode(instance, np.ones((5, 5), 'uint8'), iterations=shrink_rate)
            # TODO think about using concave hull with dilation

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
        #     filter_img = cv2.bitwise_not((projection == instance) * cv2.bitwise_not(masks[i, :, :])) // 255
        #     projection *= filter_img
        # debug purposes (projection != 0).astype('uint8') * 255

        # delete empty masks
        self.masks = np.delete(self.masks, empty_masks, 0)
        labels_present = np.delete(labels_present, empty_masks, 0)

        # REFINEMENT
        print("[INFO] Processing label number: ", end='')
        for i, label_num in enumerate(labels_present):
            instance = self.masks[i, :, :]
            print(label_num, ", ", end='')
            self.unknown_reg.disconnected_patches(region=self.masks[i, :, :])
            instance = largest_region(instance) if largest_only else instance
            instance = fill_holes(instance) if fill else instance

            instance_before = instance.copy()
            blur = int(self.blur / 2500 * instance.nonzero()[0].shape[0])
            blur = blur + 1 if blur % 2 == 0 else blur
            instance = cv2.GaussianBlur(instance, (blur, blur), 0)

            if refinement_method == "crf":
                self.masks[i, :, :] = crf_refinement(img=original, mask=instance, times=self.times,
                                            gsxy=self.gsxy, gcompat=self.gcompat,
                                            bsxy=self.bsxy, brgb=self.brgb, bcompat=self.bcompat)
            elif refinement_method == "graph":
                instance = cv2.threshold(instance, graph_mask_thresh, 255, cv2.THRESH_BINARY)[1]
                self.masks[i, :, :] = graph_cut_refinement(img=original, mask=instance.copy(), iter_count=self.iter_count)
            else:
                print("[ERROR] No correct refinement method chosen!")
                instance = cv2.threshold(instance, graph_mask_thresh, 255, cv2.THRESH_BINARY)[1]
                self.masks[i, :, :] = instance

            self.unknown_reg.refinement_lost(before=instance_before, after=self.masks[i, :, :])
            self.unknown_reg.disconnected_patches(region=self.masks[i, :, :])
            self.unknown_reg.small_region(region=self.masks[i, :, :])

            self.masks[i, :, :] = largest_region(self.masks[i, :, :])  # if largest_only else self.masks[i, :, :]
            self.masks[i, :, :] = fill_holes(self.masks[i, :, :])  # if fill else self.masks[i, :, :]

        print()

    def _generate_label(self):
        segments_sum = np.zeros((self.height, self.width), dtype='uint8')
        border_sum = np.zeros((self.height, self.width), dtype='uint8')

        for mask_img in self.masks:
            if mask_img.any():
                mask_img = np.rint(mask_img / 255).astype('uint8')
                segments_sum += mask_img

                border = cv2.filter2D(mask_img, -1, self.kernel)
                border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=self.border_thickness)
                # make sure the border is on top of stone
                border *= fill_holes(mask_img * 255) // 255
                # turn into 0 and 1
                border = np.array(border, dtype=bool).astype("uint8")

                border_sum += border

        self.label = np.array(segments_sum, dtype=bool).astype("uint8") + \
                     np.array(border_sum, dtype=bool).astype("uint8")
        self.label = np.rint(self.label * 127.5).astype('uint8')

    def _apply_unknown(self):
        unknown = self.unknown_reg.retrieve_label_img()
        self.label[unknown != 0] = self.unknown_label
