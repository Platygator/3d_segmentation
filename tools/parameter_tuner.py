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

from scipy import ndimage

import pydensecrf.densecrf as dcrf
from utilities import *


def crf_refinement(img: np.ndarray, mask: np.ndarray, times: int, n_classes: int, gsxy, gcompat, bsxy, brgb, bcompat,
                   depth_img,  dsxy, drgb, dcompat) -> np.ndarray:
    """
    Based on this dudes code: https://github.com/seth814/Semantic-Shapes/blob/master/CRF%20Cat%20Demo.ipynb
    :param img: reprojected_cloud image
    :param mask: label mask
    :param t: repetitions
    :param n_classes: number of classes (actually always 2 here)
    :return: refined label image
    """

    not_mask = cv2.bitwise_not(mask)
    not_mask = np.expand_dims(not_mask, axis=2)
    mask = np.expand_dims(mask, axis=2)
    im_softmax = np.concatenate([not_mask, mask], axis=2)
    im_softmax = im_softmax / 255.0

    feat_first = im_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=gsxy, compat=gcompat, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=bsxy, srgb=brgb, rgbim=img,
                           compat=bcompat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    depth_img * 255/depth_img.max()
    depth_img = depth_img.astype('uint8')
    depth_img = depth_img[:, :, np.newaxis].repeat(3, axis=2)
    # d.addPairwiseBilateral(sxy=dsxy, srgb=drgb, rgbim=depth_img,
    #                        compat=dcompat,
    #                        kernel=dcrf.DIAG_KERNEL,
    #                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(times)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    res *= 255
    res = res.astype('uint8')
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


def read_all_crf_positions():
    t = cv2.getTrackbarPos("t", "Label Generator")
    G_sxy = cv2.getTrackbarPos("G_sxy", "Label Generator")
    G_compat = cv2.getTrackbarPos("G_compat", "Label Generator")
    B_sxy = cv2.getTrackbarPos("B_sxy", "Label Generator")
    B_srgb = cv2.getTrackbarPos("B_srgb", "Label Generator")
    B_compat = cv2.getTrackbarPos("B_compat", "Label Generator")
    D_sxy = cv2.getTrackbarPos("D_sxy", "Label Generator")
    D_srgb = cv2.getTrackbarPos("D_srgb", "Label Generator")
    D_compat = cv2.getTrackbarPos("D_compat", "Label Generator")

    return t, G_sxy, G_compat, B_sxy, B_srgb, B_compat, D_sxy, D_srgb, D_compat


def read_all_filter_positions():
    threshold = cv2.getTrackbarPos('threshold', "Label Generator")
    min_pixel = cv2.getTrackbarPos('min_pixel', "Label Generator")
    dilate = cv2.getTrackbarPos('dilate', "Label Generator")
    erode = cv2.getTrackbarPos('erode', "Label Generator")
    largest = cv2.getTrackbarPos('largest', "Label Generator")
    fill = cv2.getTrackbarPos('fill', "Label Generator")
    blur = cv2.getTrackbarPos('blur', "Label Generator")

    if blur % 2 == 0:
        blur += 1

    return threshold, min_pixel, dilate, erode, largest, fill, blur


def IoU(label: np.ndarray, ground_truth: np.ndarray) -> (np.ndarray, float):
    """
    Calculate Intersection of Union
    :param label: generated label
    :param ground_truth: ground truth to compare to
    :return: IoU per instance and mean IoU
    """
    if ground_truth.max() == 2:
        gt_vals = [0, 1, 2]
    else:
        gt_vals = [0, 128, 255]
    iou_per_instance = np.zeros(3)
    for i, (instance_lab, instance_gt) in enumerate(zip([0, 128, 255], gt_vals)):
        org_instance = np.zeros_like(ground_truth)
        org_instance[np.where(ground_truth == instance_gt)] = 1
        rec_instance = np.zeros_like(label)
        rec_instance[np.where(label == instance_lab)] = 1

        intersection = np.logical_and(org_instance, rec_instance).astype('uint8')
        union = np.logical_or(org_instance, rec_instance).astype('uint8')
        iou_per_instance[i] = np.sum(intersection) / np.sum(union)

    return iou_per_instance, np.mean(iou_per_instance)


def create_full_label():
    global labels
    global instance_to_show
    global ground_truth
    global mask

    instance_store = instance_to_show

    kernel = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])

    n_instances = labels.max()

    segments_sum = np.zeros_like(labels)
    border_sum = np.zeros_like(labels)
    for i in range(1, n_instances + 1):
        instance_to_show = i
        update_filter()
        update_crf()

        mask_inst = np.rint(mask / 255).astype('uint8')
        segments_sum += mask_inst

        border = cv2.filter2D(mask_inst, -1, kernel)
        border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=3)
        # # make sure the border is on top of stone
        border *= mask_inst
        # turn into 0 and 1
        border = np.array(border, dtype=bool).astype("uint8")

        border_sum += border

    full_label = np.array(segments_sum, dtype=bool).astype("uint8") + np.array(border_sum, dtype=bool).astype("uint8")
    full_label = np.rint(full_label * 127.5).astype("uint8")

    iou = IoU(full_label, ground_truth)
    text = np.zeros([full_label.shape[0], full_label.shape[1]//2])
    cv2.putText(text, f"Mean: {iou[1]:.5f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Background: {iou[0][0]:.5f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Stone: {iou[0][1]:.5f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Border: {iou[0][2]:.5f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    label_show = np.concatenate([text.astype('uint8'), full_label, ground_truth], axis=1)

    scale = 1.5
    resized = cv2.resize(label_show, (int(label_show.shape[1] * scale),
                                      int(label_show.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Full label", resized)
    cv2.waitKey(0)
    cv2.destroyWindow("Full label")

    instance_to_show = instance_store


def create_full_label_all():
    global labels
    global instance_to_show
    global ground_truth
    global mask

    instance_store = instance_to_show

    kernel = np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])

    n_instances = labels.max()

    segments_sum = np.zeros_like(labels)
    border_sum = np.zeros_like(labels)
    filters = np.zeros([filtered.shape[0], filtered.shape[1]], dtype='int64')
    for i in range(1, n_instances + 1):
        instance_to_show = i
        update_filter()
        threshold, min_pixel, dilate, erode, largest, fill, blur = read_all_filter_positions()
        # filters[:, :, i-1] = (cv2.threshold(filtered, 128, 255, cv2.THRESH_BINARY)[1] // 255).astype('int64')
        thresh = (cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)[1] // 255).astype('int64')
        thresh = thresh * (cv2.bitwise_not(filters.astype('bool').astype('uint8')) // 255).astype('int64')
        filters += thresh * i

    t, G_sxy, G_compat, B_sxy, B_srgb, B_compat, D_sxy, D_srgb, D_compat = read_all_crf_positions()
    mask = crf_refinement_all(img=img_original, mask=filters, times=t, n_classes=n_instances,
                              gsxy=G_sxy, gcompat=G_compat,
                              bsxy=B_sxy, brgb=B_srgb, bcompat=B_compat,
                              dsxy=D_sxy, drgb=D_srgb, dcompat=D_compat, depth_img=depth_map)

    for instance in np.unique(mask):
        if instance == 0:
            continue
        mask_inst = np.zeros_like(mask)
        mask_inst[mask == instance] = 1
        mask_inst = mask_inst.astype('uint8')
        if mask_inst.any():
            mask_inst = largest_region(mask_inst)
            mask_inst = fill_holes(mask_inst)
        segments_sum += mask_inst

        border = cv2.filter2D(mask_inst, -1, kernel)
        border = cv2.dilate(border, np.ones((5, 5), 'uint8'), iterations=3)
        # # make sure the border is on top of stone
        border *= mask_inst
        # turn into 0 and 1
        border = np.array(border, dtype=bool).astype("uint8")

        border_sum += border

    full_label = np.array(segments_sum, dtype=bool).astype("uint8") + np.array(border_sum, dtype=bool).astype("uint8")
    full_label = np.rint(full_label * 127.5).astype("uint8")

    iou = IoU(full_label, ground_truth)
    text = np.zeros([full_label.shape[0], full_label.shape[1]//2])
    cv2.putText(text, f"Mean: {iou[1]:.5f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Background: {iou[0][0]:.5f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Stone: {iou[0][1]:.5f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    cv2.putText(text, f"Border: {iou[0][2]:.5f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 3)
    label_show = np.concatenate([text.astype('uint8'), full_label, ground_truth], axis=1)

    scale = 1.5
    resized = cv2.resize(label_show, (int(label_show.shape[1] * scale),
                                      int(label_show.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Full label", resized)
    cv2.waitKey(0)
    cv2.destroyWindow("Full label")

    instance_to_show = instance_store


def update_crf():
    global img_original
    global filtered
    global mask
    global depth_map
    t, G_sxy, G_compat, B_sxy, B_srgb, B_compat, D_sxy, D_srgb, D_compat = read_all_crf_positions()
    # depth_refinement = np.rint(depth_map / depth_map.max() * 255).astype('uint8')[:, :, np.newaxis].repeat(3, axis=2)

    # if mask.any():
    #     mask = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)[1]
    #     mask = graph_cut_refinement(img=img_original, mask=mask, iter_count=5)
    mask = crf_refinement(img_original, filtered, t, 2, G_sxy, G_compat, B_sxy, B_srgb, B_compat, depth_map,  D_sxy, D_srgb, D_compat)
    if mask.any():
        mask = largest_region(mask)
        mask = fill_holes(mask)
    # blur = int(1 / 10000 * filtered.nonzero()[0].shape[0])
    # blur = blur + 1 if blur % 2 == 0 else blur
    # mask = cv2.GaussianBlur(filtered, (blur, blur), 0)
    # mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]


def occlusion_check():
    global img_reprojection_viz
    global labels
    global distance_map
    all_labels = np.unique(labels)
    all_labels = np.delete(all_labels, 0)

    masks = np.zeros([labels.shape[0], labels.shape[1], all_labels.shape[0]], dtype='uint8')
    distances = []
    for i, label in enumerate(all_labels):
        label_mask = np.zeros_like(labels)
        label_mask[labels == label] = 255
        label_mask = cv2.dilate(label_mask, np.ones((5, 5), 'uint8'), iterations=10)
        label_mask = cv2.erode(label_mask, np.ones((5, 5), 'uint8'), iterations=10)
        masks[:, :, i] = label_mask  # [:, :, np.newaxis]
        distances.append(sum(distance_map[labels == label]) / distance_map[labels == label].shape[0])

    distances = np.array(distances)
    sort_order = distances.argsort()
    # inverse =
    all_labels = all_labels[sort_order]
    masks = masks[:, :, sort_order]

    for i in range(all_labels.shape[0]):
        master = masks[:, :, i].copy()
        # cv2.imwrite(f"mask{i}.png", master)
        for j in range(i + 1, all_labels.shape[0]):
            masks[:, :, j] = cv2.bitwise_or(masks[:, :, j], masks[:, :, j], mask=cv2.bitwise_not(master))

    for i, instance in enumerate(all_labels):
        filter_img = cv2.bitwise_not((labels == instance) * cv2.bitwise_not(masks[:, :, i])) // 255
        img_reprojection_viz *= filter_img[:, :, np.newaxis].repeat(3, axis=2)
        labels *= filter_img


def update_filter():
    global filtered
    global depth_map
    global distance_map
    global img_reprojection_viz_original
    global img_reprojection_viz
    global labels_original
    global labels
    threshold, min_pixel, dilate, erode, largest, fill, blur = read_all_filter_positions()

    # dist_filter = np.abs(depth_map - distance_map) * distance_map.astype('bool') < distance_map * occlusion_range
    # img_reprojection_viz = img_reprojection_viz_original * np.repeat(dist_filter[:, :, np.newaxis], 3, axis=2)
    #
    # labels = labels_original * dist_filter

    filtered = (labels == instance_to_show) * 255
    filtered = filtered.astype('uint8')
    filtered = cv2.GaussianBlur(filtered, (blur, blur), 0)
    filtered = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)[1]

    filtered = cv2.dilate(filtered, np.ones((3, 3), 'uint8'), iterations=dilate)
    filtered = cv2.erode(filtered, np.ones((3, 3), 'uint8'), iterations=erode)

        # core = filtered.copy()

    # blur = int(blur / 2500 * filtered.nonzero()[0].shape[0])
    # blur = blur + 1 if blur % 2 == 0 else blur

    filtered = cv2.GaussianBlur(filtered, (blur, blur), 0)


        # core = cv2.erode(core, np.ones((5, 5), 'uint8'), iterations=erode)
        # core = cv2.GaussianBlur(core, (blur, blur), 0)
        # core = cv2.threshold(core, threshold, 255, cv2.THRESH_BINARY)[1]
    filtered = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)[1]

        # filtered[core == 255] = 255

    if filtered.any():
        filtered = largest_region(filtered) if largest else filtered
        filtered = fill_holes(filtered) if fill else filtered


def update_viewport():
    global img_original
    global img_reprojection_viz
    global filtered
    global mask

    filtered_show = np.repeat(filtered[:, :, np.newaxis], 3, axis=2)
    filtered_show = cv2.addWeighted(filtered_show, 0.8, img_reprojection_viz, 0.2, 0.0)

    mask_show = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype('uint8')
    mask_show = cv2.addWeighted(mask_show, 0.3, img_original, 0.7, 0.0)
    # mask_show = cv2.addWeighted(img_reprojection_viz, 0.9, img_original, 0.1, 0.0)
    # cv2.imwrite("reprojection.png", mask_show)

    img = np.concatenate([img_reprojection_viz,
                          grey_bar,
                          filtered_show,
                          grey_bar,
                          mask_show], axis=1)
    scale = 1.5
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Results", resized)


def trackbar_function(value):
    pass


if __name__ == '__main__':
    img_name = "pile_2_28"
    img_original = cv2.imread(f"{img_name}.png")
    # ground_truth = cv2.imread("ground_truth_03_01_01850.png", 0)
    ground_truth = cv2.imread(f"{img_name}.png", 0)
    img_reprojection_viz_original = cv2.imread(f"visual_projection_{img_name}.png")
    img_reprojection_viz = img_reprojection_viz_original.copy()
    labels_original = cv2.imread(f"label_projection_{img_name}.png", 0)
    labels = labels_original.copy()
    # depth_map = np.load(f"{img_name}_depth.npy", allow_pickle="True")
    depth_map = np.load(f"39_depth.npy", allow_pickle="True")
    distance_map = np.load(f"{img_name}_distance.npy", allow_pickle="True")
    filtered = np.zeros_like(labels_original)
    mask = np.zeros_like(labels_original)

    grey_bar = np.ones([img_original.shape[0], 1, 3], dtype='uint8') * 128

    instance_to_show = 1

    cv2.namedWindow("Label Generator")
    cv2.imshow("Label Generator", np.zeros([1, 500]))
    cv2.namedWindow("Results")

    occlusion_check()
    update_viewport()

    cv2.createTrackbar('t', "Label Generator", 1, 100, trackbar_function)
    cv2.setTrackbarPos('t', 'Label Generator', 6)
    cv2.createTrackbar('G_sxy', "Label Generator", 2, 100, trackbar_function)
    cv2.setTrackbarPos('G_sxy', 'Label Generator', 95)
    cv2.createTrackbar('G_compat', "Label Generator", 1, 100, trackbar_function)
    cv2.setTrackbarPos('G_compat', 'Label Generator', 1)
    cv2.createTrackbar('B_sxy', "Label Generator", 2, 100, trackbar_function)
    cv2.setTrackbarPos('B_sxy', 'Label Generator', 58)
    cv2.createTrackbar('B_srgb', "Label Generator", 2, 100, trackbar_function)
    cv2.setTrackbarPos('B_srgb', 'Label Generator', 3)
    cv2.createTrackbar('B_compat', "Label Generator", 1, 100, trackbar_function)
    cv2.setTrackbarPos('B_compat', 'Label Generator', 75)
    cv2.createTrackbar('D_sxy', "Label Generator", 2, 100, trackbar_function)
    cv2.setTrackbarPos('D_sxy', 'Label Generator', 74)
    cv2.createTrackbar('D_srgb', "Label Generator", 2, 100, trackbar_function)
    cv2.setTrackbarPos('D_srgb', 'Label Generator', 20)
    cv2.createTrackbar('D_compat', "Label Generator", 1, 100, trackbar_function)
    cv2.setTrackbarPos('D_compat', 'Label Generator', 22)


    cv2.createTrackbar('threshold', "Label Generator", 0, 255, trackbar_function)
    cv2.setTrackbarPos('threshold', 'Label Generator', 0)
    cv2.createTrackbar('min_pixel', "Label Generator", 0, 100, trackbar_function)
    cv2.setTrackbarPos('min_pixel', 'Label Generator', 5)
    cv2.createTrackbar('dilate', "Label Generator", 0, 20, trackbar_function)
    cv2.setTrackbarPos('dilate', 'Label Generator', 5)
    cv2.createTrackbar('erode', "Label Generator", 0, 20, trackbar_function)
    cv2.setTrackbarPos('erode', 'Label Generator', 5)
    cv2.createTrackbar('largest', "Label Generator", 0, 1, trackbar_function)
    cv2.setTrackbarPos('largest', 'Label Generator', 1)
    cv2.createTrackbar('fill', "Label Generator", 0, 1, trackbar_function)
    cv2.setTrackbarPos('fill', 'Label Generator', 1)
    cv2.createTrackbar('blur', "Label Generator", 50, 200, trackbar_function)
    cv2.setTrackbarPos('blur', 'Label Generator', 15)

    while True:
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        elif k == ord("r"):
            print("Refreshing")
            update_filter()
            update_crf()
            update_viewport()
        elif k == ord("f"):
            print("Refreshing Filter")
            update_filter()
            update_viewport()
        elif k == ord("c"):
            print("Refreshing CRF")
            update_crf()
            update_viewport()
        elif k == ord("p"):
            instance_to_show += 1
            print("Instance: ", instance_to_show)
        elif k == ord("l"):
            instance_to_show = instance_to_show - 1 if instance_to_show - 1 > 0 else instance_to_show
            print("Instance: ", instance_to_show)
        elif k == ord("a"):
            print("Creating full label")
            create_full_label_all()
            print(read_all_crf_positions())
            print()
            print(read_all_filter_positions())
        elif k == ord("s"):
            print("Exporting Params")
            threshold, min_pixel, dilate, erode, largest, fill, blur = read_all_filter_positions()
            t, G_sxy, G_compat, B_sxy, B_srgb, B_compat, D_sxy, D_srgb, D_compat = read_all_crf_positions()
            settings_dict = {
                "label_generation": {
                    "min_number": min_pixel,
                    "growth_rate": dilate,
                    "shrink_rate": erode,
                    "largest_only": largest == 1,
                    "fill": fill == 1,
                    "blur": blur,
                    "blur_thresh": threshold
                },
                "crf": {
                    "times": t,
                    "gsxy": G_sxy,
                    "gcompat": G_compat,
                    "bsxy": B_sxy,
                    "brgb": B_srgb,
                    "bcompat": B_compat,
                    "dsxy": D_sxy,
                    "dddd": D_srgb,
                    "dcompat": D_compat
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
    cv2.destroyAllWindows()
