"""
Created by Jan Schiffeler on 25.03.12021
jan.schiffeler[at]gmail.com

Changed by

Different clustering and label generation functions

Python 3.8
Library version:


"""

import open3d as o3d
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from .general_functions import turn_ply_to_npy
from settings import DATA_PATH, CAM_MAT, DIST_MAT, WIDTH, HEIGHT
from camera_parameters import *
from .image_utilities import crf_refinement, graph_cut_refinement, largest_region, fill_holes

from os import mkdir


@turn_ply_to_npy
def km_cluster(points: np.ndarray, k: int) -> [o3d.utility.Vector3dVector, np.ndarray]:
    """
    Using k-means to cluster point cloud
    :param points:
    :param k:
    :return:
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    label = kmeans.labels_
    np.random.seed(5)
    rand_col = np.random.random([k, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def gmm_cluster(points: np.ndarray):
    # PLACEHOLDER
    gausmmix = GaussianMixture(n_components=3, random_state=0)
    gausmmix.fit(points)


@turn_ply_to_npy
def dbscan_cluster(points: np.ndarray, epsilon: float) -> [o3d.utility.Vector3dVector, np.ndarray]:
    """
    Using dbscan to cluster point cloud
    :param points:
    :return:
    """
    db = DBSCAN(eps=epsilon, min_samples=2)
    print("[INFO] Fitting DBSCAN")
    db.fit(points)
    print("[INFO] Done")
    label = db.labels_
    n_clusters_ = len(set(label)) - (1 if -1 in label else 0)
    rand_col = np.random.random([n_clusters_, 3])
    coloured_points = rand_col[label]

    return o3d.utility.Vector3dVector(coloured_points), label


@turn_ply_to_npy
def reproject(points: np.ndarray, color: np.ndarray, label: np.ndarray,
              transformation_mat: np.ndarray, depth_map: np.ndarray,
              depth_range: float, name: str,
              cam_mat: np.ndarray = CAM_MAT, dist_mat: np.ndarray = DIST_MAT,
              height: int = HEIGHT, width: int = WIDTH, save_img: bool = False) -> np.ndarray:
    """
    Project all point cloud points into the image scene pixel points
    :param points: 3D point coordinates
    :param color: color vector for each point
    :param label: label for each point
    :param transformation_mat: transformation of the current camera location
    :param depth_map: depth map of image
    :param depth_range: acceptable deviation between distance and depth
    :param name: image name
    :param cam_mat: camera matrix
    :param dist_mat: distortion matrix [k1, k2, p1, p2]
    :param height: img height
    :param width: img width
    :param save_img: save label and visual image for debugging
    :return: image where each pixel has a label [0 = background]
    """

    R = transformation_mat[:3, :3]
    camera_center = transformation_mat[:3, 3]
    real_center = -1 * R.T.dot(camera_center.T)

    rvec = cv2.Rodrigues(R)
    cv_projection = cv2.projectPoints(objectPoints=points, rvec=rvec[0], tvec=transformation_mat[:3, 3],
                                      cameraMatrix=cam_mat, distCoeffs=dist_mat)

    pixels = np.rint(cv_projection[0]).astype('int')

    distance_map = points - np.repeat(real_center[np.newaxis, :], points.shape[0], axis=0)
    distance_map = np.linalg.norm(distance_map, axis=1)
    # distance_map = distance_map[:, 2]

    save_distance = np.zeros_like(depth_map)

    save_index = np.zeros([height, width], dtype='uint')

    # TODO vectorize
    for i, pixel in enumerate(pixels):
        y, x = pixel[0]
        if 0 <= x < height and 0 <= y < width:
            dist = distance_map[i]
            depth = depth_map[x, y]
            # abs_dist = abs(dist - depth)
            save_distance[x, y] = dist
            # if abs_dist <= abs(depth_range * dist):
            save_index[x, y] = i+1

    # based on the index select the respective index
    label = np.concatenate([np.zeros([1]), label+1], axis=0)
    reprojection = label[save_index]

    if save_img:
        print("[INFO] Saving debug images")
        cv2.imwrite(f"debug_images/label_projection_{name}.png", reprojection)

        # based on the index select the respective colour
        color = np.concatenate([np.zeros([1, 3]), color], axis=0)
        visual_label_img = color[save_index]

        visual_label_img = cv2.cvtColor(np.floor(visual_label_img*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"debug_images/visual_projection_{name}.png", visual_label_img)

    return reprojection, save_distance


def generate_masks(projection: np.ndarray, original: np.ndarray, growth_rate: int, shrink_rate: int,
                   distance_map: np.ndarray,
                   name: str, min_number: int, refinement_method: str, t: int, iter_count: int,
                   data_path: str = DATA_PATH, **kwargs):
    """
    Generate masks for each label instance (similar to blender mask output)
    :param projection: image with instance label for each pixel [0 = background]
    :param original: reprojected_cloud image
    :param growth_rate: number of dilation steps
    :param shrink_rate: number of erosion steps after dilation
    :param name: name of the current image
    :param min_number: minimum number of instanced of one label
    :param refinement_method: Chose "crf" or "graph"
    :param data_path: folder containing depth, images, masks, positions, pointclouds
    :param kwargs: largest: bool -> use only the largest connected region
                   fill: bool -> fill holes
                   graph_thresh: int -> threshold for gaussian blur for graph cut mask
    :return: save mask images to separate folders in masks
    """
    labels_present, counts = np.unique(projection, return_counts=True)
    filter_small = counts < min_number
    labels_present = np.delete(labels_present, filter_small, 0)
    labels_present = np.delete(labels_present, 0)
    labels_present = np.rint(labels_present).astype('uint8')

    largest_only = False
    fill = False
    graph_mask_thresh = 125
    for argument, value in kwargs.items():
        if argument == "largest" and value:
            largest_only = True
        elif argument == "fill" and value:
            fill = True
        elif argument == "graph_thresh":
            graph_mask_thresh = value
        else:
            print("[ERROR] Unknown keyword argument: ", argument)

    masks = np.zeros([labels_present.shape[0], projection.shape[0], projection.shape[1]], dtype='uint8')
    distances = []

    print("[INFO] Creating occlusion image")
    for i, label in enumerate(labels_present):
        instance = np.zeros_like(projection)
        instance[np.where(projection == label)] = 255
        instance = cv2.dilate(instance, np.ones((5, 5), 'uint8'), iterations=growth_rate)
        instance = cv2.erode(instance, np.ones((5, 5), 'uint8'), iterations=shrink_rate)
        # TODO think about using concave hull with dilation

        masks[i, :, :] = instance.astype('uint8')
        distances.append(sum(distance_map[projection == label]) / distance_map[projection == label].shape[0])

    # Check for occlusion by hierarchy
    distances = np.array(distances)
    sort_order = distances.argsort()
    labels_present = labels_present[sort_order]
    masks = masks[sort_order, :, :]

    empty_masks = []
    for i in range(labels_present.shape[0]):
        master = masks[i, :, :].copy()
        if master.any():
            for j in range(i + 1, labels_present.shape[0]):
                masks[j, :, :] = cv2.bitwise_or(masks[j, :, :], masks[j, :, :], mask=cv2.bitwise_not(master))
        else:
            empty_masks.append(i)

    # for i, instance in enumerate(labels_present):
    #     filter_img = cv2.bitwise_not((projection == instance) * cv2.bitwise_not(masks[i, :, :])) // 255
    #     projection *= filter_img
        # debug purposes (projection != 0).astype('uint8') * 255

    # delete empty masks
    masks = np.delete(masks, empty_masks, 0)
    labels_present = np.delete(labels_present, empty_masks, 0)

    print("[INFO] Processing label number: ", end='')
    for instance, label_num in zip(masks, labels_present):
        print(label_num, ", ", end='')
        mask_dir = data_path + f"/masks/mask{int(label_num)}/"
        try:
            mkdir(mask_dir)
        except FileExistsError:
            pass
        instance = largest_region(instance) if largest_only else instance
        instance = fill_holes(instance) if fill else instance

        # instance = cv2.GaussianBlur(instance, (101, 101), 0)
        instance = cv2.GaussianBlur(instance, (51, 51), 0)

        if refinement_method == "crf":
            # TODO add all other params
            label = crf_refinement(img=original, mask=instance, t=t, n_classes=2)
        elif refinement_method == "graph":
            instance = cv2.threshold(instance, graph_mask_thresh, 255, cv2.THRESH_BINARY)[1]
            label = graph_cut_refinement(img=original, mask=instance.copy(), iter_count=iter_count)
            # print("Is label equal to mask?: ", not (label - instance).any())
        else:
            print("[ERROR] No correct refinement method chosen!")
            instance = cv2.threshold(instance, graph_mask_thresh, 255, cv2.THRESH_BINARY)[1]
            label = instance

        if not label.any():
            print("\n[ERROR] There are no pixels present after refinement")
        cv2.imwrite(mask_dir + name + ".png", label)
    print()
