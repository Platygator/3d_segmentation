"""
Created by Jan Schiffeler on 13.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from settings import *

cloud = o3d.io.read_point_cloud("/Users/jan/Programming/PycharmProjects/master/3d_segmentation/data/simulation_2/pointclouds/clustered.ply")
cloud_points = np.asarray(cloud.points)
position = [0.375561, -0.186453, 0.812475, 0.40506, -0.00877366, -0.583942, 1.50191]

camera_center = np.array([[position[4], position[5], position[6]]])
trans_mat = np.eye(4)
R = cloud.get_rotation_matrix_from_quaternion([position[0], position[1], position[2], position[3]])
trans_mat[:3, :3] = R
trans_mat[:3, 3] = position[-3:]

depth = np.load("depth.npy")
# distance = np.load("distance.npy")
# distance = np.load("distance_euc.npy")
# pixels = np.load("pixel.npy")

rvec = cv2.Rodrigues(trans_mat[:3, :3])
cv_projection = cv2.projectPoints(objectPoints=cloud_points, rvec=rvec[0], tvec=trans_mat[:3, 3],
                                  cameraMatrix=CAM_MAT, distCoeffs=DIST_MAT)
pixels = np.rint(cv_projection[0]).astype('int')

height, width = depth.shape

points_depth = []

distance = np.concatenate([cloud_points, np.ones([cloud_points.shape[0], 1])], axis=1)
distance = distance.dot(np.linalg.inv(trans_mat))
# distance = distance[:, 2]
distance1 = np.linalg.norm(distance, axis=1)

distance2 = cloud_points - np.repeat(camera_center, cloud_points.shape[0], axis=0)
distance2 = np.linalg.norm(distance2, axis=1)

# a = np.histogram(depth)
# depth[depth > 3] = 0
# img = depth.copy()/depth.max()
# img = np.transpose(np.repeat(img[np.newaxis, :, :], 3, axis=0), [1, 2, 0])

count = 0
img = np.zeros_like(depth)
for i, pixel in enumerate(pixels):
    y, x = pixel[0]
    if 0 <= x < height and 0 <= y < width:
        dist = distance2[i]
        dep = depth[x, y]
        if dep <= 0.01:
            count += 1
        dist_point = abs(dist - dep)
        img[x, y] = dist
        points_depth.append([dep, cloud_points[i, :], (y, x), dist])

points_depth_array = np.zeros([len(points_depth), 3])
points_depth_array2 = np.zeros([len(points_depth), 3])
origin = np.zeros([1, 3])
for i, point in enumerate(points_depth):
    points_depth_array[i, :] = camera_center + (point[1] - camera_center)/np.linalg.norm(point[1] - camera_center) * point[0]
    # points_depth_array[i, :] = origin + (point[1] - origin) * 10
    # points_depth_array2[i, :] = point[1] - camera_center

# all_points = np.concatenate([cloud_points, points_depth_array], axis=0)
all_points = np.concatenate([points_depth_array, cloud_points, origin, camera_center], axis=0)
cloud.points = o3d.utility.Vector3dVector(all_points)

colour0 = np.asarray(cloud.colors)
# colour = np.concatenate([colour, np.zeros_like(points_depth_array)], axis=0)
colour1 = np.zeros_like(points_depth_array)
colour1[:, 0] = 1
colour2 = np.zeros_like(points_depth_array2)
colour2[:, 2] = 1
colour3 = np.zeros_like(colour0)
colour3[:, 1] = 1
colourR = np.random.random([len(points_depth), 3])
colour = np.concatenate([colour1, colour2, np.zeros([1, 3]), np.array([[1, 0, 1]])], axis=0)
cloud.colors = o3d.utility.Vector3dVector(colour)

o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Origin Frame")


print("Done")
# plt.imshow(img)
# plt.imsave("depth_investigation_2.png", img)
# plt.show()