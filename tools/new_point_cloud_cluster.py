"""
Created by Jan Schiffeler on 15.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
import open3d as o3d
from settings import DATA_PATH
import pyransac3d as prc
from utilities import *
import matplotlib.pyplot as plt

np.random.seed(5)

step = 0.2
k = 15

# FILTER
# load
try:
    cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/filtered_point_cloud.ply")
except FileNotFoundError:
    print("[ERROR] No filtered file found")
    quit()

origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=3, origin=[0, 0, 0])

normal_moved = move_along_normals(points=cloud.points, normals=cloud.normals, step=step)
cloud.points = normal_moved

c, inliers = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.1)
print(len(inliers))

inlier_cloud = cloud.select_by_index(inliers)
outlier_cloud = cloud.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], width=3000, height=1800, window_name="Unsure")

clustered, labels = km_cluster(points=inlier_cloud.points, k=k)
# labels = np.asarray(inlier_cloud.cluster_dbscan(eps=0.05, min_points=15))
max_label = labels.max()
print(max_label)
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
inlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
# cloud.colors = clustered

normal_moved_back = move_along_normals(points=inlier_cloud.points, normals=inlier_cloud.normals, step=-step)
inlier_cloud.points = normal_moved_back

normal_moved_back = move_along_normals(points=cloud.points, normals=cloud.normals, step=-step)
cloud.points = normal_moved_back

# o3d.visualization.draw_geometries([inlier_cloud], width=3000, height=1800, window_name="Clustered")
# quit()

points = np.asarray(cloud.points)
all_labels = np.ones([points.shape[0]]) * -1
all_labels[inliers] = labels
centers = []
radii = []
spheres = []
label_list = []
for l in range(max_label):
    point_set = points[all_labels == l]
    inliers = np.where(all_labels == l)[0].tolist()
    if len(inliers) < 50:
        continue
    inlier_cloud = cloud.select_by_index(inliers)
    outlier_cloud = cloud.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    center = np.mean(point_set, axis=0)[np.newaxis, :]
    radius = np.mean(np.linalg.norm(points - np.repeat(center, points.shape[0], axis=0), axis=1)) * 0.3
    # sph = prc.Sphere()
    # center, radius, inliers = sph.fit(point_set, thresh=0.4)
    if radius < 50.0:
        centers.append(center)
        radii.append(radius)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(colors[l][:-1])
        sphere.translate(center[0])
        spheres.append(sphere)
        label_list.append(l)

        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, sphere], width=3000, height=1800, window_name="Unsure")

belonging = np.zeros([points.shape[0], len(label_list)])
normals = np.asarray(cloud.normals)
for i in range(len(centers)):
    # distance = points - np.repeat(np.array([centers[i]]), points.shape[0], axis=0)
    distance_xyz = points - np.repeat(centers[i], points.shape[0], axis=0)
    distance = np.linalg.norm(distance_xyz, axis=1)
    distance_xyz /= distance[:, np.newaxis].repeat(3, axis=1)
    normal_oriention = np.sum(normals * distance_xyz, axis=1)

    # belonging_col = 1 - (distance - np.ones_like(distance) * radii[i])**2 * normal_oriention
    belonging_col = 1 - (distance * normal_oriention**2)
    belonging[:, i] = belonging_col


most_likely_label_id = np.argmax(belonging, axis=1)
most_likely_label = np.array(label_list)[most_likely_label_id]

rand_col = np.random.random([most_likely_label.max(), 3])
coloured_points = rand_col[most_likely_label - 1]
cloud.colors = o3d.utility.Vector3dVector(coloured_points)

# o3d.visualization.draw_geometries([cloud, *spheres], width=3000, height=1800, window_name="Clustered")
o3d.visualization.draw_geometries([cloud], width=3000, height=1800, window_name="Clustered")
