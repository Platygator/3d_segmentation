"""
Created by Jan Schiffeler on 10.06.21
jan.schiffeler[at]gmail.com

Changed by


Python 3.

"""
import numpy as np
import open3d as o3d
from settings import DATA_PATH
import json

cloud = o3d.io.read_point_cloud(f"{DATA_PATH}/pointclouds/point_cloud.pcd")
colours = np.asarray(cloud.colors)


# labels = {label: point for label, point in zip(range(1, 27), [[] for i in range(26)])}
with open(f"{DATA_PATH}/pointclouds/point_cloud.pcd.json") as d:
    data = json.load(d)

key2label = {k['key']: k['classTitle'] for k in data['objects']}

# labels1 = {key2label[k['objectKey']]: k['geometry']['indices'] for k in data['figures']}
labels = {}
for k in data['figures']:
    if key2label[k['objectKey']] not in labels.keys():
        labels[key2label[k['objectKey']]] = k['geometry']['indices']
    else:
        labels[key2label[k['objectKey']]] = labels[key2label[k['objectKey']]] + k['geometry']['indices']

delete_point = labels['s99']
labels.pop('s99')
labeled_points = np.zeros([colours.shape[0]], dtype=int)
for label, points in labels.items():
    if label != 's99':
        labeled_points[points] = label[1:]

points = np.asarray(cloud.points)
points = np.delete(points, labeled_points == 0, 0)
labeled_points = np.delete(labeled_points, labeled_points == 0, 0)
rand_col = np.random.random([len(labels) + 1, 3])  # 28
coloured_points = rand_col[labeled_points]


cloud.colors = o3d.utility.Vector3dVector(coloured_points)
cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries_with_editing([cloud])
o3d.io.write_point_cloud(f"{DATA_PATH}/pointclouds/clustered.ply", cloud)
np.save(f"{DATA_PATH}/pointclouds/labels.npy", labeled_points)
