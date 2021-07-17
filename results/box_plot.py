"""
Created by Jan Schiffeler on 01.07.21
jan.schiffeler[at]gmail.com

Changed by


Python 3.

"""

import seaborn as sns
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

renamer = {"r18_base_r139": "ResNet18", "r18_base_umarked": "ResNet18 \nunknown",
           "r50_r139_marked": "ResNet50  \nunknown", "r50_real_300": "ResNet50", "instance": "Instance",
           "semantic": "Semantic", "instance_ep_16": "Ours", "huang": "Huang et al."}
names = [k for k in glob.glob("instance/*.npy")]
results = [np.load(k, allow_pickle=True)*100 for k in names]

res = {renamer[os.path.basename(n)[:-4]]: r for n, r in zip(names, results)}
res_sort = dict(sorted(res.items(), key=lambda x: np.mean(x[1])))
sns.set_theme(style="ticks")
sns.boxplot(data=[k for k in res_sort.values()], width=0.4, whis=10)


plt.xticks(plt.xticks()[0], [k for k in res_sort.keys()])

plt.title("IoU Instance Segmentaiton")
# plt.xlabel("Experiment name")
plt.ylabel("mean IoU")
plt.ylim([60, 85])
# sns.despine(trim=True, left=True)
# sns.boxplot(x="names", y="results", data=res,
#             whis=[0, 100], width=.6, )
plt.show()

