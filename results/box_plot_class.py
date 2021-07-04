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

import pandas as pd

renamer = {"r18_base_r139": "ResNet18", "r18_base_umarked": "ResNet18 \nunknown",
           "r50_r139_marked": "ResNet50 \nunknown", "r50_real_300": "ResNet50", "instance": "instance",
           "semantic": "semantic", "instance_ep_16": "Generated", "huang": "Huang et al.",
           "crf_all": "Label", "gt_train": "ResNet 18 \nGround Truth"}
names = [k for k in glob.glob("label_test/*.npy")]
results = [np.load(k, allow_pickle=True).item() for k in names]

res = {renamer[os.path.basename(n)[:-4]]: r for n, r in zip(names, results)}

class_names = []
run_name = []
run_data = []

for name, r in res.items():
  border_name = ["Border" for k in range(len(r['Border']))]
  stone_name = ["Stone" for k in range(len(r['Stone']))]
  mean_name = ["Mean" for k in range(len(r['Mean']))]
  class_names.extend(border_name + stone_name + mean_name)
  run_name.extend([name for k in border_name + stone_name + mean_name])
  run_data.extend(r['Border'] + r['Stone'] + r['Mean'])

data = {'Run': run_name,
        'Class': class_names,
        'IoU': run_data
        }

df = pd.DataFrame(data, columns=['Run', 'Class', 'IoU'])

# res_sort = dict(sorted(res.items(), key=lambda x: np.mean(x[1])))
sns.set_theme(style="ticks")
sns.boxplot(data=df, x='Run', y='IoU', hue='Class', width=0.4, whis=10,
            order=["semantic", "instance"])
            # order=["ResNet50", "ResNet18", "ResNet18 \nunknown", "ResNet50 \nunknown"])


# plt.xticks(plt.xticks()[0], [k for k in res_sort.keys()])

plt.title("IoU Label")
plt.legend(loc="upper left")
plt.xlabel("")
plt.ylabel("mean IoU over all classes")
plt.ylim([20, 100])
# sns.despine(trim=True, left=True)
# sns.boxplot(x="names", y="results", data=res,
#             whis=[0, 100], width=.6, )
plt.show()

