import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.getcwd(), 'results_6')
os.chdir(path)
n_cluster = 14

for cluster in range(n_cluster):
    objects = os.listdir(str(cluster))

    if len(objects) > 20:
        print(f"Clipping cluster:{cluster} size {len(objects)} -> 20")
        objects = objects[:20]

    fig = plt.figure()
    rows = 4
    cols = 5

    for i, obj in enumerate(objects):
        i += 1  # 1부터 시작
        img = cv2.imread(os.path.join(path, f'{cluster}/{obj}'))
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]), ax.set_yticks([])
    
    plt.suptitle(f'cluster:{cluster}')
    plt.savefig(f'{path}/cluster_{cluster}.jpg')
    plt.show()

