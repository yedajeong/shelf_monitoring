import os
from distutils.dir_util import copy_tree
import cv2
import matplotlib.pyplot as plt


# 이미지 RGB, 픽셀 좌표 확인 (마우스오버)
img = cv2.cvtColor(cv2.imread('/Users/dajeong/Desktop/Yennie/image_clustering/blend_133.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# clustered -> 하나로 합치기
SRC_DIR = "/Users/dajeong/Desktop/Yennie/image_clustering/data_cluster"
DEST_DIR = "/Users/dajeong/Desktop/Yennie/image_clustering/data_obj"

category = os.listdir(SRC_DIR)
if '.DS_Store' in category:
    category.remove('.DS_Store')

for cat in category:  # 0~13
    f_name = os.listdir(f'{SRC_DIR}/{cat}')
    os.chdir(f'{SRC_DIR}/{cat}')

    for f in f_name:
        os.rename(f, f.split('_0.jpg')[0]+f'_c{cat}.jpg')


    copy_tree(f'{SRC_DIR}/{cat}', f'{DEST_DIR}')


# 하나로 합친 dir에서 train, test split
