import os
import shutil
import glob
import random
import cv2
from distutils.dir_util import copy_tree


os.chdir(os.getcwd())
# SRC_DIR = "lab"
SRC_DIR = "lab/to_crop"  # 일부 파일만 크롭
# DEST_DIR = "lab/gs_cropped"
DEST_DIR = "lab"
FLOOR = ['127', '128', '129', '130', '131']

start_pxl = [(42, 236), (115, 239), (193, 244), (262, 249), (323, 255)]
crop_size = [(168, 65), (163, 67), (152, 59), (142, 50), (131, 40)]

f_name = os.listdir(SRC_DIR)
if '.DS_Store' in f_name:
    f_name.remove('.DS_Store')

for file in f_name:
    img = cv2.imread(f'{SRC_DIR}/{file}')

    for i in range(5):
    # for i in range(2, 3):
        x, y = start_pxl[i]
        height, width = crop_size[i]
        
        try:
            crop = img[x:x+width, y:y+height]
        except Exception as e:
            print(file, e)  # .DS_Store 제거
            
        file_ = os.path.splitext(file)[0]
        # cv2.imwrite(f'{DEST_DIR}/{FLOOR[i]}/{file_}_'+str(i)+'.jpg', crop)
        cv2.imwrite(f'{DEST_DIR}/{file_}_'+str(i)+'.jpg', crop)

