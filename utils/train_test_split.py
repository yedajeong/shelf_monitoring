import os
import shutil
import glob
import random
from distutils.dir_util import copy_tree

SRC_DIR = "/Users/dajeong/Desktop/Yennie/labeled_dataset"
DEST_DIR = "/Users/dajeong/Desktop/Yennie/split_dataset"
DATASET = "0207"

os.makedirs(f'{DEST_DIR}/{DATASET}/train', exist_ok=True)
os.makedirs(f'{DEST_DIR}/{DATASET}/valid', exist_ok=True)
os.makedirs(f'{DEST_DIR}/{DATASET}/test', exist_ok=True)

category = os.listdir(f'{SRC_DIR}/{DATASET}')
if '.DS_Store' in category:
    category.remove('.DS_Store')

filename = []

# print(category)

for cat in category:
    copy_tree(f'{SRC_DIR}/{DATASET}/{cat}', f'{DEST_DIR}/{DATASET}')
    
    # .txt는 삭제
    [os.remove(f) for f in glob.glob(f'{DEST_DIR}/{DATASET}/*.txt')]
    
    f_name = os.listdir(f'{SRC_DIR}/{DATASET}/{cat}')
    for f in f_name:
        filename.append(os.path.splitext(f)[0])


filename = list(set(filename))
if 'classes' in filename:
    filename.remove('classes')
if '.DS_Store' in filename:
    filename.remove('.DS_Store')


data_N = len(filename)

if int(data_N*0.8) % 2 == 0:
    trainVal_N = int(data_N*0.8)
else:
    trainVal_N = int(data_N*0.8) + 1
test_N = data_N - trainVal_N

if int(trainVal_N*0.7) % 2 == 0:
    train_N = int(trainVal_N*0.7)
else:
    train_N = int(trainVal_N*0.7) + 1
valid_N = trainVal_N - train_N

# shuffle on/off
random.shuffle(filename)

train_fname = filename[:train_N]
valid_fname = filename[train_N:train_N+valid_N]
test_fname = filename[train_N+valid_N:]


for file in train_fname:
    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.jpg', f'{DEST_DIR}/{DATASET}/train')
    except:
        print(file+'.jpg')

    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.xml', f'{DEST_DIR}/{DATASET}/train')
    except:
        print(file+'.xml')

for file in valid_fname:
    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.jpg', f'{DEST_DIR}/{DATASET}/valid')
    except:
        print(file+'.jpg')

    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.xml', f'{DEST_DIR}/{DATASET}/valid')
    except:
        print(file+'.xml')

for file in test_fname:
    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.jpg', f'{DEST_DIR}/{DATASET}/test')
    except:
        print(file+'.jpg')

    try:
        shutil.move(f'{DEST_DIR}/{DATASET}/{file}.xml', f'{DEST_DIR}/{DATASET}/test')
    except:
        print(file+'.xml')
