from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping

import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ----------------------
# tmp. 라벨링 데이터 추가해서 학습
# path = "/Users/dajeong/Desktop/Yennie/image_clustering"
# os.chdir(path)

# images = []
# xmls = []

# with os.scandir(path) as files:
#     for file in files:
#         if file.name.endswith('.xml'):
#             xmls.append(file.name)
#             images.append(file.name.split('.xml')[0] + '.jpg')

# coords = {}
# for fname in xmls:
#     coords.setdefault(fname, [])

#     xml_file = os.path.join(path, fname)
#     doc = ET.parse(xml_file)
#     root = doc.getroot()

#     coord_tag = root.findall("object")

#     for object in root.iter("object"):
#         coord = []
#         coord.append(int(object.find("bndbox").findtext("xmin")))
#         coord.append(int(object.find("bndbox").findtext("ymin")))
#         coord.append(int(object.find("bndbox").findtext("xmax")))
#         coord.append(int(object.find("bndbox").findtext("ymax")))

#         coords[fname].append(coord)


# for i, img in enumerate(images):
#     tmp = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

#     for j in range(len(coords[xmls[i]])):
#         key = img.split('.jpg')[0] + '_' + str(j) + '.jpg'
#         coord = coords[xmls[i]][j]
#         crop = tmp[coord[1]:coord[3], coord[0]:coord[2]]
#         cv2.imwrite(key, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
# ----------------------



# 0. global variable
# IMG_SIZE = 224
IMG_SIZE = 70

# 1. data frame (for training)
os.chdir(os.getcwd())
path = 'data_obj'

data = pd.DataFrame()

filename = os.listdir(path)
if '.DS_Store' in filename:
    filename.remove('.DS_Store')

filename_full = [os.path.join(path, fname) for fname in filename]


labels = []
for fname in filename:
    classNum = int(fname.split('c')[1].split('.jpg')[0])  # fname에서 c와 .jpg 사이 숫자(class 번호)
    labels.append(classNum)


img_array = []
for fname in filename:
    img = cv2.imread(os.path.join(path, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array.append(img)


data['filename'] = filename_full
data['labels'] = labels
data['img_array'] = img_array


# 2. train valid split (from. full df -> class별 데이터 순서 랜덤)
# sample 420개 -> 7(294) : 3(126)
train_df = data.iloc[:294, :]
valid_df = data.iloc[294:, :]


# 3. data augmentation(train data에만 적용)
num_augment_img = int(len(train_df) * 0.5)  # train data의 20%씩 랜덤노이즈 추가, masking
origin_train_len = len(train_df)  # 기존 원본 데이터에 대해서만
data_aug = pd.DataFrame()
fname_aug = []
labels_aug = []
img_array_aug = []

# 3-a) add random noise (train)
def add_noise(img):

    row, col, ch = img.shape
    
    # 랜덤 개수만큼
    number_of_pixels = random.randint(int(row*col*0.1), int(row*col*0.2))
    for i in range(number_of_pixels):
        
        # 랜덤 위치에
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
          
        # 랜덤 컬러로 채우기
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        img[y_coord][x_coord][0] = r
        img[y_coord][x_coord][1] = g
        img[y_coord][x_coord][2] = b
        
          
    return img


for i in range(num_augment_img):
    idx = random.randint(0, origin_train_len-1)

    img = cv2.cvtColor(cv2.imread(train_df.iloc[idx, 0]), cv2.COLOR_BGR2RGB)

    # 2. 랜덤 픽셀 랜덤 값으로 
    p = 'data_augment'
    label = train_df.loc[idx, 'labels']
    fname = os.path.join(p, f'noise_{i}_c{label}.jpg')
    noise = add_noise(img)

    plt.imsave(fname, noise)  # for flow_from_dataframe

    fname_aug.append(fname)
    labels_aug.append(label)
    img_array_aug.append(noise)
    # train_df.append({'filename':fname, 'labels':label, 'img_array':noise}, ignore_index=True)
    

    # 1. 가우시안 랜덤 노이즈 추가 (x)
    # row, col, ch= img.shape
    # mean = 0
    # var = 0.1
    # sigma = var**0.5
    # gauss = np.random.normal(mean,sigma,(row,col,ch))
    # gauss = gauss.reshape(row,col,ch)

    # noisy_array = img + gauss
    # noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
    


# 3-b) masking  (train) - r, g, b, y 색상별로 영역 마스킹
def masking(hsv):

    # opencv에서 색조(hue)범위는 0~180
    blue1 = np.array([80, 50, 50])
    blue2 = np.array([125, 255,255])
    green1 = np.array([30, 50, 50])
    green2 = np.array([80, 255,255])
    red1 = np.array([-15, 50, 50])
    red2 = np.array([20, 255,255])
    yellow1 = np.array([15, 30, 50])
    yellow2 = np.array([45, 255,255])
    purple1 = np.array([120, 30, 50])
    purple2 = np.array([170, 255, 255])

    mask_blue = cv2.inRange(hsv, blue1, blue2)
    mask_green = cv2.inRange(hsv, green1,green2)
    mask_red = cv2.inRange(hsv, red1, red2)
    mask_yellow = cv2.inRange(hsv, yellow1, yellow2)
    mask_purple = cv2.inRange(hsv, purple1, purple2)

    res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
    res_green = cv2.bitwise_and(img, img, mask=mask_green)
    res_red = cv2.bitwise_and(img, img, mask=mask_red)
    res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
    res_purple = cv2.bitwise_and(img, img, mask=mask_purple)

    return [res_blue, res_green, res_red, res_yellow, res_purple]


def masking_thres_out(img):
    row, col, _ = img.shape  # channel 정보 사용 x
    masked = 0

    for i in range(row):
        for j in range(col):
            # if img[i][j] == np.array([0, 0, 0]):
            if img[i][j][0]==0 and img[i][j][1]==0 and img[i][j][2]==0:
                masked += 1

    if masked > row*col*0.8:
        return True

    else:
        return False


# threshold out되는 데이터 있어서 전체 데이터셋에 대해 적용
for i in range(len(data)):
# for i in list(data[data['labels']==9].index):

    img = cv2.imread(data.iloc[i, 0])
    img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    result = masking(hsv)  # masked b, g, r, y, p (list)
    

    p = 'data_augment'
    label = data.loc[i, 'labels']
    bgryp = ['b', 'g', 'r', 'y', 'p']
    for j, res in enumerate(result):
        if masking_thres_out(res):  # 80% 영역이 masked out -> 버려
            continue
        else:
            fname = os.path.join(p, f'mask_{i}_{bgryp[j]}_{label}.jpg')
            
            plt.imsave(fname, res)  # for flow_from_dataframe
            
            fname_aug.append(fname)
            labels_aug.append(label)
            img_array_aug.append(res) 
            #  train_df.append({'filename':fname, 'labels':label, 'img_array':res}, ignore_index=True)


data_aug['filename'] = fname_aug
data_aug['labels'] = labels_aug
data_aug['img_array'] = img_array_aug

train_df = pd.concat([train_df, data_aug], ignore_index=True)

# print(train_df.tail(10))

train_df = pd.get_dummies(train_df, columns=['labels'])
valid_df = pd.get_dummies(valid_df, columns=['labels'])

y_col = valid_df.columns.tolist()  # one-hot encoding
y_col.remove('filename')
y_col.remove('img_array')

print(y_col)


# ImageDataGenerator: 학습 이미지 개수 늘리는게 x 학습 시마다 개별 원본 이미지를 변형해서 학습함
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                             horizontal_flip=True, fill_mode='nearest')
# train_gen = ImageDataGenerator(rescale=1./255, rotation_range=90, width_shift_range=0.1,
#                              height_shift_range=0.1, shear_range=0.5, zoom_range=0.3,
#                              horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
valid_gen = ImageDataGenerator(rescale=1./255)


train_data = train_gen.flow_from_dataframe(train_df, x_col='filename', y_col=y_col, target_size=(IMG_SIZE, IMG_SIZE),\
                                            color_mode='rgb', class_mode='raw', batch_size=16, shuffle=True)
valid_data = valid_gen.flow_from_dataframe(valid_df, x_col='filename', y_col=y_col, target_size=(IMG_SIZE, IMG_SIZE),\
                                        color_mode='rgb', class_mode='raw', batch_size=16, shuffle=True)



# 4. model build
# input_shape은 VGG16's default
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))  

conv_base.trainable = True

# for layer in conv_base.layers[:15]:
for layer in conv_base.layers[:11]:
    layer.trainable = False

conv_base.layers[14].trainable = False  # tmp
conv_base.layers[18].trainable = False  # blcok5 conv만 True, 풀링 레이어는 False

conv_base.summary()

# for layer in conv_base.layers:
#     print(layer, layer.trainable)


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(14, activation='softmax'))  # 14개 label -> softmax (for 다중분류)

# model.summary()


# 5. model compile, fitting
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=Adam(learning_rate=1e-5))

epochs = 100
# steps_per_epoch = len(X_train) // batch_size (최대 이만큼 가능)
# validation_steps = len(X_valid) // batch_size (최대 이만큼 가능)

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

history = model.fit(train_data, epochs=epochs, validation_data=valid_data, callbacks=[es])

model.save('VGG16_fine_tuning.h5')


# 6. result visualization
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
try:
    plt.imsave('train_val_acc.png')  # 왜 저장 안되지?
except:
    plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
try:
    plt.imsave('train_val_loss.png')
except:
    plt.show()
