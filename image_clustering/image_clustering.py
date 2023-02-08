# loading/processing image data
from keras.applications.vgg16 import preprocess_input
import xml.etree.ElementTree as ET
import cv2

# feature extraction
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model

# clustering, dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# IMG_SIZE = 224
IMG_SIZE = 70
n_components = 100


# -----------------------
# 1. dataset load
# -----------------------
'''
# 1-a) 진열대 이미지 + xml(bbox info) 로 -> object만 crop
path = "/Users/dajeong/Desktop/Yennie/image_clustering/data"
os.chdir(path)

# image, xml filename
images = []
xmls = []

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.jpg'):
            images.append(file.name)
            xmls.append(file.name.split('.jpg')[0] + '.xml')


# 이미지 하나 당 존재하는 bbox의 [x_min, y_min, x_max, y_max]
coords = {}
for fname in xmls:
    coords.setdefault(fname, [])

    xml_file = os.path.join(path, fname)
    doc = ET.parse(xml_file)
    root = doc.getroot()

    coord_tag = root.findall("object")

    for object in root.iter("object"):   # sample data에서는 이미지 한 장 당 box 한 개
        coord = []
        coord.append(int(object.find("bndbox").findtext("xmin")))
        coord.append(int(object.find("bndbox").findtext("ymin")))
        coord.append(int(object.find("bndbox").findtext("xmax")))
        coord.append(int(object.find("bndbox").findtext("ymax")))

        coords[fname].append(coord)



objects = {}  # key: filename_objNum.jpg  value: bbox만 따로 크롭
for i, img in enumerate(images):
    tmp = cv2.imread(img)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    for j in range(len(coords[xmls[i]])):
        key = img.split('.jpg')[0] + '_' + str(j) + '.jpg'
        objects.setdefault(key, None)
        coord = coords[xmls[i]][j]  # 지금은 이미지 한 장 당 object 하나밖에 없음 (coords[xmls[i]][0])
        crop = tmp[coord[1]:coord[3], coord[0]:coord[2]]
        objects[key] = crop


# plt.imshow(objects[images[0].split('.jpg')[0] + '_0.jpg'])
# plt.axis("off")
# plt.show()
'''


# 1-b) object image 불러오기 (from /data_obj/filename_cNum.jpg)
objects = {}  # key: filename_classNum.jpg  value: bbox만 따로 크롭
os.chdir(os.getcwd())
p = 'data_obj'
filename = os.listdir(p)

if '.DS_Store' in filename:
    filename.remove('.DS_Store')

for fname in filename:
    img = cv2.imread(os.path.join(p, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    objects.setdefault(fname, img) 



# -----------------------
# 2. model load (feature extraction)
# -----------------------

# 2-a) freeze
# 마지막 fc layer(for prediction) 분류기 제거
# # 출력층 노드 개수: 4,096
# model = VGG16()
# model = Model(inputs=model.inputs, outputs=model.layers[-2].output)



# 2-b) fine tuning
# 출력층 노드 개수: 기존 4096 (with 224x224 input tensor)
model = load_model('model/VGG16_fine_tuning_4.h5')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# model.summary()

# '''
def extract_features(object, model):
    # cropped object -> 224x224 array (vgg model input shape: 224by224의 넘파이 배열)
    img = cv2.resize(object, (IMG_SIZE, IMG_SIZE))  # type(object): np.array

    # model input으로 reshape -> num_of_samples, dim 1, dim 2, channels
    reshaped_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # image for model input
    img_pp = preprocess_input(reshaped_img)

    # feature vector (from vgg16)
    features = model.predict(img_pp, use_multiprocessing=True)
    return features



data = {}  # key: image_objNum.jpg  value: object's feature
p = "data/features"

keys = list(objects.keys())

for key in keys:
    try:
        # feature extraction -> dictionary update
        feature = extract_features(objects[key], model)
        data[key] = feature
    except:
        # feature 추출 -> pickle로 저장 (혹시 모를 fail..)
        with open(p, 'wb') as file:
            pickle.dump(data, file)


# filenames(full_file + object_num .jpg) list
filenames = np.array(list(data.keys()))

# features list (object별 feature)
features = np.array(list(data.values()))


# reshape -> ( n_samples * n_vectors ) n_vectors: output layer 노드 개수
features = features.reshape(-1, model.output_shape[-1])

unique_labels = [i for i in range(14)]



# -----------------------
# 3. dimension reduction
# -----------------------

# feature vector 차원 축소
pca = PCA(n_components=n_components, random_state=22)  # n_components 임의로 (100개) 선택
pca.fit(features)
x = pca.transform(features)  # 2d array


# -----------------------
# 4. clustering
# -----------------------

# 1) kmeans
def view_cluster(cluster):
    # cluster 내 object (.jpg filename)
    objs = groups[cluster]

    for obj in objs:
        # objs[0] = 'triplet_1674036109.5201995_0_0.jpg'  -> imsave 시 확장자 지정 x
        plt.imsave(f'{cluster}/{obj}', objects[obj])


kmeans = KMeans(n_clusters=len(unique_labels), init='k-means++', random_state=22, max_iter=300)
# kmeans = KMeans(n_clusters=len(unique_labels), random_state=22)
kmeans.fit(x)

# { id: [filename] } 으로 묶기
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []  # key 없으면 새로 만들기

    groups[cluster].append(file)


# '''
# cluster 확인
path = 'results'
# path = '/Users/dajeong/Desktop/Yennie/image_clustering/results'
os.makedirs(path, exist_ok=True)
os.chdir(path)

for cluster in kmeans.labels_:
    os.makedirs(str(cluster), exist_ok=True)
    view_cluster(cluster)
# '''


# 2) cosine similarity
def get_similar(idx , top_n = 30):  # true sample len per clstr: 30
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # (iter, score)
    sim_scores = sim_scores[1:top_n+1] # 자기자신 제외 (sorting시 자기 자신과 sim값==1.0 -> 내림차순 정렬했을 때 idx==0)
    idx_rec    = [sim[0] for sim in sim_scores]
    idx_sim    = [sim[1] for sim in sim_scores]
    
    return idx_rec, idx_sim


def plot_figures(figures, nrows = 1, ncols=1,figsize=(6, 5)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, img in enumerate(figures.values()):
        axeslist.ravel()[ind].imshow(img)
        # axeslist.ravel()[ind].set_title()
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()


cosine_sim = 1 - pairwise_distances(x, metric='cosine')
index = range(len(data))
key = list(data.keys())

p = 'data_cluster'
idx_ref = []
for i in range(14):
    filename = os.listdir(os.path.join(p, str(i)))
    if '.DS_Store' in filename:
        filename.remove('.DS_Store')
    
    for idx, fname in enumerate(key):
        if fname == filename[0]:
            idx_ref.append(idx)


# plt.figure(figsize = (2,2))
for idx in idx_ref:
    idx_rec, idx_sim = get_similar(idx)  # idx_rec: idx_ref와의 유사도 상위 30개 sample's index
    figures = {'img-'+str(i): objects[key[rec]] for i, rec in enumerate(idx_rec)}
    plot_figures(figures, 6, 5)





# 4-a) best-k 찾기
# sse = []
# list_k = list(range(3, 50))
# 
# for k in list_k:
#     km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
#     km.fit(x)
# 
#     sse.append(km.inertia_)
# 
# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance');


# clustering 돌린 후 인풋 이미지 상에서 바운딩 박스의 소속 cluster을 표시
# -> 다른 cluster의 상품이 포함돼있으면 빈공간으로 간주 = 다른 cluster의 물체에 바운딩박스 치지 않기
# ==> 이렇게 ??
# '''