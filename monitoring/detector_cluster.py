# loading/processing image data
from keras.applications.vgg16 import preprocess_input
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import xml.etree.ElementTree as ET
import cv2

# feature extraction
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model

# clustering, dimension reduction
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

import os
from pathlib import Path
import numpy as np
import csv


# IMG_SIZE = 224
IMG_SIZE = 70  # input image(frame)에서 box친 object의 resized image size (70*70*3)
# k = 3  # best k 설정
script_path = os.path.dirname(os.path.realpath(__file__))  # 현재 열려있는 파일의 절대경로 - 파일 떼고 디렉토리 명만 얻어서 변수에 저장

class ClusterDetector(object):

    def __init__(self):
        self.bbox_model = models.load_model(os.path.join(script_path, 'resnet101_pascal_0216.h5'), backbone_name='resnet101')
        self.bbox_model = models.convert_model(self.bbox_model)

        self.feat_model = load_model(os.path.join(script_path, 'VGG16_fine_tuning_4.h5'))
        self.feat_model = Model(inputs=self.feat_model.inputs, outputs=self.feat_model.layers[-1].output)

        self.objects = {}  # key: object numbering  value: bbox coord([xmin, ymin, xmax, ymax])
        self.features = {}  # key: object numbering  value: extracted feature vector (from fine-tuning vgg16)
        self.groups = {}  # key: cluster numbering  value: list of object numbering
        # self.unique_labels = [i for i in range(k)]
        # self.colors = [(int(255/k)*i, int(255/k)*i, int(255/k)*i) for i in range(1, k+1)]  # rgb color (cluster 개수만큼 지정)

        # self.n_components = 0


    def detect_image(self, input_img, store_id=0, port=0, category_id=0, is_test=False):
        
        p = Path('static/image/process/{}/{}/{}'.format(store_id, port, category_id))
        if False == p.exists():
            p.mkdir(parents=True)

        input_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
        
        
        if is_test:  # bbox labeled img로 테스트 (학습시킨 self.bbox_model이 bbox를 잘 잡는다고 가정)
            # image, xml filename list
            images = []
            xmls = []

            images.append("2_0_127_2023_02_02_13_56_45.jpg")
            xmls.append("2_0_127_2023_02_02_13_56_45.xml")

            # 이미지 하나 당 존재하는 bbox의 [x_min, y_min, x_max, y_max]
            coords = {}
            for fname in xmls:
                coords.setdefault(fname, [])

                xml_file = fname
                doc = ET.parse(xml_file)
                root = doc.getroot()

                for object in root.iter("object"):   # sample data에서는 이미지 한 장 당 box 한 개
                    coord = []
                    coord.append(int(object.find("bndbox").findtext("xmin")))
                    coord.append(int(object.find("bndbox").findtext("ymin")))
                    coord.append(int(object.find("bndbox").findtext("xmax")))
                    coord.append(int(object.find("bndbox").findtext("ymax")))

                    coords[fname].append(coord)


            for i, img in enumerate(images):
                tmp = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                for j in range(len(coords[xmls[i]])):
                    key = j
                    coord = coords[xmls[i]][j]
                    self.objects.setdefault(key, coord)  # bbox coord 정보 dict에 저장

            # 결과 저장 위치 (is_test==True)
            path = os.getcwd()
        
        
        elif is_test==False:
            float_image = preprocess_image(input_img)  # subtracting the ImageNet mean from input image
            final_image, scale = resize_image(float_image)  # resize an image such that the size is constrained to min_side and max_side

            boxes, scores, _ = self.bbox_model.predict_on_batch(np.expand_dims(final_image, axis=0))  # bbox labels는 필요 x
            boxes /= scale

            boxes = np.squeeze(boxes)  # [[1,2,3]] -> [1,2,3]
            scores = np.squeeze(scores)

            result_boxes = list()

            for idx, score in enumerate(scores):
                if score >= 0.4:  #confidence 0.4 이상만 후보로 추출
                    result_boxes.append(boxes[idx])

            # obj 하나 당 후보 박스 여러개 (result_boxes) 중에서 하나씩만 골라내기 by.nms (piked_boxes)
            picked_boxes = self.non_max_suppression_fast(np.asarray(result_boxes), overlapThresh=0.8)
            for idx, box in enumerate(picked_boxes):
                self.objects.setdefault(idx, box)
                
            # 결과 저장 위치 (is_test==False)
            # path = f'static/image/process/{store_id}/{port}/{category_id}/diff_obj'
            # os.makedirs(path, exist_ok=True)
        

        return_boxes = []  # 함수 return
        return_img = input_img.copy()  # 다른 상품 디텍팅된 bbox 누적시켜서 저장할 이미지

        sort_list = sorted(self.objects.items(), key=lambda x: x[1][0])
        objects = {}
        for key, value in sort_list:
            objects.setdefault(key, value)
        self.objects = objects

        self.extract_features(input_img)
        
        ref_feat_p = os.path.join(p, 'ref_feature.npy')
        ref_xmin_p = os.path.join(p, 'ref_xmin.npy')
        if os.path.isfile(ref_feat_p) and os.path.isfile(ref_xmin_p):
            ref_feat = np.load(ref_feat_p)
            ref_xmin = np.load(ref_xmin_p)
        else:
            ref_feat = np.array(list(self.features.values()))
            ref_xmin = np.array([xmin for xmin, _, _, _ in list(self.objects.values())])
            np.save(ref_feat_p, ref_feat)
            np.save(ref_xmin_p, ref_xmin)
            return []  # return empty results

        for i, compare in enumerate(feat):
            similarity = self.cos_sim(feat[i], self.features[i])
            close = abs(ref_xmin[i] - self.objects[i][0])

            if similarity < 0.7 and close < 50:
                x1, y1, x2, y2 = self.objects[i]

                return_boxes.append([x1, y1, x2, y2])
                return_img = cv2.rectangle(return_img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)


        if is_test:
            cv2.imwrite(os.path.join(p, 'test_result.jpg'), cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(os.path.join(p, 'result.jpg'), cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB))
        
        return return_boxes


    def cos_sim(self, feat1, feat2):
        feat2 = np.transpose(feat2)
        return np.dot(feat1, feat2)/(np.linalg.norm(feat1)*np.linalg.norm(feat2))


    def intersection(self, box1, box2):
        # box = (xmin, ymin, xmax, ymax)
        # x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # intersection's width, height
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        return inter
    
    
    
    def calc_area(self, coord):
        # x1, y1, x2, y2로 면적 계산
        w = coord[2] - coord[0]
        h = coord[3] - coord[1]
        return w*h              



    def non_max_suppression_fast(self, boxes, overlapThresh):
		# no boxes, return an empty list
        if len(boxes) == 0:
            return []

		# if the bounding boxes integers, convert them to floats (for bunch of divisions)
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

		# list of picked indexes
        pick = []

		# bbox coordinate
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

		# compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

		# while some indexes still remain in the index list
        while len(idxs) > 0:
			# grab the last index in the indexes list and add the
			# index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                    np.where(overlap > overlapThresh)[0])))


        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")



    def extract_features(self, input_img):
        keys = list(self.objects.keys())

        for key in keys:
            # cropped object -> IMG_SIZE * IMG_SIZE resize
            b = np.array(self.objects[key]).astype(int)
            obj_crop = input_img[b[1]:b[3], b[0]:b[2]]  # (slicing은 x, y 반대..) lu, rd 좌표값으로 input image에서 탐지된 obj만 전체 이미지에서 슬라이싱
            
            obj_resize = cv2.resize(obj_crop, (IMG_SIZE, IMG_SIZE))  # type(object): np.array

            # model input으로 reshape -> num_of_samples, dim 1, dim 2, channels
            reshaped_img = obj_resize.reshape(1, IMG_SIZE, IMG_SIZE, 3)

            # image for model input
            img_pp = preprocess_input(reshaped_img)

            # feature vector (from vgg16)
            feat = self.feat_model.predict(img_pp, use_multiprocessing=True)

            self.features[key] = feat


    '''
    # (차원 축소 생략함)
    def dim_reduction(self, random_state=22):
        feat = np.array(list(self.features.values()))
        feat = feat.reshape(-1, self.feat_model.output_shape[-1])  # output layer(dense) 노드 개수

        pca = PCA(n_components=self.n_components, random_state=random_state)
        pca.fit(feat)

        x = pca.transform(feat)

        return x
    '''


    def clustering(self, input_img, is_test=False, random_state=22):
        self.extract_features(input_img)
        # x = self.dim_reduction()
        
        # 그림자 없는 127(제일 윗층) 칸은 그냥 input image data로 했을 때 더 잘 되고
        # 나머지 128~131은 그림자 때문인지 raw image(data)만 넣으면 군집화 잘 안됨 (feature 추출하는게 군집 더 잘 나눠짐)
        # -> kmeans input으로 data 사용할거면 아래쪽 선반은 이미지 밝기 높여서 해보기 (수정)

        # kmeans input (1)
        feat = np.array(list(self.features.values()), np.float32)
        feat = feat.reshape(-1, self.feat_model.output_shape[-1])

        # kmeans input (2)
        data = []
        for key in list(self.objects.keys()):
            b = np.array(self.objects[key]).astype(int)
            obj_crop = input_img[b[1]:b[3], b[0]:b[2]]
            obj_resize = cv2.resize(obj_crop, (IMG_SIZE, IMG_SIZE))
            obj_reshape = obj_resize.reshape(IMG_SIZE*IMG_SIZE*3)
            data.append(obj_reshape)
        data = np.array(data, np.float32)


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        # comp, labels, centers = cv2.kmeans(data, K=len(self.unique_labels), bestLabels=None,\
        #                                 criteria=criteria, attempts=10, flags=flags)
        comp, labels, centers = cv2.kmeans(feat, K=len(self.unique_labels), bestLabels=None,\
                                        criteria=criteria, attempts=10, flags=flags)

        labels = list(np.squeeze(labels))

        obj_num = np.array(list(self.features.keys()))
        
        for obj, cluster in zip(obj_num, labels):
            if cluster not in self.groups.keys():
                self.groups[cluster] = []  # key 없으면 새로 만들기

            self.groups[cluster].append(obj)

        


        # view_cluster (for test)
        if is_test:
            os.makedirs('cluster', exist_ok=True)
            for cluster in labels:
                for obj in self.groups[cluster]:
                    b = np.array(self.objects[obj]).astype(int)
                    crop = input_img[b[1]:b[3], b[0]:b[2]]
                    os.makedirs((f'cluster/{cluster}'), exist_ok=True)
                    cv2.imwrite(f'cluster/{cluster}/{obj}.jpg', cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))




if __name__ == '__main__':
    input_img = cv2.imread("2_0_127_2023_02_02_13_56_45.jpg")  # input: rgb img로 cvt
    cd = ClusterDetector()
    result = cd.detect_image(input_img, is_test=True)
    
    print(len(result) > 0)

