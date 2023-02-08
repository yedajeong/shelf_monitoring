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
# from sklearn.cluster import KMeans  # cv2.kmeans로 변경
# from sklearn.decomposition import PCA

import os
import numpy as np


# IMG_SIZE = 224
IMG_SIZE = 70  # input image(frame)에서 box친 object의 resized image size (70*70*3)
k = 3  # 뉴월드_best k 찾도록 (수정)
script_path = os.path.dirname(os.path.realpath(__file__))  # 현재 열려있는 파일의 절대경로 - 파일 떼고 디렉토리 명만 얻어서 변수에 저장

class ClusterDetector(object):

    def __init__(self):
        self.bbox_model = models.load_model(os.path.join(script_path, 'resnet101_pascal_0120.h5'), backbone_name='resnet101')
        self.bbox_model = models.convert_model(self.bbox_model)

        self.feat_model = load_model(os.path.join(script_path, 'VGG16_fine_tuning_4.h5'))
        self.feat_model = Model(inputs=self.feat_model.inputs, outputs=self.feat_model.layers[-1].output)

        self.objects = {}  # key: object numbering  value: bbox coord([xmin, ymin, xmax, ymax])
        self.features = {}  # key: object numbering  value: extracted feature vector (from fine-tuning vgg16)
        self.groups = {}  # key: cluster numbering  value: list of object numbering
        self.unique_labels = [i for i in range(k)]

        self.n_components = 0


    def detect_image(self, input_img, store_id=0, port=0, category_id=0, is_test=False):
        '''
            return: 상품 군집 내 다른 상품 디텍팅 됐을 경우 해당 obj의 bbox 좌표들의 list
            
            +) input image에 박스 그려서 지정된 path에 저장
            
            1. input image 내의 bbox 읽어서 self.objects에 딕셔너리로 저장함 (coord 4개값)
            
            2. 한 cluster 내에 포함된 obj들의 bbox coord (xmin, ymin, xmax, ymax)를 각각 lu(min), rd(max)에 저장함
            lu중 min_x, rd중 max_x에 해당하는 obj 사이를 input image 내의 이 cluster의 영역으로 지정,
            이 때 예외 사항의 경우 여러 개의 clstr로 분할될 수 있음

            3. 다른 cluster들의 obj를 순회하면서 현재 지정된 cluster의 영역 내에 bbox가 위치했을 경우
            = obj 면적의 80% 이상이 현재 지정된 cluster 영역과 겹칠 경우라고 판단
        '''
        
        input_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
        
        if is_test:  # bbox labeled img로 테스트 (학습시킨 self.bbox_model이 bbox를 잘 잡는다고 가정했을 때..)
            # image, xml filename
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
            path = f'static/image/process/{store_id}/{port}/{category_id}/diff_obj'
            os.makedirs(path, exist_ok=True)
        
        
        self.n_components = min(len(self.objects), 100)

        # clustering결과 -> self.groups에 저장
        self.clustering(input_img, is_test=is_test)

        
        return_boxes = []  # 함수 return
        return_img = input_img.copy()  # 다른 상품 디텍팅된 bbox 누적시켜서 저장할 이미지
        
        for cluster in self.groups.keys():
            objs = self.groups[cluster]

            # 물체 1개 -> pass (cluster로 묶지 않음)
            if len(objs) < 2:
                continue
            
            lu = []
            rd = []
            for obj in objs:
                xmin, ymin, xmax, ymax = self.objects[obj]
                lu.append((xmin, ymin))
                rd.append((xmax, ymax))

            # y값 기준 오름차순 정렬
            lu.sort(key = lambda x:x[1]) 
            rd.sort(key = lambda x:x[1])
            # cluster 영역의 lu, rd 좌표 -> list로 저장
            coord = [lu[0][0], lu[0][1], rd[-1][0], rd[-1][1]]
            clstr_area = [coord]

            '''
            # 리스트의 obj 순회하면서 range 이상으로 x좌표값이 확 튈 때 해당 obj 기준으로 분할 (수정)
            area = self.calc_area(clstr_area[cluster][0])
            obj_area = self.calc_area(self.objects[objs[0]])
            if area > obj_area * len(objs) * 1.5:  # 2? 값 조절하면서 확인하기 -> obj_area * len 말고 obj_area 각각 구한거 더해서 계산하기 (수정)
                clstr_area = []
                split = True
                i = 0
                while(split and i < len(lu)-1):
                    if (lu[i][2]-lu[i][0]) * 2 < lu[i+1][0]:
                        left_lu = lu[:i+1]
                        right_lu = lu[i+1:]
                        left_rd = rd[:i+1]
                        right_rd = rd[i+1:]
                        
                        # 분할 왼/오 len() 계산해서 len<2이면 pass (묶지 않음) len>1이면 더 쪼갤 수 있는지 확인
                        if len(left_lu) > 1:
                            clstr_area.append([left_lu[0][0], left_lu[0][1], left_rd[-1][0], left_rd[-1][1]])
                        else:
                            pass
                            
                        if len(right_lu) > 1:
                            split = True  # 오른쪽 한 번 더 스캔.. 남은 거 없을 때까지
                            lu = right_lu
                            rd = right_rd
                            i = 0
                        else:
                            split = False
                    else:
                        i += 1
            '''

            for other in self.groups.keys():
                if other == cluster:
                    continue
                else:
                    objs_other = self.groups[other]
                    for obj_other in objs_other:
                        # cluster 안에 area 여러 개일 때도 각각 비교연산 (수정)
                        # inter = np.array([self.intersection(area, self.objects[obj_other]) for area in clstr_area])
                        # inter = np.where(inter > self.calc_area(self.objects[obj_other])*0.8)  # obj_other의 80% 이상이 다른 cluster 영역과 겹치면 다른 상품으로 디텍팅
                        # if len(inter) > 0:
                        #     return_img = cv2.rectangle(return_img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                        #     return_boxes.append([x1, y1, x2, y2])

                        inter = self.intersection(clstr_area[0], self.objects[obj_other])
                        x1, y1, x2, y2 = self.objects[obj_other] 
                        if inter > self.calc_area(self.objects[obj_other])*0.8:  # obj의 80% 이상이 다른 cluster 영역과 겹치면 다른 상품으로 디텍팅
                            return_img = cv2.rectangle(return_img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                            return_boxes.append([x1, y1, x2, y2])

                        
                            black_img = np.zeros_like(input_img.copy())
                            clst_img = cv2.rectangle(black_img, (clstr_area[0][0], clstr_area[0][1]), (clstr_area[0][2], clstr_area[0][3 ]), (0, 0, 255), -1, cv2.LINE_AA)
                            obj_img = cv2.rectangle(black_img, (x1, y1), (x2, y2), (255, 0, 0), -1, cv2.LINE_AA)

                            blend = cv2.addWeighted(clst_img, 0.5, obj_img, 0.5, 0)
                            filename = f'blend_c{cluster}_with_c{other}.jpg'
                            cv2.imwrite(os.path.join(path, filename), blend)

        if is_test:
            cv2.imwrite(os.path.join(path, 'test_result.jpg'), cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(os.path.join(path, 'result.jpg'), cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB))
        return return_boxes

        

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



# # best-k 찾기
# sse = []
# list_k = list(range(3, 50))

# for k in list_k:
#     km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
#     km.fit(x)

#     sse.append(km.inertia_)

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance')
