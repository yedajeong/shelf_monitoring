import time
import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from pathlib import Path
import os


labels_to_names = {0: 'product'}
script_path = os.path.dirname(os.path.realpath(__file__))

class Detector(object):

	def __init__(self):
		self._area_th = 200
		self.model = models.load_model(os.path.join(script_path, 'resnet101_pascal_0207.h5'), backbone_name='resnet101')
		self.model = models.convert_model(self.model)

	@property
	def area_th(self):
		return self._area_th

	@area_th.setter
	def area_th(self, value):
		self._area_th = value


	def detect_image(self, image, store_id=0, port=0, category_id=0, is_test=False, is_train=False):
		rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		float_image = preprocess_image(rgb_image)
		final_image, scale = resize_image(float_image)

		start = time.time()
		boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(final_image, axis=0))
		boxes /= scale

		boxes = np.squeeze(boxes)
		scores = np.squeeze(scores)
		labels = np.squeeze(labels)

		# print("processing time: ", time.time() - start)
		result_boxes = list()
		result_labels = list()
		for idx, score in enumerate(scores):
			if score >= 0.4:  # 0.2 -> 0.4 (23/2/3)
				#confidence threshold 이상만 후보로 추출
				result_boxes.append(boxes[idx])
				result_labels.append(labels[idx])

		#nms 50% 이상 겹치면 merge
		# nms 80% (23/2/3)
		result_boxes = self.non_max_suppression_fast(np.asarray(result_boxes), .8)

		if is_train:
			return_results = []
			for box, label in zip(result_boxes, result_labels):
				b = np.array(box).astype(int)
				return_results.append([label, b])

			return return_results

		else:
			if is_test:
				test_img = image.copy()
				for box in result_boxes:
					b = np.array(box).astype(int)
					cv2.rectangle(test_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2, cv2.LINE_AA)
				cv2.imwrite('test.jpg', test_img)
			else:
				#후 반영이 맞다
				temp = time.time()
				empty_results = self.calc(image, result_boxes, store_id, port, category_id)
				self.make_ref(image, result_boxes, store_id, port, category_id)
				# print("calc time: ", time.time()-temp)

				return empty_results

	def make_ref(self, image, boxes, store_id, port, category_id):
		p = Path('static/image/process/{}/{}/{}'.format(store_id, port, category_id))
		if False == p.exists():
			p.mkdir(parents=True)

		ref_img = cv2.imread('static/image/process/{}/{}/{}/ref.jpg'.format(store_id, port, category_id))
		if ref_img is None:
			# print('ref none')
			ref_img = np.zeros_like(image)
		old_ref = ref_img.copy()

		detect_img = cv2.imread('static/image/process/{}/{}/{}/detect.jpg'.format(store_id, port, category_id))
		if detect_img is None:
			detect_img = np.zeros_like(image)

		for box in boxes:
			#reference 이미지와 detect image 의 bbox 내 상품들을 전부 merge 한다
			b = np.array(box).astype(int)
			ref_img[b[1]:b[3], b[0]:b[2]] = image[b[1]:b[3], b[0]:b[2]]  # input image에서 디텍팅된 bbox 부분 크롭해서 ref_img의 같은 영역에 붙여넣기
			#detect 이미지의 bbox 를 빨간색으로 칠한다
			detect_img = cv2.rectangle(detect_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), -1, cv2.LINE_AA)  # detect_img에 디텍팅된 bbox 색칠

		# cv2.imwrite('{}/old_ref.jpg'.format(cam_id), old_ref)
		cv2.imwrite('static/image/process/{}/{}/{}/ref.jpg'.format(store_id, port, category_id), ref_img)
		ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
		cv2.imwrite('static/image/process/{}/{}/{}/ref_rgb.jpg'.format(store_id, port, category_id), ref_rgb)
		cv2.imwrite('static/image/process/{}/{}/{}/detect.jpg'.format(store_id, port, category_id), detect_img)

	def calc(self, image, boxes, store_id, port, category_id):
		ref_img = cv2.imread('static/image/process/{}/{}/{}/ref.jpg'.format(store_id, port, category_id))
		if ref_img is None:
			#reference 이미지가 없으면 종료
			return []
		
		#이전 턴에 상품이 인식된(빨간색으로 칠해진)이미지 로드
		detect_img = cv2.imread('static/image/process/{}/{}/{}/detect.jpg'.format(store_id, port, category_id))
		blue = np.zeros_like(image)
		my_img = image.copy()

		input_w = image.shape[0]
		input_h = image.shape[1]

		for box in boxes:
			b = np.array(box).astype(int)
			# my_img[b[1]:b[3], b[0]:b[2]] = image[b[1]:b[3], b[0]:b[2]]

			# my.jpg에 디텍팅된 obj bbox 치기
			cv2.rectangle(my_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2, cv2.LINE_AA)

			# --------------bbox 크기 수정 (yennie)--------------
			# bbox 칠 때 실제 크기보다 크게 해서 ㄱ,ㄴ 모양 억제
			# 빈공간 bbox 칠 때도 실제 윤곽선(흰부분)보다 키워주기
			# 고정 픽셀(delta)만큼 일괄적으로 키움
			tmp_delta = 2
			# out of range일 경우 좌표 계산 추가 (수정)
			# b0 = max(0, b[0]-tmp_delta)  # xmin
			# b1 = max(0, b[1]-tmp_delta)  # ymin
			# b2 = min(input_w, b[2]+tmp_delta)  # xmax
			# b3 = min(input_h, b[3]+tmp_delta)  # ymax
			b0 = b[0]-tmp_delta  # xmin
			b1 = b[1]-tmp_delta  # ymin
			b2 = b[2]+tmp_delta  # xmax
			b3 = b[3]+tmp_delta  # ymax
			# ---------------------------------------------------

			#이번턴에 인식된 상품은 파란색으로 칠한다
			# cv2.rectangle(blue, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), -1, cv2.LINE_AA)
			cv2.rectangle(blue, (b0, b1), (b2, b3), (255, 0, 0), -1, cv2.LINE_AA)

		#이전 턴에 상품이 인식된 이미지와 이번 턴에 상품이 인식된 이미지를 5:5 비율로 블렌딩한다
		blend = cv2.addWeighted(detect_img, .5, blue, .5, 0)

		#빨간색만 필터링해서 binary 이미지로 변경하고 contour 추출
		red_mask = cv2.inRange(blend, np.array([0, 0, 125]), np.array([0, 0, 225]))
		red = np.zeros_like(image)
		red[red_mask > 0] = (255, 255, 255)
		red_bin = cv2.cvtColor(red.copy(), cv2.COLOR_BGR2GRAY)
		red_bin = cv2.erode(red_bin, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))  # 흰색 부분 깎아냄, 커널 크기 (3, 3)
		cv2.imwrite('static/image/process/{}/{}/{}/red_bin.jpg'.format(store_id, port, category_id), red_bin)

		_, red_bin = cv2.threshold(red_bin, 10, 255, cv2.THRESH_BINARY)
		cv2.imwrite('static/image/process/{}/{}/{}/red_bin_threshold.jpg'.format(store_id, port, category_id), red_bin)

		cnts, _ = cv2.findContours(red_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		result = image.copy()
		diff = np.zeros_like(image)
		fast = cv2.FastFeatureDetector_create()
		result_boxes = list()
		# print(self._area_th)
		

		for idx, cnt in enumerate(cnts):
			cnt_area = cv2.contourArea(cnt)  # contour 면적 계산
			x, y, w, h = cv2.boundingRect(cnt)  # contour 둘러싸는 rect 정보 (lu x, y좌표)
			#이전턴과 이번턴의 다른 부분만 추출
			diff[y:y + h, x:x + w] = ref_img[y:y + h, x:x + w]
			
			# --------------조건 수정 (yennie)--------------
			#너무 얇은것들 컷 (세로 대비 가로길이가 너무 짧거나 길면 길쭉한 모양 ex)진열대 프레임)
			ratio = w/h
			# if ratio < 0.2 or ratio > 5: continue
			if ratio > 3.5 or ratio < 0.15:
			# 	cv2.putText(diff, 'ratio {:.2f}'.format(ratio), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
			# 				color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
				continue

			#일정 크기 이하의 영역은 패스
			if cnt_area < self._area_th:
				cv2.putText(diff, 'area {:.2f}'.format(cnt_area), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
							color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
				continue
			
			# ㄱ, ㄴ .. -> 진열대 프레임
			if cnt_area < w*h*0.4:
				continue
			
			# 일정 크기 이상인 경우도 패스 (input image의 0.5배 이상)
			_area_th_upper = input_w * input_h * 0.5 
			if cnt_area > _area_th_upper:
				continue
			# ------------------------------------------

			#ㄱ,ㄴ 컷
			# cv2.drawContours(result, cnts, idx, (255, 255, 255), 2)
			# cv2.putText(result, str(cnt_area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
			# hull = cv2.convexHull(cnt, returnPoints=False)
			# defects = cv2.convexityDefects(cnt, hull)
			# fars = list()
			# if defects is not None:
			# 	for i in range(defects.shape[0]):
			# 		s, e, f, d = defects[i, 0]
			# 		far = tuple(cnt[f][0])
			# 		fars.append(far)
			#
			# 	if len(fars) < 2:
			# 		pts = [tuple(list(c)) for c in np.squeeze(cnt)]
			# 		poly = Polygon(pts)
			# 		splitter = LineString(([0, 0], list(fars[0])))
			# 		sp_result = split(poly, splitter)
			# 		ratio_results = [False, False]
			# 		for i, rp in enumerate(sp_result):
			# 			rp_box = rp.bounds
			# 			rp_w = rp_box[2] - rp_box[0]
			# 			rp_h = rp_box[3] - rp_box[1]
			# 			rp_ratio = rp_w/rp_h
			# 			print(rp_ratio)
			# 			cv2.rectangle(result, (int(rp_box[0]), int(rp_box[1])), (int(rp_box[2]), int(rp_box[3])), (255, 0, 0), 2)
			# 			if rp_ratio > 0.2 and rp_ratio < 5: ratio_results[i] = True
			#
			# 		if not(ratio_results[0] and ratio_results[1]): continue
			candidate_gray = cv2.cvtColor(image[y:y+h, x:x+w].copy(), cv2.COLOR_BGR2GRAY)
			kps = fast.detect(candidate_gray, None)

			# -------------kps_thresh 수정 (yennie)--------------
			if len(kps) > 90:
			# if len(kps) > 110:
				#키포인트가 많으면 상품이 있는것(상품을 bbox로 detect 못했을 경우 예외처리)
				# cv2.putText(diff, 'kps {:d}'.format(len(kps)), (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255),
							# thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
				continue


			# cv2.putText(result, '{:.2f}/{}'.format(ratio, int(cnt_area)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255),
			# 			thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

			# -------------외곽선 영역 bbox 크기 수정 (yennie)--------------
			# out of range일 경우 좌표 계산 추가 (수정)
			# x1 = max(0, x-tmp_delta)  # detected bbox 크기 키운만큼 외곽선 bbox 크기도 키우기
			# y1 = max(0, y-tmp_delta)
			# x2 = min(input_w, x+tmp_delta)
			# y2 = min(input_h, y+tmp_delta)
			x1 = x-tmp_delta
			y1 = y-tmp_delta
			x2 = x+w+tmp_delta
			y2 = y+h+tmp_delta
			# -----------------------------------------------------------
			result = cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)  # input image(result에 복사)에 외곽선 영역(빈공간)의 rect만큼 bbox 치기
			result_boxes.append([x1, y1, x2, y2])
			# result = cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)  # input image(result에 복사)에 외곽선 영역(빈공간)의 rect만큼 bbox 치기
			# result_boxes.append([x, y, x+w, y+h])

		cv2.imwrite('static/image/process/{}/{}/{}/result.jpg'.format(store_id, port, category_id), result)
		cv2.imwrite('static/image/process/{}/{}/{}/diff.jpg'.format(store_id, port, category_id), diff)
		cv2.imwrite('static/image/process/{}/{}/{}/my.jpg'.format(store_id, port, category_id), my_img)
		cv2.imwrite('static/image/process/{}/{}/{}/blend.jpg'.format(store_id, port, category_id), blend)
		return result_boxes  # 결품 발생한 빈공간에 rect쳐서 좌표 정보 반환


	def non_max_suppression_fast(self, boxes, overlapThresh):
		# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []

		# if the bounding boxes integers, convert them to floats --
		# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")

		# initialize the list of picked indexes
		pick = []

		# grab the coordinates of the bounding boxes
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		# compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)

		# keep looping while some indexes still remain in the indexes
		# list
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

if __name__ == '__main__':
	d = Detector()
	img = cv2.imread('server/backend/static/image/upload/2/0/127/2_0_127_2023_02_03_08_45_27.jpg')
	d.detect_image(img, is_test=True)
	# for i in range(1, 9):
	# 	print(i)
	# 	for j in range(1, 3):
	# 		img = cv2.imread('images/{}_{}.jpg'.format(i,j))
	# 		d.detect_image(img, cam_id=i)


'''
1. ref: 현재턴에 디텍팅 Bbox부분을 input image에서 크롭해서 이전턴 ref 이미지 내의 같은 영역에 붙여넣음 (ref.jpg로 저장)
2. detect: 검정 바탕에 현재턴 디텍팅 Bbox부분만 빨간색으로 색칠
3. blend: detect의 이전턴에 디텍팅된 bbox 빨강 + 현재턴에 디텍팅된 Bbox 파랑 = 변화 없는 부분은 보라색
4. red_bin: blend에서 빨간 부분 (이전턴에는 인식됐지만 현재턴에 없는 == 결품인 공간) 만 흰색으로 & erode로 침식
5, diff: ref와 현재턴 다른 부분을 frame에서 crop해옴

- red_bin에서 흰색 부분 외곽선 추출
	너무 얇은 부분 -> 상품 있던 공간 아니라고 판단, pass ex)진열대 프레임
	area_th보다 면적 작은 공간은 pass

- fast: 코너 검출 알고리즘 -> len(kps)>110 이면 키포인트_모서리가 많은것 == 해당 영역을 흰색(빈공간)으로 판단했지만 실제로 상품이 있다고 판단, input image에 표시 안함
'''
