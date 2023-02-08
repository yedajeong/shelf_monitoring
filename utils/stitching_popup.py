from pkgutil import read_code
import cv2
from cv2 import resize
from cv2 import INTER_CUBIC
import numpy as np
import time

def rotate(src, angle):
	height, width, _ = src.shape
	matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
	dst = cv2.warpAffine(src, matrix, (width, height))
	return dst

def mouse_event(event, x, y, flags, param):
	global mouse_points
	if event == cv2.EVENT_FLAG_LBUTTON:
		print(f'{x},{y}')




cam = cv2.imread('/Users/dajeong/Desktop/Yennie/lab/gs_raw/triplet_1674035307.257064.jpg')
# cam = cam[27:,:1796]

# cam = cv2.copyMakeBorder(cam, 100,100,100,100, cv2.BORDER_CONSTANT, value=[0,0,0])
# cam = cv2.resize(cam, None, fx=0.6, fy=0.6)
h, w, _ = cam.shape
before = np.float32([[0,0], [0,h], [w-150, 150], [w-150, h-150]])
after = np.float32([[0,0], [0,h], [w,0], [w,h]])
# mtrx = cv2.getPerspectiveTransform(before, after)
# cam =  cv2.warpPerspective(cam, mtrx, (w, h))
# cam = rotate(cam, 5)

# cv2.rectangle(cam, (330,100), (680,170), (0,0, 255), 3) #real
# # cv2.rectangle(cam, (345,340), (940,380), (255,0, 0), 3) #intersection
# cv2.rectangle(cam, (330,100), (680,270), (0,255, 0), 3) #in

cv2.imshow('mix', cam)
cv2.setMouseCallback('mix', mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
