import glob
import os


img_list = glob.glob('static/image/upload/1/555/84/*.jpg')
img_list = sorted(img_list)
print(img_list[:20])

for img in img_list:
    os.remove(img)
