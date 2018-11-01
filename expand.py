# coding=utf-8
from basic_lib import Get_List,mkdir
import cv2
import os
import numpy as np

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 矩形结构
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))  # 十字形结构


path_root = "../data/source_pose"
save_path = '../data/result'
_,img_names = Get_List(path_root)
img_names.sort()

for i in range(len(img_names)):
    img = cv2.imread(os.path.join(path_root, img_names[i]))
    save_name = os.path.join(save_path, img_names[i])
    img = img * 1.5
    # kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(img, kernel)
    cv2.imwrite(save_name,dilation)
    print(1.0*i/len(img_names))