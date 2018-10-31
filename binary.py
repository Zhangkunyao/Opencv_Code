'''
简单阈值
像素值高于阈值时 我们给这个像素 赋予一个新值， 可能是白色 ，
 否则我们给它赋予另外一种颜色， 或是黑色 。
 这个函数就是 cv2.threshhold()。
 这个函数的第一个参数就是原图像
 原图像应 是灰度图。
 第二个参数就是用来对像素值进行分类的阈值。
 第三个参数 就是当像素值高于， 有时是小于  阈值时应该被赋予的新的像素值。
 OpenCV 提供了多种不同的阈值方法 ， 是由第四个参数来决定的。
  些方法包括
• cv2.THRESH_BINARY
• cv2.THRESH_BINARY_INV • cv2.THRESH_TRUNC
• cv2.THRESH_TOZERO
• cv2.THRESH_TOZERO_INV
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from basic_lib import Get_List,mkdir

path_root = '/media/kun/Dataset/Pose/openpose/tmp/'
dir_name,_ = Get_List(path_root)

for i in dir_name:
    img_path_root = os.path.join(path_root, i)
    binary_map_root = os.path.join(img_path_root, 'binary')
    mkdir(binary_map_root)
    img_path_root = os.path.join(img_path_root, 'sal')
    _,sal_names = Get_List(img_path_root)
    print(i)
    for j in sal_names:
        sal_name = os.path.join(img_path_root, j)
        binary_name = os.path.join(binary_map_root, j)
        img = cv2.imread(sal_name, 0)
        _, thresh3 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(binary_name, thresh3)
print('finished')