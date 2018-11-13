# coding=utf-8
from basic_lib import Get_List,mkdir
import cv2
import os
import numpy as np

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 矩形结构
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))  # 十字形结构


path_root = "../data/pose"
save_path = '../data/result'
_,img_names = Get_List(path_root)
img_names.sort()

for i in range(len(img_names)):
    img = cv2.imread(os.path.join(path_root, img_names[i]))
    save_name = os.path.join(save_path, img_names[i])
    img = img*1.5
    cv2.imwrite(save_name, img)
    img = cv2.imread(save_name)

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV)
    Lowerwhite = np.array([0, 0, 100])
    Upperwhite = np.array([180, 200, 255])
    mask = cv2.inRange(HSV, Lowerwhite, Upperwhite) # 提取彩色部分
    mask = mask[..., np.newaxis]
    mask = np.repeat(mask,3,axis=2)

    WhiteThings = cv2.bitwise_and(img, mask)
    Clour = cv2.bitwise_and(img, 255 - mask)
    Clour = cv2.dilate(Clour, kernel)
    Clour = cv2.bitwise_and(Clour, 255 - mask)

    Clour[...,0] = Clour[...,0] + WhiteThings[...,0]
    Clour[..., 1] = Clour[..., 1] + WhiteThings[..., 1]
    Clour[..., 2] = Clour[..., 2] + WhiteThings[..., 2]
    # dilation = Clour+WhiteThings
    # cv2.imshow('1',Clour)
    # cv2.waitKey(0)
    #
    cv2.imwrite(save_name,Clour)
    print(1.0*i/len(img_names))

