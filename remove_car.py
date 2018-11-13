# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List
import matplotlib.pyplot as plt

sal_root = "/home/kun/Documents/DataSet/video_06/CRF"
img_root = "/home/kun/Documents/DataSet/video_06/img"
back_ground = "/home/kun/Documents/DataSet/video_06/back_ground.png"
save_root = "/home/kun/Documents/DataSet/video_06/person"
# _,sal_list = Get_List(sal_root)
_,img_list = Get_List(img_root)
img_list.sort()
back_img = cv2.imread(back_ground)
back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2YCR_CB)

fps = 20
img_size = (back_img.shape[1], back_img.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('./CRF.avi', fourcc, fps, img_size)

for index,name in enumerate(img_list):
    img_path = os.path.join(img_root,name)
    sal_path = os.path.join(sal_root, name)
    save_path = os.path.join(save_root, name)
    img = cv2.imread(img_path)
    sal = cv2.imread(sal_path)
    sal = sal[...,0] + sal[...,1] + sal[...,2]
    sal = sal[...,np.newaxis]
    sal = np.repeat(sal,3,2)
    sal[sal > 10] = 255
    sal[sal < 10] = 0
    out = cv2.bitwise_and(img,sal)
    zero_idx = out == 0
    out[zero_idx[...,0],0] = 0
    out[zero_idx[..., 1], 1] = 255
    out[zero_idx[..., 2], 2] = 0
    # cv2.imshow('tmp',out)
    # cv2.waitKey(1)
    cv2.imwrite(save_path,out)
    videoWriter.write(out)
    print(index)
videoWriter.release()