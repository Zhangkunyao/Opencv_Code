#coding=utf-8
from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,get_muliti_bbox
import cv2
import matplotlib.pyplot as plt
import sys

path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/WSHP'
_,name_all = Get_List(path)
name_all.sort()
for i in name_all:
    img = cv2.imread(os.path.join(path,i))
    img = cv2.resize(img, (int(img.shape[1]/4),int(img.shape[0]/4)), interpolation=cv2.INTER_CUBIC)

    bbox_loc = get_muliti_bbox(img)
    cv2.imshow('a',cv2.rectangle(img, (bbox_loc['xmin']-5, bbox_loc['ymin']-5), (bbox_loc['xmax']+5, bbox_loc['ymax']+5), (0, 0, 255), 1))
    cv2.waitKey(1)