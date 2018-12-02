# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List,get_bbox
import matplotlib.pyplot as plt
# 提取前景，并做剪切

def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i))
        file.write('\t')
    file.write('\n')
    file.close()

img_root = "/media/kun/Dataset/Pose/DataSet/new_data/0001/img"
pose_root = "/media/kun/Dataset/Pose/DataSet/new_data/0001/WSHP"
save_root = "/media/kun/Dataset/Pose/DataSet/new_data/0001/WSHP_Cut"
txt_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001/WSHP_Cut/loc.txt'

_,img_list = Get_List(img_root)
img_list.sort()

for index,name in enumerate(img_list):
    img_path = os.path.join(img_root,name)
    # sal_path = os.path.join(sal_root, name)
    pose_path = os.path.join(pose_root, name)

    pose_save_path = os.path.join(save_root, 'pose')
    pose_save_path = os.path.join(pose_save_path,name)
    img_save_path = os.path.join(save_root, 'img')
    img_save_path = os.path.join(img_save_path, name)

    img = cv2.imread(img_path)
    sal = cv2.imread(pose_path)
    pose = cv2.imread(pose_path)

    sal = sal[...,0] + sal[...,1] + sal[...,2]
    sal = sal[...,np.newaxis]
    sal = np.repeat(sal,3,2)
    sal[sal > 10] = 255
    sal[sal < 10] = 0
    bbox = get_bbox(sal)
    out = cv2.bitwise_and(img,sal)
    zero_idx = sal == 0
    out[zero_idx[...,0],0] = 0
    out[zero_idx[..., 1], 1] = 255
    out[zero_idx[..., 2], 2] = 0
    if bbox['xmax'] > bbox['xmin'] and bbox['ymax'] > bbox['ymin']:
        out = out[bbox['ymin']:bbox['ymax'],bbox['xmin']:bbox['xmax'],...]
        pose = pose[bbox['ymin']:bbox['ymax'],bbox['xmin']:bbox['xmax'],...]
    text_save(txt_path, [bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']])
    cv2.imwrite(img_save_path,out,[int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    cv2.imwrite(pose_save_path, pose, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    # videoWriter.write(out)
    print(index)
# videoWriter.release()
    # cv2.imshow('tmp',out)
    # cv2.waitKey(1)

# back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2YCR_CB)
# fps = 20
# img_size = (back_img.shape[1], back_img.shape[0])
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('./CRF.avi', fourcc, fps, img_size)