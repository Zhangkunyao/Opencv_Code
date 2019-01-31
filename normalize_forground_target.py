# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

# 这个版本主要解决scale上的变化问题，通过求解 y 与 h之间的关系，解出scale
# 把跑出来的测试结果重新resize 用于前景背景之间的融合
def get_all_loc(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

# 输入图片 目标尺寸 原始尺寸
def img_process(img, org_size):
    roi_region = None
    loadsize = img.shape[0]
    w = org_size[0]
    h = org_size[1]
    if h >= w:
        w = int(w * loadsize / h)
        h = loadsize
        bias = int((loadsize - w) / 2)
        img = np.array(img)
        roi_region = img[0:h, bias:bias + w, ...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        bias = int((loadsize - h) / 2)
        img = np.array(img)
        roi_region = img[bias:bias + h, 0:w, ...]
    w = org_size[0]
    h = org_size[1]
    return cv2.resize(roi_region, (w, h), interpolation=cv2.INTER_CUBIC)


target_tmp_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'

target_path = "/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/save_result"
save_path = "/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/normal_result"
data_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/'

loc_all_target = get_all_loc("/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/loc.txt")

_, target_imgs = Get_List(target_path)
target_imgs.sort()

target_pose_img = cv2.imread(target_tmp_path)
target_shape = target_pose_img.shape

# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter(os.path.join(data_root,'normal.avi'), fourcc, fps, (target_shape[1],target_shape[0]))

# 主程序
for i, pose_name in enumerate(target_imgs):
    img_path = os.path.join(target_path, pose_name)
    source_pose_img = cv2.imread(img_path)  # input 256*256 img

    tmp = loc_all_target[i]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}

    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    if w < 10 or h < 10:
        result = np.zeros(target_shape)
        result[..., 1] = 255
    else:
        result = np.zeros(target_shape)
        result[..., 1] = 255
        source_pose_img = img_process(source_pose_img, [w, h])
        ymin = point['ymin']
        ymax = point['ymax']
        xmin = point['xmin']
        xmax = point['xmax']
        result[ymin:ymax, xmin:xmax, ...] = source_pose_img

    cv2.imwrite(os.path.join(save_path, pose_name), result, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(i / len(target_imgs))
    # result = cv2.resize(result, (int(target_shape[1]/2), int(target_shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a',result)
    # key = cv2.waitKey(1)
    # if key == ord(" "):
    #     print(pose_name)
    #     cv2.waitKey(0)
    # if key == ord("q"):
    #     break
# videoWriter.release()