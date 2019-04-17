# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

def write_data(txt_path,data):
    file = open(txt_path, 'a')
    for tmp in data:
        file.write(str(tmp))
        file.write('\t')
    file.write('\n')
    file.close()

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
def img_process(img, org_size,pose):
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
    tmp = cv2.resize(roi_region, (w, h), interpolation=cv2.INTER_CUBIC)
    x,y = np.where(pose==0)
    tmp[x,y,1] = 255
    return tmp

def get_scale(source,source_h,source_w,target,target_h,target_w):
    # source = source[0:len(source)-100]
    source_all = []
    target_all = []
    for loc in source:
        point = {'xmin': loc[0], 'xmax': loc[1], 'ymin': loc[2], 'ymax': loc[3]}
        w = point['xmax'] - point['xmin']
        h = point['ymax'] - point['ymin']
        if w > 20 and h > 20:
            source_all.append([h/source_h,point['ymax']/source_h,(point['xmax']+point['xmin'])/2.0/source_w])
    for loc in target:
        point = {'xmin': loc[0], 'xmax': loc[1], 'ymin': loc[2], 'ymax': loc[3]}
        w = point['xmax'] - point['xmin']
        h = point['ymax'] - point['ymin']
        if w > 20 and h > 20:
            target_all.append([h/target_h,point['ymax']/target_h,(point['xmax']+point['xmin'])/2.0/target_w])

    target_all = np.array(target_all)*1.0
    target_data = [target_all[:,0].mean(),target_all[:,0].var(ddof=1),
                   target_all[:,1].mean(),target_all[:,1].var(ddof=1),
                   target_all[:, 2].mean(), target_all[:, 2].var(ddof=1)]
    source_all = np.array(source_all) * 1.0
    source_data = [source_all[:,0].mean(),source_all[:,0].var(ddof=1),
                   source_all[:,1].mean(),source_all[:,1].var(ddof=1),
                   source_all[:, 2].mean(), source_all[:, 2].var(ddof=1),]
    return target_data,source_data

source_root = '/media/kun/Dataset/Pose/test_data/'
target_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06'
print(source_root)
_,image_all = Get_List(os.path.join(source_root,'img'))
image_all.sort()

source_data_root = os.path.join(source_root,'DensePoseProcess')
source_org_path = os.path.join(source_data_root,'org')
save_path = os.path.join(source_data_root,'normal_loc.txt')
loc_all_source = get_all_loc(os.path.join(source_root,'DensePoseProcess','loc.txt'))
loc_all_target = get_all_loc(os.path.join(target_root,'DensePoseProcess','loc.txt'))

target_tmp_path = os.path.join(target_root,'back_ground.png')
target_pose_img = cv2.imread(target_tmp_path)
target_shape = target_pose_img.shape

img_name_now = 'a'
source_shape = cv2.imread(os.path.join(source_root,'img',image_all[0])).shape

target_scale,source_scale = get_scale(loc_all_source,source_shape[0]*1.0,source_shape[1]*1.0,
                                      loc_all_target,target_shape[0]*1.0,target_shape[1]*1.0)

source_h_mean = source_scale[0]
source_h_var = source_scale[1]
source_y_mean = source_scale[2]
source_y_var = source_scale[3]
source_x_mean = source_scale[4]
source_x_var = source_scale[5]

target_h_mean = target_scale[0]
target_h_var = target_scale[1]
target_y_mean = target_scale[2]
target_y_var = target_scale[3]
target_x_mean = target_scale[4]
target_x_var = target_scale[5]

# 主程序
for index in range(len(loc_all_source)):
    img_name = image_all[index]
    if img_name[0]!=img_name_now:
        source_shape = cv2.imread(os.path.join(source_root, 'img', img_name)).shape
        img_name_now = img_name[0]

    tmp = loc_all_source[index]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}

    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    y = point['ymax']
    x = (point['xmax'] + point['xmin'])/2.0
    if w < 10 or h < 10:
        result = [0,0,0,0]
    else:
        scale = target_shape[0]/source_shape[0]
        # 得出放大后的尺寸
        scale_w = int(w * scale)
        scale_h = int(h * scale)
        w_max = int(target_shape[1]*scale)
        # 计算位置
        x_pose = int((x/source_shape[1] - source_x_mean+0.5)*w_max)
        y_pose = int((y * 1.0 / source_shape[0] - source_y_mean + target_y_mean) * target_shape[0])

        xmin = int(x_pose - scale_w/2)
        xmax = int(x_pose + scale_w/2)
        if xmax-xmin<scale_w:
            xmax = xmax+1
        elif xmax-xmin>scale_w:
            xmax = xmax - 1
        # 计算偏差量
        delt_x_min = xmin - max(xmin, 0)
        delt_x_max = xmax - min(xmax, w_max)
        xmin = max(xmin, 0)
        xmax = min(xmax, w_max)

        ymin = max(y_pose - scale_h, 0)
        ymax = min(y_pose, target_shape[0])

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        # source坐标
        s_xmin = 0 + abs(delt_x_min)
        s_xmax = scale_w - abs(delt_x_max)

        s_middle = scale_h/2.0
        s_ymax = min(int(s_middle + (ymax - ymin)/2.0), scale_h)
        s_ymin = max(s_ymax - (ymax - ymin),0)

        delt = int((w_max-target_shape[1])/2)
        write_data(save_path,[scale_w,scale_h,xmin,xmax,ymin,ymax,s_xmin,s_xmax,s_ymin,s_ymax,delt,w_max])

    print(index / len(loc_all_source))
