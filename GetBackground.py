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

def get_kmeans(data,n_clusters=3):
    # 返回x轴和y轴的聚类坐标
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    # x 部分
    consit = np.array([10 for i in range(len(x))])
    x = np.array(x)
    point_loc = np.array([consit, x])
    point_loc = np.transpose(point_loc)
    kmeans_cell = KMeans(n_clusters=n_clusters, random_state=9)
    y_pred = kmeans_cell.fit_predict(point_loc)
    plt.scatter(x, y, c=y_pred)
    plt.show()
    print(kmeans_cell.cluster_centers_)
    x_result = kmeans_cell.cluster_centers_
    # y 部分
    consit = np.array([10 for i in range(len(y))])
    y = np.array(y)
    point_loc = np.array([consit, y])
    point_loc = np.transpose(point_loc)
    kmeans_cell = KMeans(n_clusters=n_clusters, random_state=9)
    y_pred = kmeans_cell.fit_predict(point_loc)
    plt.scatter(x, y, c=y_pred)
    plt.show()
    print(kmeans_cell.cluster_centers_)
    y_result = kmeans_cell.cluster_centers_
    return [x_result,y_result]

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

img_root = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/img'
save_img = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/img/back_ground.png'
loc_root = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/DensePoseProcess/loc.txt'
print(img_root)

loc_all = get_all_loc(loc_root)
_, all_imgs = Get_List(img_root)
all_imgs.sort()

img = cv2.imread(os.path.join(img_root, all_imgs[0]))
background = np.zeros(img.shape).astype(np.uint8)
# 主程序
for index, name in enumerate(all_imgs):
    img_path = os.path.join(img_root, name)
    img = cv2.imread(img_path)  # input 256*256 img

    tmp = loc_all[index]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}

    xmin = int(point['xmin'])
    xmax = int(point['xmax'])
    ymin = int(point['ymin'])
    ymax = int(point['ymax'])

    img[ymin:ymax, xmin:xmax, ...] = 0

    x,y = np.where((background[...,0]+background[...,1]+background[...,2]) == 0)
    # np.where((head_part[..., 0] + head_part[..., 1] + head_part[..., 2]) != 0)
    background[x,y,...] = img[x,y,...]
    # cv2.imshow('a',background)
    # cv2.waitKey(1)
    # cv2.imwrite(os.path.join(save_path, pose_name), result, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(index / len(all_imgs))
cv2.imwrite(save_img, background, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
