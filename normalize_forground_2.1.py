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

target_tmp_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'
source_tmp_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'

source_path = "/home/kun/Documents/DataSet/video_06/forground/cut_expend"
save_path = "/home/kun/Documents/DataSet/video_06/forground/normal_pose"
data_root = '/home/kun/Documents/DataSet/video_06/forground/'

loc_all_source = get_all_loc("/home/kun/Documents/DataSet/video_06/forground/loc.txt")
loc_all_target = get_all_loc("/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/loc.txt")

_, source_imgs = Get_List(source_path)
source_imgs.sort()

source_pose_img = cv2.imread(source_tmp_path)
target_pose_img = cv2.imread(target_tmp_path)
target_shape = target_pose_img.shape
source_shape = source_pose_img.shape


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


fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(os.path.join(data_root,'normal.avi'), fourcc, fps, (target_shape[1],target_shape[0]))

# 主程序
for i, pose_name in enumerate(source_imgs):

    img_path = os.path.join(source_path, pose_name)
    source_pose_img = cv2.imread(img_path)  # input 256*256 img

    tmp = loc_all_source[i]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}

    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    # y = (point['ymax'] + point['ymin'])/2.0
    y = point['ymax']
    x = (point['xmax'] + point['xmin'])/2.0
    if w < 10 or h < 10:
        result = np.zeros(target_shape)
        # result[..., 1] = 255
    else:
        y_pose = ((y * 1.0 / source_shape[0] - source_y_mean) / (source_y_var) * target_y_var + target_y_mean) * \
                 target_shape[0]

        scale = target_h_mean/source_h_mean*(target_shape[0]/source_shape[0])

        # x_pose = ((x * 1.0 / source_shape[1] - source_x_mean) / (source_x_var) * target_x_var + target_x_mean) * \
        #          target_shape[1]
        x_pose = x/source_shape[1]*target_shape[1]
        # source_pose_img = img_process(source_pose_img, [w, h])
        source_pose_img = cv2.resize(source_pose_img, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_CUBIC)

        result = np.zeros(target_shape)
        # result[..., 1] = 255

        xmin = max(x_pose - source_pose_img.shape[1]/2.0, 0)
        xmax = min(x_pose + source_pose_img.shape[1]/2.0, target_shape[1])

        ymin = max(y_pose - source_pose_img.shape[0], 0)
        ymax = min(y_pose, target_shape[0])

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        # source坐标
        s_middle = int(source_pose_img.shape[1]/2)
        s_xmin = max(int(s_middle - (x_pose - xmin)),0)
        s_xmax = min(s_xmin+(xmax - xmin),source_pose_img.shape[1])

        s_middle = source_pose_img.shape[0]/2.0
        s_ymax = min(int(s_middle + (ymax - ymin)/2.0), source_pose_img.shape[0])
        s_ymin = max(s_ymax - (ymax - ymin),0)
        try:
            result[ymin:ymax, xmin:xmax, ...] = source_pose_img[
                                                s_ymin:s_ymax,
                                                s_xmin:s_xmax, ...]
        except:
            print("bad_img")
        result = result.astype(np.uint8)
    videoWriter.write(result.astype(np.uint8))
    cv2.imwrite(os.path.join(save_path, pose_name), result, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(i / len(source_imgs))
    # result = cv2.resize(result, (int(target_shape[1]/2), int(target_shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a',result)
    # key = cv2.waitKey(1)
    # if key == ord(" "):
    #     print(pose_name)
    #     cv2.waitKey(0)
    # if key == ord("q"):
    #     break
videoWriter.release()