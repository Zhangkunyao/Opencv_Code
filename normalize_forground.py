# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List,get_bbox
# 把跑出来的测试结果重新resize 用于前景背景之间的融合
def get_all_loc(file_path):
    file = open(file_path,'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

# 输入图片 目标尺寸 原始尺寸
#     source_pose_img = cv2.resize(source_pose_img, (int(scale * w),
#                                      int(scale * h)),
#                           interpolation=cv2.INTER_CUBIC)
def img_process(img,org_size):
    roi_region = None
    loadsize = img.shape[0]
    w = org_size[0]
    h = org_size[1]
    if h >= w:
        w = int(w*loadsize/h)
        h = loadsize
        bias = int((loadsize - w)/2)
        img = np.array(img)
        roi_region = img[0:h,bias:bias+w,...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        bias = int((loadsize - h)/2)
        img = np.array(img)
        roi_region = img[bias:bias+h,0:w,...]
    w = org_size[0]
    h = org_size[1]
    return cv2.resize(roi_region, (w,h),interpolation=cv2.INTER_CUBIC)

# target video_06
target_pose = {}
target_pose['y_min'] = 807.900398
target_pose['y_middle'] = 944.796877
target_pose['y_max'] = 973.400870
target_pose['h_close'] = 853.156591
target_pose['h_far'] = 804.780441
target_pose['h_middle'] = 790.138627
target_pose['bias'] = 103
# source video_27
# source_pose = {}
# source_pose['y_min'] = 765.709552
# source_pose['y_middle'] = 936.218109
# source_pose['y_max'] = 979.515083
# source_pose['h_close'] = 746.738775
# source_pose['h_far'] = 450.751785
# source_pose['h_middle'] = 685.977859
# source_pose['bias'] = 85
source_pose = {}
source_pose['y_min'] = 515.840048
source_pose['y_middle'] = 592.495087
source_pose['y_max'] = 625.104940
source_pose['h_close'] = 437.892544
source_pose['h_far'] = 235.639149
source_pose['h_middle'] = 382.719857
source_pose['bias'] = 103
# video_27 50
# video_06 103
# scale = target_pose['h_far']/source_pose['h_far'] + (source_pose['y_middle'] - source_pose['y_min'])/(source_pose['y_max'] - source_pose['y_min'])*(target_pose['h_close']/source_pose['h_close'] - target_pose['h_far']/source_pose['h_far'])
scale = (target_pose['h_middle']/source_pose['h_middle'])

loc_all = get_all_loc("/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/cut/loc.txt")

target_tmp_path = '/home/kun/Documents/DataSet/video_06/back_ground.png'
source_tmp_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/back_ground.png'

source_path = "/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/cut/save_result"
save_path = "/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/cut/normal_result"

_,source_imgs = Get_List(source_path)
source_imgs.sort()

source_pose_img = cv2.imread(source_tmp_path)
target_pose_img = cv2.imread(target_tmp_path)
target_shape = target_pose_img.shape
source_shape = source_pose_img.shape

point_loc = []

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('tmp.avi', fourcc, fps, (target_shape[1],target_shape[0]))

for i,pose_name in enumerate(source_imgs):

    img_path = os.path.join(source_path,pose_name)
    source_pose_img = cv2.imread(img_path) #input 256*256 img

    tmp = loc_all[i]
    point = {'xmin':tmp[0],'xmax':tmp[1],'ymin':tmp[2],'ymax':tmp[3]}

    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    if w < 4 or h<4:
        result = np.zeros(target_shape)
        result[..., 1] = 255
    else:
        source_pose_img = img_process(source_pose_img,[w,h])
        source_pose_img = cv2.resize(source_pose_img,(int(scale*w),int(scale * h)),interpolation=cv2.INTER_CUBIC)
        result = np.zeros(target_shape)
        result[..., 1] = 255


        if (point['ymax']-source_pose['bias']) <= source_pose['y_middle']:
            y_pose = target_pose['y_min'] + (target_pose['y_middle'] - target_pose['y_min']) / \
                 (source_pose['y_middle'] - source_pose['y_min']) * (point['ymax']-source_pose['bias'] - source_pose['y_min'])
        else:
            y_pose = target_pose['y_middle'] + (target_pose['y_max'] - target_pose['y_middle']) / \
                 (source_pose['y_max'] - source_pose['y_middle']) * (point['ymax']-source_pose['bias'] - source_pose['y_middle'])
        # y_pose = bias = target_pose['y_min'] + (source_pose['y_middle'] - source_pose['y_min']) / \
        #      (source_pose['y_max'] - source_pose['y_min']) * (target_pose['y_max'] - target_pose['y_min'])-(point['ymax'] - 50)

        y_pose = int(y_pose + target_pose['bias']+100)

        x_pose = int(point['xmin']/source_shape[1]*target_shape[1] - (scale*w - w)/2)
        result = np.zeros(target_shape)
        result[...,1] = 255
        xmin = max(x_pose,0)
        ymin = max(y_pose - source_pose_img.shape[0],0)
        xmax = min(xmin + source_pose_img.shape[1],target_shape[1])
        ymax = min(y_pose,target_shape[0])
        # try:
        result[ymin:ymax,xmin:xmax,...] = source_pose_img[source_pose_img.shape[0]-(ymax - ymin):source_pose_img.shape[0],
                                          source_pose_img.shape[1] - (xmax-xmin):source_pose_img.shape[1],...]
        # except:
        #     print(xmin)
        #     print(ymin)
        #     print(xmax)
        #     print(ymax)
        #     print(pose_name)
        #     while 1:
        #         continue
    cv2.imwrite(os.path.join(save_path, pose_name), result,[int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(i/len(source_imgs))