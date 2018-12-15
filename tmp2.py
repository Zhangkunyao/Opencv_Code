from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,get_bbox
import random
import cv2

def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i))
        file.write('\t')
    file.write('\n')
    file.close()

def get_all_loc(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

def refresh(lis,data):
    for i in range(len(lis)-1):
        lis[len(lis)-i-1] = lis[len(lis)-i-2]
    lis[0] = data
    return lis

name_root = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/img'
map_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePose'
img_root = '/home/kun/Documents/DataSet/video_06/img'
img_save_root = '/home/kun/Documents/DataSet/video_06/cut/img'
txt_save_root = '/home/kun/Documents/DataSet/video_06/cut/loc.txt'
txt_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/loc.txt'
loc_all_source = get_all_loc(txt_root)
kernel = np.ones((5,5),np.uint8)
x_min_filter = [loc_all_source[0][0] for i in range(10)]
x_max_filter = [loc_all_source[0][1] for i in range(10)]
y_min_filter = [loc_all_source[0][2] for i in range(10)]
y_max_filter = [loc_all_source[0][3] for i in range(10)]

_,img_names = Get_List(name_root)
img_names.sort()
for i in range(len(img_names)):
    map_name =img_names[i][:-4]+'_IUV.png'
    img_path = os.path.join(img_root, img_names[i])
    img = cv2.imread(img_path)
    map_path = os.path.join(map_root, map_name)
    map = cv2.imread(map_path)

    map = map[...,0] + map[...,1] + map[...,2]
    map = map[...,np.newaxis]
    map = np.repeat(map,3,2)
    map[map > 0] = 255
    # wshp[wshp < 0] = 0
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel)
    zero_idx = map == 0
    img[zero_idx[...,0],0] = 0
    img[zero_idx[..., 1], 1] = 255
    img[zero_idx[..., 2], 2] = 0

    tmp = loc_all_source[i]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}
    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    xmin = point['xmin'] - w*0.1
    xmax = point['xmax'] + w*0.1
    ymin = point['ymin'] - h*0.05
    ymax = point['ymax'] + h*0.05

    x_min_filter = refresh(x_min_filter, xmin)
    x_max_filter = refresh(x_max_filter, xmax)
    y_min_filter = refresh(y_min_filter, ymin)
    y_max_filter = refresh(y_max_filter, ymax)

    xmin = max(int(np.array(x_min_filter).mean()),0)
    xmax = min(int(np.array(x_max_filter).mean()),img.shape[1])
    ymin = max(int(np.array(y_min_filter).mean()),0)
    ymax = min(int(np.array(y_max_filter).mean()),img.shape[0])

    # text_save(txt_save_root, [xmin, xmax, ymin, ymax])
    # cv2.imwrite(os.path.join(img_save_root, img_names[i]), img[ymin:ymax, xmin:xmax, ...],
    #             [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if xmax>xmin and ymax>ymin:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    cv2.imshow('1', img)
    cv2.waitKey(1)
    print(i/len(img_names))
print('finished')
