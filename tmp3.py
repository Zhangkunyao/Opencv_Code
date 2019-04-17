# -*- coding: utf-8 -*-

"""
two_camera.py:
"""

import cv2
import numpy as np
from basic_lib import Get_List
import random
import os
import time

def Get_all_IUV_point(IUV):
    generated_image = np.zeros((1200, 800)).astype(np.uint8)
    ###
    for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
        x, y = np.where(IUV[:, :, 0] == PartInd)
        u_current_points = IUV[:, :, 1][x, y]  # Pixels that belong to this specific part.
        v_current_points = IUV[:, :, 2][x, y]
        v_tmp = ((255 - v_current_points) * 199. / 255.).astype(int)
        u_tmp = (u_current_points * 199. / 255.).astype(int)
        i = (PartInd - 1) // 6
        j = (PartInd - 1) % 6
        generated_image[(200 * j) + v_tmp, (200 * i) + u_tmp] = 255
    return generated_image

def IUVToImage(Tex_Atlas,IUV):
    for PartInd in range(1,25):
        i = (PartInd - 1) // 6
        j = (PartInd - 1) % 6
        x, y = np.where(IUV[:, :, 0] == PartInd)
        if len(x) == 0:
            continue
        IUV[x, y, ...] = Tex_Atlas[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), :][((255-IUV[:,:,2][x,y])*199./255.).astype(int),(IUV[:,:,1][x,y]*199./255.).astype(int),...]
    return IUV

def refresh(lis,data):
    for i in range(len(lis)-1):
        lis[len(lis)-i-1] = lis[len(lis)-i-2]
    lis[0] = data
    return lis

_,name_all = Get_List('./video/tmp')
texture_all = []
for name in name_all:
    texture_all.append(cv2.imread(os.path.join('./video/tmp/',name)))
texture_length = len(texture_all)

root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess'
pose_path = os.path.join(root,'org')
uv_map_path = os.path.join(root,'uv_unwrap')
_,name_all = Get_List(pose_path)
_,uv_name_all = Get_List(uv_map_path)


for name1,name2 in zip(name_all,uv_name_all):
    time_old = time.time()
    pose = cv2.imread(os.path.join(pose_path,name1))
    pose = cv2.resize(pose, (pose.shape[1] // 4, pose.shape[0] // 4))
    uv = cv2.imread(os.path.join(uv_map_path, name2))

    texture_all = refresh(texture_all,uv)
    all_point = Get_all_IUV_point(pose)

    x_all,y_all = np.where(all_point > 0)
    # 确定一次需要更新多少个点
    point_replace = len(x_all) // texture_length
    texture_choise = np.random.choice(texture_length, size=texture_length, replace=False)
    all_point = np.zeros((1200, 800,3)).astype(np.uint8)

    for index in texture_choise[:-1]:
        index_exist = np.arange(x_all.shape[0])
        inxex_choise = np.random.choice(x_all.shape[0], point_replace, replace=False)
        all_point[x_all[inxex_choise], y_all[inxex_choise], ...] = \
            texture_all[index][x_all[inxex_choise], y_all[inxex_choise], ...]

        index_exist = np.delete(index_exist, inxex_choise)
        x_all = x_all[index_exist]
        y_all = y_all[index_exist]
    all_point[x_all, y_all, ...] = texture_all[-1][x_all, y_all, ...]

    pose = IUVToImage(all_point,pose)
    print(time.time() - time_old)
    cv2.imshow('a', pose)
    cv2.waitKey(1)