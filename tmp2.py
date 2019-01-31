# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List,ImageToIUV,IUVToImage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

# target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/back_ground.png'
# target_img = cv2.imread(target_img_path)
# size_target = target_img.shape
#
# img_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/img'
# _,name_all = Get_List(img_root_path)
# name_all.sort()
#
# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/0001_cut_densepose.avi',
#                               fourcc, fps, (size_target[1],size_target[0]))
# if not videoWriter.isOpened():
#     print("video error")
#     exit(0)
#
# for i in range(len(name_all)):
#     img_path = os.path.join(img_root_path,name_all[i])
#     img = cv2.imread(img_path)
#     videoWriter.write(img)
#     print(1.0*i/len(name_all))
# videoWriter.release()
# print('finish')


# txt_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/x_loc.txt'
# dense_pose_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/org'
# _,name_all = Get_List(dense_pose_path)
# name_all.sort()
#
# file = open(txt_path, 'w')
#
# for i,name in enumerate(name_all):
#     img_path = os.path.join(dense_pose_path,name)
#     IUV = cv2.imread(img_path)
#     I = IUV[...,0]
#     # loc_x = []
#     # loc_y = []
#     # loc_id = []
#     # tmp = np.zeros(I.shape)
#     # x_loc_final=0
#     # for PartInd in range(1, 25):
#     #     x, y = np.where(I == PartInd)
#     #     if len(x)!=0:
#     #         loc_x.append(x.mean())
#     #         loc_y.append(y.mean())
#     #         loc_id.append(PartInd)
#     # if len(loc_x) == 0:
#     #     x_loc_final=0
#     #     print("error")
#     # else:
#     #     index = loc_y.index(max(loc_y))
#     #     x_loc_final = loc_x[index]
#     #     PartInd = loc_id[index]
#     PartInd = 6
#     x, y = np.where(I == PartInd)
#     if len(x) == 0:
#         x_loc_final = IUV.shape[1]/2
#     else:
#         x_loc_final = x.mean()
#     # tmp[I>0]=128
#     # tmp[x, y] = 255
#     # cv2.imshow('a',tmp.astype(np.uint8))
#     # cv2.waitKey(0)
#     file.write(str(int(x_loc_final)))
#     file.write('\n')
#     print(i/len(name_all))
# file.close()
# densepose_img = cv2.imread('./video/video_06_IUV.png')
# I = densepose_img[:,:,0]
# tmp=np.zeros(densepose_img.shape)
# img_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/save_result'
# video_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/video_iuv_refresh.avi'
# cap0 = cv2.VideoCapture(video_path)
# _,name_pose = Get_List(img_path)
# name_pose.sort()
# for name in name_pose:
#     img = cv2.imread(os.path.join(img_path,name))
#     ret0, frame0 = cap0.read()
#     frame0 = cv2.resize(frame0, (512, 512), interpolation=cv2.INTER_CUBIC)
#     img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
#     result = np.concatenate([frame0,img],axis=1)
#     cv2.imshow('1',result.astype(np.uint8))
#     # cv2.waitKey(10)
#
#
#     # frame1 = cv2.resize(frame1, (640,480), interpolation=cv2.INTER_CUBIC)
#     # out = np.concatenate([frame0,frame1],axis=1)
#     # # if ret0:
#     # #     cv2.imshow('frame0', frame0)
#     # #     cv2.setWindowTitle('frame0','On Top')
#     # # if ret1:
#     # cv2.imshow('frame1', out)
#     # # cv2.moveWindow('frame1', x=frame0.shape[1], y=0)
#     # # cv2.moveWindow('frame1', x=0, y=0)
#     #
#     key = cv2.waitKey(delay=10)
#     if key == ord("q"):
#         break
#     if key == ord(" "):
#         cv2.waitKey(delay=0)


# pose_test = cv2.imread('./video/机械哥_bilibili_IUV.png')
# body = [[1,2],[0,0,255]]
# head = [[23,24],[0,255,0]]
# R_Arm = [[3,16,18,20,22],[255,0,0]]
# L_Arm = [[4,15,17,19,21],[255,255,0]]
# R_Leg = [[6,9,13,7,11],[0,255,255]]
# L_Leg = [[5,10,14,8,12],[255,0,255]]
# dict_all = {'body':body,'head':head,'R_Arm':R_Arm,'L_Arm':L_Arm,'R_Leg':R_Leg,'L_Leg':L_Leg}
# path = '/media/kun/Dataset/Pose/DataSet/new_data/bilibili_3/DensePoseProcess/org'
# _,name_all =  Get_List(path)
# name_all.sort()
# for name in name_all:
#     # bilibili_3_000000002390_rendered.png
#     pose = cv2.imread(os.path.join(path,name))
#     I = pose[:,:,0]
#     out = np.zeros(pose.shape).astype(np.uint8)
#     for PartInd in range(1, 25):
#         x, y = np.where(I == PartInd)
#         for colour in dict_all:
#             idx = dict_all[colour][0]
#             if PartInd in idx:
#                 out[x,y,0]=dict_all[colour][1][0]
#                 out[x, y, 1] = dict_all[colour][1][1]
#                 out[x, y, 2] = dict_all[colour][1][2]
#     cv2.imshow('a',out)
#     key = cv2.waitKey(1)
#     if key == ord(' '):
#         print(name)
#         cv2.waitKey(0)


# path = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/img/'
# _,name_all = Get_List(path)
# name_all.sort()
# for name in name_all:
#     img = cv2.imread(os.path.join(path,name))
#     img = img*0
#     img[...,1]=255
#     cv2.imshow('img',img)
#     cv2.imwrite('basic_zero.png',img)
    # key = cv2.waitKey(1)
    # if key ==ord(' '):
    #     print(name)

# a=[i for i in range(10)]
# a = np.array(a)
# plt.plot(a,'ro')
# plt.show()
pose_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/org'
texture_1_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/uv_unwrap_0.1'
texture_2_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/uv_unwrap'
_,tex_name_1 = Get_List(texture_1_path)
_,tex_name_2 = Get_List(texture_2_path)
_,pose_name = Get_List(pose_root)
tex_name_1.sort()
tex_name_2.sort()
pose_name.sort()
for name_1,name_2,name_3 in zip(tex_name_1,tex_name_2,pose_name):
    tex_img_1 = cv2.imread(os.path.join(texture_1_path,name_1))
    tex_img_2 = cv2.imread(os.path.join(texture_2_path, name_2))
    pose_img = cv2.imread(os.path.join(pose_root, name_3))
    result_1 = IUVToImage(tex_img_1,pose_img)
    result_2 = IUVToImage(tex_img_2, pose_img)
    result = np.concatenate([result_1, result_2], axis=1)
    cv2.imshow('a',result)
    cv2.waitKey(1)