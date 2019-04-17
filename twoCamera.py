# -*- coding: utf-8 -*-
# @Time    : 2017/8/15 00:19
# @Author  : play4fun
# @File    : two_camera.py
# @Software: PyCharm

"""
two_camera.py:
"""

import cv2
import numpy as np
# 贴图
path_old = '/home/kun/Documents/DataSet/video/Densepose/贴图方法对比/video_0.1_keepface_test.avi'
path_new = '/home/kun/Documents/DataSet/video/Densepose/贴图方法对比/分part贴图/video_part_0.1.avi'
# 时间
path_time_1 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video.avi'
path_time_2 = '/home/kun/Documents/DataSet/video/Densepose/去除抖动/机械哥_bilibili/densepose/video.avi'
path_time_3 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local.avi'
path_time_4 = '/home/kun/Documents/DataSet/video/Densepose/time/mulit/video_time_mulit_normal_texture.avi'
path_time_5 = '/home/kun/Documents/DataSet/video/Densepose/time/mulit/video_time_mulit_trip_texture_lowrefresh.avi'
# 网络
path_net_1 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/video.avi'
path_net_2 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/video_2.avi'
path_net_3 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/video_time_7.avi'
# local
path_new_1 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local.avi'
path_new_2 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local_3.avi'
path_new_3 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local_Redio.avi'
path_new_4 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local_GP.avi'
path_new_5 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_time.avi'
path_new_6 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_time_7.avi'
path_new_7 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_local_trip.avi'
path_new_8 = '/home/kun/Documents/DataSet/video/Densepose/IUV_Refresh0.1/机械哥bilibili/video_trip_time.avi'
# trip loss
path_trip_1 = '/home/kun/Documents/DataSet/video/Densepose/tripletloss/video_net_triplet_normal.avi'
path_trip_2 = '/home/kun/Documents/DataSet/video/Densepose/tripletloss/video_net_triplet_no_trip_texture.avi'
path_trip_3 = '/home/kun/Documents/DataSet/video/Densepose/tripletloss/video_net_triplet_lowrefresh.avi'
#
path_tmp_1 = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/video_gan_loss.avi'
path_tmp_2 = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/video_gan_loss_openpose.avi'
cap0 = cv2.VideoCapture(path_tmp_1)
cap1 = cv2.VideoCapture(path_tmp_2)
# ret = cap0.set(3, 320)
# ret = cap0.set(4, 240)
# ret = cap1.set(3, 320)
# ret = cap1.set(4, 240)
index = 0
while cap0.isOpened() and cap1.isOpened():

    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = frame0[:,256:,:]
    frame1 = frame1[:,256:,:]
    frame0 = cv2.resize(frame0, (640,960),interpolation=cv2.INTER_CUBIC)
    frame1 = cv2.resize(frame1, (640,960), interpolation=cv2.INTER_CUBIC)
    out = np.concatenate([frame0,frame1],axis=1)

    # if ret0:
    #     cv2.imshow('frame0', frame0)
    #     cv2.setWindowTitle('frame0','On Top')
    # if ret1:
    cv2.imshow('frame1', out)
    # cv2.moveWindow('frame1', x=frame0.shape[1], y=0)
    # cv2.moveWindow('frame1', x=0, y=0)

    key = cv2.waitKey(delay=1)
    if key == ord("q"):
        break
    if key == ord(" "):
        cv2.waitKey(delay=0)

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
