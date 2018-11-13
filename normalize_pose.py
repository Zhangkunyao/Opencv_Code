# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List,get_bbos

save_path = '/media/kun/Dataset/Pose/DataSet/result_hand_face/0001.哔哩哔哩-【机器人索尔】每天学一点，简单街舞教学，机械舞震感舞动作配合音乐即兴表演[超清版]/normallize'
root_path = '/media/kun/Dataset/Pose/DataSet/result_hand_face/0001.哔哩哔哩-【机器人索尔】每天学一点，简单街舞教学，机械舞震感舞动作配合音乐即兴表演[超清版]'

load_size = 256
target_data = {'ave_hight':158.132784,'ave_weight':44.472980,'ave_y':73.224910,'ave_x':139.159290}
source_data = {'ave_hight':162.241520,'ave_weight':47.344300,'ave_y':72.888039,'ave_x':111.002168}
hight_redio = target_data['ave_hight']/source_data['ave_hight']
width_redio = target_data['ave_weight']/source_data['ave_weight']
y_redio = target_data['ave_y']-source_data['ave_y']
x_redio = target_data['ave_x']-source_data['ave_x']



_,source_pose_all = Get_List(os.path.join(root_path,'pose'))
# opencv shape (hight,weight,(bgr))
count_all = 0
for i in range(len(source_pose_all)):
    print(i/len(source_pose_all))
    pose_name = source_pose_all[i]
    pose_path = os.path.join(os.path.join(root_path,'expend'), pose_name)
    img_path = os.path.join(os.path.join(root_path,'img'), pose_name)
    source_pose = cv2.imread(pose_path)

    width_redio_img = 1.0*load_size/source_pose.shape[1]
    hight_redio_img = 1.0*load_size/source_pose.shape[0]
    # target_pose = cv2.resize(target_pose, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    glob_bbox = get_bbos(source_pose)

    xmin = max(glob_bbox['xmin']-5,0)
    ymin = max(glob_bbox['ymin']-5,0)
    xmax = min(glob_bbox['xmax']+5,source_pose.shape[1])
    ymax = min(glob_bbox['ymax']+5,source_pose.shape[0])

    high = ymax - ymin
    weight = xmax - xmin
    # 有些图片没有检测出来pose 删除
    if high<0 or weight<0:
        os.remove(pose_path)
        os.remove(img_path)
        print("bad data "+pose_name)
        continue
    # 计算resize之后的区域大小
    x_min_resize = max(int(xmin * width_redio_img + x_redio * hight_redio_img),0)
    y_min_resize = max(int(ymin * hight_redio_img + y_redio * hight_redio_img),0)

    # 计算归一化之后的bbox大小
    pose_roi = source_pose[ymin:ymax:,xmin:xmax,...]
    pose_roi = cv2.resize(pose_roi, (int(weight*width_redio* width_redio_img),
                                     int(high*hight_redio* hight_redio_img)),
                          interpolation=cv2.INTER_CUBIC)
    # 计算坐标位置
    xmin = x_min_resize
    ymin = y_min_resize
    xmax = min(xmin + pose_roi.shape[1],load_size)
    ymax = min(ymin + pose_roi.shape[0],load_size)

    result = np.zeros((load_size,load_size,3))
    result[ymin:ymax,xmin:xmax,...] = pose_roi[0:(ymax-ymin),0:(xmax-xmin),...]
    count_all = count_all+1
    cv2.imwrite(os.path.join(save_path, pose_name), result)
print(count_all)