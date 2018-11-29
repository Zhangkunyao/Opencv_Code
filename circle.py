# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os

from basic_lib import Get_List,get_bbox


path_root = '/media/kun/Dataset/Pose/DataSet/result_hand_face/芭蕾_1/expend'
load_size = 256
_,target_pose_all = Get_List(path_root)
target_pose_all.sort()
ave_hight = 0
ave_weight = 0
ave_y = 0
ave_x = 0
for i in range(len(target_pose_all)):
    print(i/len(target_pose_all))
    pose_name = target_pose_all[i]
    pose_path = os.path.join(path_root, pose_name)
    target_pose = cv2.imread(pose_path)
    target_pose = cv2.resize(target_pose, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    glob_bbox = get_bbox(target_pose)

    xmin = glob_bbox['xmin']
    ymin = glob_bbox['ymin']
    xmax = glob_bbox['xmax']
    ymax = glob_bbox['ymax']
    ave_hight += ymax - ymin
    ave_weight += xmax - xmin
    ave_y += ymin
    ave_x += (xmax + xmin)/2.0
ave_hight = ave_hight/len(target_pose_all)
ave_weight = ave_weight/len(target_pose_all)
ave_y = ave_y/len(target_pose_all)
ave_x = ave_x/len(target_pose_all)
print("ave_hight = %4f"%ave_hight)
print("ave_weight = %4f"%ave_weight)
print("ave_y = %4f"%ave_y)
print("ave_x = %4f"%ave_x)
#     cv2.rectangle(target_pose, (xmin-5, ymin-5), (xmax+5, ymax+5), (0, 0, 255), 1)
#     cv2.imshow('1', target_pose)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()