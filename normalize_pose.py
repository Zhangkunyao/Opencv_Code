# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os

def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    return [dirList, fileList]

def get_bbos(img):
    img = img>0
    sp = img.shape  # 行 列
    # 先做二值化处理
    test_scale = 1
    # 思想 选取目标点按以前的方式，暴力外扩 跳变点的方式无法确定是不是同一个物体
    data_map = np.ones(sp)
    glob_bbox = [sp[0], sp[1], 0, 0]
    for hang in range(0, sp[0], int(sp[0] / 100)):
        for lie in range(0, sp[1], int(sp[1] / 100)):
            if data_map[hang][lie] == 1:  # 该点还没被查找过
                if tmp[hang][lie]:  # 该处存在特征点
                    data_map[hang][lie] = 0
                    xmin = lie
                    xmax = lie
                    ymin = hang
                    ymax = hang
                    exit_flag = 0
                    # center=lie   #寻找的中心位置
                    # 先找右边
                    right_hang = hang
                    right_lie = lie
                    left_hang = hang
                    left_lie = lie

                    flag_right_out = 0
                    flag_left_out = 0
                    while exit_flag == 0:
                        flag_right_out = 0
                        flag_left_out = 0
                        # right_lie=center
                        while right_lie + test_scale < sp[1] and tmp[right_hang][right_lie]:  # 找列
                            right_lie = right_lie + test_scale

                        if xmax < right_lie:
                            xmax = right_lie;

                        # 找左边
                        # left_lie = center
                        while left_lie - test_scale > 0 and tmp[left_hang][left_lie]:  # 找列
                            left_lie = left_lie - test_scale

                        if xmin > left_lie:
                            xmin = left_lie
                        # 行标下移
                        if left_hang + test_scale < sp[0]:
                            left_hang = left_hang + test_scale
                        while left_lie + test_scale < right_lie and (tmp[left_hang][left_lie] == False):  # 找行
                            left_lie = left_lie + test_scale
                        if left_lie + test_scale >= right_lie or left_hang + test_scale >= sp[0]:
                            flag_left_out = 1

                        if right_hang + test_scale < sp[0]:
                            right_hang = right_hang + test_scale
                        while right_lie - test_scale > left_lie and (tmp[right_hang][right_lie] == False):  # 找行
                            right_lie = right_lie - test_scale
                        if right_lie - test_scale <= left_lie or right_hang + test_scale >= sp[0]:
                            flag_right_out = 1

                        if (flag_left_out == 1 and flag_right_out == 1):
                            exit_flag = 1

                    if left_hang > right_hang:
                        ymax = left_hang
                    else:
                        ymax = right_hang
                    ymin = hang
                    if xmin < glob_bbox[0]:
                        glob_bbox[0] = xmin
                    if ymin < glob_bbox[1]:
                        glob_bbox[1] = ymin
                    if xmax > glob_bbox[2]:
                        glob_bbox[2] = xmax
                    if ymax > glob_bbox[3]:
                        glob_bbox[3] = ymax
                    data_map[ymin - 1:ymax + 1, xmin - 1:xmax + 1] = 0;
    return {'xmin':glob_bbox[0],'ymin':glob_bbox[1],'xmax':glob_bbox[2],'ymax':glob_bbox[3]}

save_path = '/media/kun/Dataset/Pose/DataSet/result_hand_face/芭蕾_1/result'
load_size = 256
target_data = {'ave_hight':158.132784,'ave_weight':44.472980,'ave_y':152.291302}
source_data = {'ave_hight':158.078426,'ave_weight':44.654822,'ave_y':153.798096}
hight_redio = target_data['ave_hight']/source_data['ave_hight']
width_redio = target_data['ave_weight']/source_data['ave_weight']
y_redio = target_data['ave_y']-source_data['ave_y']
root_path = '/media/kun/Dataset/Pose/DataSet/result_hand_face/芭蕾_1/'

_,source_pose_all = Get_List(os.path.join(root_path,'pose'))
# opencv shape (hight,weight,(bgr))
count_all = 0
for i in range(len(source_pose_all)):
    print(i/len(source_pose_all))
    pose_name = source_pose_all[i]
    pose_path = os.path.join(os.path.join(root_path,'pose'), pose_name)
    img_path = os.path.join(os.path.join(root_path,'img'), pose_name)
    target_pose = cv2.imread(pose_path)

    width_redio_img = 1.0*load_size/target_pose.shape[1]
    hight_redio_img = 1.0*load_size/target_pose.shape[0]
    # target_pose = cv2.resize(target_pose, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    tmp = (target_pose[:,:,0]>0)*1 + (target_pose[:,:,1]>0)*1 + (target_pose[:,:,2]>0)*1
    tmp = tmp > 0
    glob_bbox = get_bbos(tmp)

    xmin = max(glob_bbox['xmin']-5,0)
    ymin = max(glob_bbox['ymin']-5,0)
    xmax = min(glob_bbox['xmax']+5,target_pose.shape[1])
    ymax = min(glob_bbox['ymax']+5,target_pose.shape[0])

    high = ymax - ymin
    weight = xmax - xmin
    if high<0 or weight<0:
        os.remove(pose_path)
        os.remove(img_path)
        print("bad data "+pose_name)
        continue
    middle_x = int((xmax + xmin) / 2 * width_redio_img)
    middle_y = int((ymax + ymin) / 2 * hight_redio_img + y_redio * hight_redio_img)

    pose_roi = target_pose[ymin:ymax:,xmin:xmax,...]
    pose_roi = cv2.resize(pose_roi, (int(weight*width_redio* width_redio_img),
                                     int(high*hight_redio* hight_redio_img)),
                          interpolation=cv2.INTER_CUBIC)

    xmin = max(middle_x - int((weight * width_redio* width_redio_img / 2)),0)
    ymin = max(middle_y - int((high * hight_redio* width_redio_img / 2)), 0)
    xmax = min(xmin + pose_roi.shape[1],load_size)
    ymax = min(ymin + pose_roi.shape[0],load_size)

    result = np.zeros((load_size,load_size,3))
    result[ymin:ymax,xmin:xmax,...] = pose_roi[0:(ymax-ymin),0:(xmax-xmin),...]
    count_all = count_all+1
    cv2.imwrite(os.path.join(save_path, pose_name), result)
print(count_all)