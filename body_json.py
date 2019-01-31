# coding=utf-8
import json
import os
import cv2
import numpy as np
import sys
'''
max min
判断在地上的坐标才是有效的
最大值的y坐标表示距离图片底部最近的距离
最小值通过对小于中心值的点做聚类并且与最大值到中心的距离相等。比例缩放和平移
计算 source ankle的平均值、最大最小值，
ankle positions
对最大和最小的脚踝位置计算高度，
'''
list_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
             "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip",
             "LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe",
             "LSmallToe","LHeel","RBigToe","RSmallToe","RHeel","Background"]


pose_Candidate = [17,0,18,4,3,2,7,6,5,9,8,12,10,13,11,14]
pose_Candidate = [list_name[i] for i in pose_Candidate]
middle_Candidate = ['MidHip','RHip','LHip','LKnee','RKnee']
low_Candidate = ['LAnkle','RAnkle','LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']
top_Candidate = ['Nose','REye','LEye','REar','LEar']

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



def text_save(filename, data):
    file = open(filename,'a')
    if len(data) == 2:
        file.write(str(data[0]))
        file.write('\t')
        file.write(str(data[1]))
        file.write('\n')
    else:
        file.write(str(data[0]))
        file.write('\n')
    file.close()

def read_json_file(body_json_name):
    pose_dict = {}
    flag_dele = False
    with open(body_json_name, 'r') as load_f:
        load_dict = json.load(load_f)
        try:
            pose_list = load_dict['people'][0]['pose_keypoints_2d']
        except:
            flag_dele = True
            print("no pose")
            return {}
        if not flag_dele and len(load_dict['people'][0]['pose_keypoints_2d']) != 75:
            print("no pose")
            return {}
        for i in range(0, 75, 3):
            tmp = int(i / 3)
            pose_dict[list_name[tmp]] = [int(pose_list[i]), int((pose_list[i + 1]))]
    return pose_dict

target_body_json_path = "/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/body_json"
target_img_path = "/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/cut_expend"

source_body_json_path = "/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/body_json"
source_img_path = "/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/cut_expend"

# 文件路径确定
_,body_json_name_list = Get_List(source_body_json_path)
body_json_name_list.sort()
_,source_img_name_list = Get_List(source_img_path)
source_img_name_list.sort()

_,target_body_json_name_list = Get_List(target_body_json_path)
target_body_json_name_list.sort()
_,target_img_name_list = Get_List(target_img_path)
target_img_name_list.sort()


all_pose = []
for name_index in range(0,len(body_json_name_list),1):

    body_json_name = os.path.join(source_body_json_path, body_json_name_list[name_index])
    pose_dict = read_json_file(body_json_name)
    if len(pose_dict) == 0:
        continue
    # 坐标筛选
    tmp = [pose_dict[i] for i in pose_Candidate]
    all_pose.append(tmp)
    print(1.0*name_index/len(body_json_name_list))
index = 1000
print("start_test")
while 1:
    print("waity key please in put index for 0 ~ %d"%len(target_img_name_list))
    test_path = index
    index+=1
    if test_path == 'break':
        break
    print(test_path)
    test_path = int(test_path)
    target_img = cv2.imread(os.path.join(target_img_path,target_img_name_list[test_path]))
    cv2.imshow('target_img',target_img)
    cv2.waitKey(0)

    print("start chack")
    json_path = os.path.join(target_body_json_path,target_body_json_name_list[test_path])
    if os.path.isfile(json_path):
        pose_dict = read_json_file(json_path)
    else:
        print("file not exist")
        continue
    tmp = [pose_dict[i] for i in pose_Candidate]
    dist_all = []
    for i in all_pose:
        dist = 0
        for j in range(len(i)):
            dist += abs(tmp[j][0]-i[j][0]) + abs(tmp[j][1]-i[j][1])
        dist_all.append(dist)
    index = dist_all.index(min(dist_all))
    print("get close img " + body_json_name_list[index])
    source_img = cv2.imread(os.path.join(source_img_path,source_img_name_list[index]))
    cv2.imshow('source_img',source_img)
    cv2.waitKey(0)
print("finish all")





