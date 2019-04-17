# coding=utf-8
import json
import os
from basic_lib import Get_List

'''
max min
按照Candidate 的坐标提取数据，存放成文件.这样会快一些
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


def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i[0]))
        file.write('\t')
        file.write(str(i[1]))
        file.write('\t')
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

body_json_path = "/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/openpose/body_json"
file_path = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/openpose/body_loc.txt'
# 文件路径确定
_,body_json_name_list = Get_List(body_json_path)
body_json_name_list.sort()

all_pose = []
for name_index in range(len(body_json_name_list)):

    body_json_name = os.path.join(body_json_path, body_json_name_list[name_index])
    pose_dict = read_json_file(body_json_name)
    if len(pose_dict) == 0:
        tmp = [[0,0] for i in range(len(list_name)-1)]
    else:
        tmp = [pose_dict[i] for i in list_name[:-1]]
    text_save(file_path, tmp)
    # all_pose.append(tmp)
    print(1.0*name_index/len(body_json_name_list))






