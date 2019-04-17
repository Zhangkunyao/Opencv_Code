# -*- coding: utf-8 -*-
import numpy as np
import cv2
import json
from basic_lib import Get_List
import os
import random
list_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
             "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip",
             "LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe",
             "LSmallToe","LHeel","RBigToe","RSmallToe","RHeel","Background"]

key_points = ['pose_keypoints_2d','face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
              ]

# pose_colour = [[153,0,153],[153,0,102],[102,0,153],[51,0,153],
#           [153,0,51],[153,51,0],[153,102,0],[153,153,0],[102,153,0],[51,153,0],[0,153,0],
#           [153,0,0],[0,153,51],[0,153,102],[0,153,153],   [102,0,51],[102,0,51],[102,0,51],
#           [0,102,153],[0,51,153],[0,0,153],   [51,0,102],[51,0,102],[51,0,102]]
#
# pose_order = [[17,15],[15,0],[0,16],[16,18],
#          [0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
#          [1,8],[8,9],[9,10],[10,11],[11,24],[11,22],[22,23],
#          [8,12],[12,13],[13,14],[14,21],[14,19],[19,20]]
pose_order = [[0,1],[1,2],[1,5],[1,8],[8,9],[8,12],[17,15],[15,0],[0,16],[16,18],
         [2,3],[3,4],[5,6],[6,7],
         [9,10],[10,11],[12,13],[13,14],
         [11,24],[11,22],[22,23],[14,21],[14,19],[19,20]]

pose_colour = [[153,0,153],[153,0,102],[102,0,153],[51,0,153],
          [153,0,51],[153,51,0],[153,102,0],[153,153,0],[102,153,0],[51,153,0],[0,153,0],
          [153,0,0],[0,153,51],[0,153,102],[0,153,153],
          [0,102,153],[0,51,153],[0,0,153],
          [102,0,51],[102,0,51],[102,0,51], [51,0,102],[51,0,102],[51,0,102]]

hand_order = [[0,1],[1,2],[2,3],[3,4],
              [0,5],[5,6],[6,7],[7,8],
              [0,9],[9,10],[10,11],[11,12],
              [0,13],[13,14],[14,15],[15,16],
              [0,17],[17,18],[18,19],[19,20],
              ]

face_order = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
              [17,18],[18,19],[19,20],[20,21],
              [21,22],[22,23],[23,24],[24,25],[25,26],
              [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
              [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
              [27,28],[28,29],[29,30],
              [31,32],[32,33],[33,34],[34,35],
              [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
              [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
              [68],[69]
            ]


name_to_index ={0:"Nose",1:"Neck",2:"RShoulder",3:"RElbow",4:"RWrist",5:"LShoulder",6:"LElbow",7:"LWrist",8:"MidHip",9:"RHip",10:"RKnee",
11:"RAnkle",12:"LHip",13:"LKnee",14:"LAnkle",15:"REye",16:"LEye",17:"REar",18:"LEar",19:"LBigToe",20:"LSmallToe",21:"LHeel",
 22:"RBigToe",23:"RSmallToe",24:"RHeel",25:"Background"}



for i in range(len(pose_colour)):
    for j in range(len(pose_colour[i])):
        pose_colour[i][j] = min(255,int(pose_colour[i][j]*(255./153.)))


def read_json_file_all(body_json_name,threshold=0.05):
    pose_dict = {}
    with open(body_json_name, 'r') as load_f:
        load_dict = json.load(load_f)
        load_dict = load_dict['people'][0]
        for name in key_points:
            if not (name in load_dict):
                continue
            pose_list = load_dict[name]
            if name == 'pose_keypoints_2d':
                tmp = {}
                for i in range(0, 75, 3):
                    tmp2 = int(i / 3)
                    if pose_list[i+2]>threshold:
                        tmp[list_name[tmp2]] = [int(pose_list[i]), int((pose_list[i + 1]))]
                    else:
                        tmp[list_name[tmp2]] = [0, 0]
                pose_dict['pose_keypoints_2d']=tmp
            else:
                tmp = []
                for i in range(0,len(pose_list),3):
                    if pose_list[i + 2] > threshold:
                        tmp.append([int(pose_list[i]), int((pose_list[i + 1]))])
                    else:
                        tmp.append([0,0])
                pose_dict[name] = tmp
    return pose_dict


def read_json_file_pose(body_json_name):
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

def plot_pose(img,point_all,order,thickness=5,colour=None,flage=False):
    if flage:
        for name in point_all:
            delt = random.randint(-8, 8)
            if point_all[name][0]>0 and point_all[name][0]+delt>0:
                point_all[name][0] = point_all[name][0]+delt
                delt = random.randint(-8, 8)
            if point_all[name][1]>0 and point_all[name][1]+delt>0:
                point_all[name][1] = point_all[name][1]+delt
    for point, col in zip(order, colour):
        start = point[0]
        end = point[1]
        start_name = name_to_index[start]
        end_name = name_to_index[end]
        if start_name in point_all and end_name in point_all:
            start_point = tuple(point_all[start_name])
            end_point = tuple(point_all[end_name])
            if sum(end_point) !=0 and sum(start_point) !=0:
                cv2.line(img, start_point, end_point, col, thickness)

def plot_point(img,point_all,order,thickness=5,flage = False):
    if flage:
        for i in range(len(point_all)):
            delt = random.randint(-3, 3)
            if point_all[i][0]>0 and point_all[i][0]+delt>0:
                point_all[i][0] = point_all[i][0]+delt
                delt = random.randint(-3, 3)
            if point_all[i][1]>0 and point_all[i][1]+delt>0:
                point_all[i][1] = point_all[i][1]+delt

    for point in order:
        if len(point) ==2:
            start = point[0]
            end = point[1]
            if end< len(point_all):
                start_point = tuple(point_all[start])
                end_point = tuple(point_all[end])
                if sum(end_point) !=0 and sum(start_point) !=0:
                    cv2.line(img, start_point, end_point, [255,255,255], thickness)
        else:
            if point[0]<len(point_all):
                start_point = tuple(point_all[point[0]])
                if sum(start_point) > 0:
                    cv2.circle(img, start_point, thickness, [255,255,255], thickness)


# json_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/openpose/all_json'
# _,name_all = Get_List(json_root)

path_tmp = './video/tmp/video_06_000000001960_rendered_keypoints.json'
img=np.zeros((1080,1080,3), np.uint8)
for name in range(10000000):
    json_path = path_tmp# os.path.join(json_root,name_all[1200])
    point_all = read_json_file_all(json_path)
    for point_name in key_points:
        if point_name == 'pose_keypoints_2d':
            plot_pose(img,point_all[point_name],pose_order,thickness=15,colour=pose_colour,flage=True)
        elif point_name == 'face_keypoints_2d':
            plot_point(img, point_all[point_name], face_order, thickness=5,flage=True)
        else:
            plot_point(img, point_all[point_name], hand_order, thickness=5,flage=True)

    cv2.imshow('a',cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    img = img*0