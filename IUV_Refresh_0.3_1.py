# coding=utf-8
import json
import os
import cv2
import numpy as np
import sys
from basic_lib import Get_List, ImageToIUV, IUVToImage
import random
list_name = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
             "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip",
             "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe",
             "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]

head_Candidate = [0, 15, 16, 17, 18]
L_arm_Candidate = [5, 6, 7]
R_arm_Candidate = [2, 3, 4]
body_Candidate = [2, 5, 1, 9, 8, 12]
L_leg_Candidate = [12, 13, 14, 21, 20, 19]
R_leg_Candidate = [9, 10, 11, 24, 22, 23]

body_iuv = [1, 2]
head_iuv = [23, 24]
R_Arm_iuv = [3, 16, 18, 20, 22]
L_Arm_iuv = [4, 15, 17, 19, 21]
R_Leg_iuv = [6, 9, 13, 7, 11]
L_Leg_iuv = [5, 10, 14, 8, 12]

all_part = [head_Candidate, L_arm_Candidate, R_arm_Candidate, body_Candidate, L_leg_Candidate, R_leg_Candidate]
all_part_iuv = [head_iuv, L_Arm_iuv, R_Arm_iuv, body_iuv, L_Leg_iuv, R_Leg_iuv]


# head_Candidate = [list_name[i] for i in head_Candidate]
# L_arm_Candidate = [list_name[i] for i in L_arm_Candidate]
# R_arm_Candidate = [list_name[i] for i in R_arm_Candidate]
# body_Candidate = [list_name[i] for i in body_Candidate]
# L_leg_Candidate = [list_name[i] for i in L_leg_Candidate]
# R_leg_Candidate = [list_name[i] for i in R_leg_Candidate]

# 剪切图片生成video 为openpose做准备
def img_process(img, loadsize):
    try:
        h, w, _ = img.shape
    except:
        print("hah")
    result = np.zeros((loadsize, loadsize, 3))
    if h >= w:
        w = int(w * loadsize / h)
        h = loadsize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - w) / 2)
        img = np.array(img)
        result[0:h, bias:bias + w, ...] = img[0:h, 0:w, ...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - h) / 2)
        img = np.array(img)
        result[bias:bias + h, 0:w, ...] = img[0:h, 0:w, ...]
    result = result.astype(np.uint8)
    return result


def text_save(path_root, name_all, save_path):
    file = open(save_path, 'a')
    for name_index in range(0, len(name_all)):
        body_json_name = os.path.join(path_root, name_all[name_index])
        pose_dict = read_json_file(body_json_name)
        write_data = []
        if len(pose_dict) == 0:
            for i in range(25):
                write_data.append(0)
        else:
            for name in list_name[:-1]:
                write_data.append(pose_dict[name])
        for data in write_data:
            file.write(str(data))
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


def find_close_part_pose(pose_all, Candidate, find_data,delt):
    dist_all = []
    star_point = -1
    for i in Candidate:
        if sum(find_data[i]) != 0:
            star_point = i
            break
    if star_point == -1:
        return -1
    for hang in pose_all:
        tmp = 0
        for index in Candidate:
            if sum(find_data[index]) == 0 or sum(hang[index]) == 0:
                tmp += hang[index][0] + find_data[index][0] + hang[index][1] + find_data[index][1]
            else:
                tmp += abs((hang[index][0] - hang[star_point][0]) - (find_data[index][0] - find_data[star_point][0])) \
                       + abs((hang[index][1] - hang[star_point][1]) - (find_data[index][1] - find_data[star_point][1]))
        dist_all.append(tmp)
    tmp = dist_all.copy()
    tmp.sort()
    return dist_all.index(tmp[delt])


def Get_all_pose(file_path):
    with open(file_path, 'r') as file:
        listall = file.readlines()
        listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
        result = []
        for i in range(len(listall)):
            tmp = []
            for j in range(0,len(listall[i]),2):
                tmp.append([int(listall[i][j]),int(listall[i][j+1])])
            result.append(tmp)
    return result


def Mulit_ImageToIUV_part(im_all, IUV_all, part_id, IUV_map):
    TextureIm = np.zeros([len(part_id), 200, 200, 3]).astype(np.uint8)
    TextureIm_tmp = np.zeros([len(part_id), 200, 200, 3]).astype(np.uint8)
    for im, IUV in zip(im_all, IUV_all):
        U = IUV[:, :, 1]
        V = IUV[:, :, 2]
        I = IUV[:, :, 0]
        ###
        for index, PartInd in enumerate(part_id):  ## Set to xrange(1,23) to ignore the face part.
            x, y = np.where(I == PartInd)
            u_current_points = U[x, y]  # Pixels that belong to this specific part.
            v_current_points = V[x, y]
            v_tmp = ((255 - v_current_points) * 199. / 255.).astype(int)
            u_tmp = (u_current_points * 199. / 255.).astype(int)
            TextureIm_tmp[index, v_tmp, u_tmp, ...] = im[x, y, ...]
        for index, PartInd in enumerate(part_id):
            x, y = np.where((TextureIm[index, :, :, 0] + TextureIm[index, :, :, 2]) == 0)
            TextureIm[index, x, y, ...] = TextureIm_tmp[index, x, y, ...]
        TextureIm_tmp = TextureIm_tmp * 0

    for index, PartInd in enumerate(part_id):
        j = (PartInd - 1) % 6
        i = int((PartInd - 1) / 6)
        IUV_map[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), ...] = TextureIm[index, ...]

    return IUV_map


def refresh_IUV(img_path, dense_path, part_id, IUV_map):
    im_all = []
    IUV_all = []
    for img, dense in zip(img_path, dense_path):
        im_all.append(cv2.imread(img))
        IUV_all.append(cv2.imread(dense))
    IUV_map_new = Mulit_ImageToIUV_part(im_all, IUV_all, part_id, IUV_map)
    return IUV_map_new


def generate_all_texture(root):
    pose_root = os.path.join(root, 'org')
    img_root = os.path.join(root, 'img')
    _, name_pose_all = Get_List(pose_root)
    _, name_img_all = Get_List(img_root)
    name_pose_all.sort()
    name_img_all.sort()
    all = []
    for index in range(len(name_img_all)):
        tmp = os.path.join(img_root, name_img_all[index])
        tmp_1 = os.path.join(pose_root, name_pose_all[index])
        all.append([tmp, tmp_1])
    return all


target_root = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/'
target_body_json_path = os.path.join(target_root, "openpose/body_json")
target_org_path = os.path.join(target_root, 'DensePoseProcess','org')
target_save_path = os.path.join(target_root, 'DensePoseProcess','IUV_map_part')

source_root = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/DensePoseProcess/'
texture_root_all = generate_all_texture(source_root)

txt_save_path = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/openpose/body_loc.txt'
# 读取所有的pose信息
pose_all = Get_all_pose(txt_save_path)
# init
basic_img = './video/video_1.png'
basic_pose = './video/video_1_IUV.png'
basic_back_img = './video/video_1_back.png'
basic_back_pose = './video/video_1_back_IUV.png'
basic_IUV = ImageToIUV(cv2.imread(basic_img),cv2.imread(basic_pose))
basic_IUV_back = ImageToIUV(cv2.imread(basic_back_img),cv2.imread(basic_back_pose))
tmp = basic_IUV[..., 0] + basic_IUV[..., 1] + basic_IUV[..., 2]
x,y = np.where((tmp) == 0)
basic_IUV[x,y,...] = basic_IUV_back[x,y,...]
IUV_map = basic_IUV.copy()

# 文件路径确定

_, target_body_json_name_list = Get_List(target_body_json_path)
_, target_org_name_list = Get_List(target_org_path)
target_org_name_list.sort()
target_body_json_name_list.sort()
# video
fps = 30
img_size = (1152, 1152)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_path = 'video_1_train.avi'
videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
kernel = np.ones((5, 5), np.uint8)
redio = 0.1
delt = 30
for index, org_name in enumerate(target_org_name_list):
    org_img = cv2.imread(os.path.join(target_org_path, org_name))
    json_path = os.path.join(target_body_json_path, target_body_json_name_list[index])
    pose_dict = read_json_file(json_path)
    if len(pose_dict) != 0:
        find_data = []
        for name in list_name[:-1]:
            find_data.append(pose_dict[name])

        IUV_map_new = TextureIm = np.zeros((1200, 800, 3)).astype(np.uint8)
        for pose_coor, pose_iuv_coor in zip(all_part, all_part_iuv):
            part_index = find_close_part_pose(pose_all, pose_coor, find_data,delt)

            if part_index != -1 and part_index < len(texture_root_all):
                img_path = [texture_root_all[i][0] for i in
                            range(max(part_index - 5, 0), min(max(part_index - 5, 0) + 6, len(texture_root_all)))]
                img_path.remove(texture_root_all[part_index][0])
                img_path.insert(0, texture_root_all[part_index][0])
                dense_path = [texture_root_all[i][1] for i in
                              range(max(part_index - 5, 0), min(max(part_index - 5, 0) + 6, len(texture_root_all)))]
                dense_path.remove(texture_root_all[part_index][1])
                dense_path.insert(0, texture_root_all[part_index][1])
                IUV_map_new = refresh_IUV(img_path, dense_path, pose_iuv_coor, IUV_map_new)

                x, y = np.where((IUV_map[..., 0] + IUV_map[..., 1] + IUV_map[..., 2]) == 0)
                IUV_map[x, y, ...] = IUV_map_new[x, y, ...]
                x, y = np.where((IUV_map_new[..., 0] + IUV_map_new[..., 1] + IUV_map_new[..., 2]) != 0)
                rand_index = random.sample(range(len(x)), int(len(x) * redio))
                x = [x[i] for i in rand_index]
                y = [y[i] for i in rand_index]
                IUV_map[x, y, ...] = IUV_map_new[x, y, ...]

                texture_img = IUVToImage(IUV_map, org_img)
            else:
                texture_img = IUVToImage(IUV_map, org_img)
    else:
        texture_img = IUVToImage(IUV_map, org_img)

    tmp = org_img[..., 0] + org_img[..., 1] + org_img[..., 2]
    tmp[tmp > 0] = 255
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
    x, y = np.where(tmp == 0)
    texture_img[x, y, 1] = 255

    cv2.imwrite(os.path.join(target_save_path, org_name), texture_img)
    texture_img = img_process(texture_img, 1152)
    videoWriter.write(texture_img)

    print(index / len(target_org_name_list))
    # cv2.imshow('img',texture_img)
    # cv2.waitKey(1)