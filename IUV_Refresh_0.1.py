# coding=utf-8
import random
import os
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage
import numpy as np
'''
读取txt文件，存放成列表。
按照最近的距离找出相近的pose
按照部分刷新的原则刷新IUV_map
'''

# 剪切图片生成video 为openpose做准备
def img_process(img,loadsize):
    try :
        h, w ,_= img.shape
    except:
        print("hah")
    result = np.zeros((loadsize,loadsize,3))
    if h >= w:
        w = int(w*loadsize/h)
        h = loadsize
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - w)/2)
        img = np.array(img)
        result[0:h,bias:bias+w,...] = img[0:h,0:w,...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - h)/2)
        img = np.array(img)
        result[bias:bias+h,0:w,...] = img[0:h,0:w,...]
    result = result.astype(np.uint8)
    return result

def txt_read(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

def Mulit_ImageToIUV(im_all,IUV_all):
    TextureIm = np.zeros([24, 200, 200, 3]).astype(np.uint8)
    TextureIm_tmp = np.zeros([24, 200, 200, 3]).astype(np.uint8)
    for im,IUV in zip(im_all,IUV_all):
        U = IUV[:,:,1]
        V = IUV[:,:,2]
        I = IUV[:,:,0]
        ###
        for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
            x,y = np.where(I==PartInd)
            u_current_points = U[x,y]   #  Pixels that belong to this specific part.
            v_current_points = V[x,y]
            v_tmp = ((255 - v_current_points) * 199. / 255.).astype(int)
            u_tmp = (u_current_points * 199. / 255.).astype(int)
            TextureIm_tmp[PartInd - 1,v_tmp,u_tmp,...] = im[x, y,...]
        for PartInd in range(1,25):
            x, y = np.where((TextureIm[PartInd - 1,:,:, 0] + TextureIm[PartInd - 1,:,:,2]) == 0)
            TextureIm[PartInd - 1,x, y, ...] = TextureIm_tmp[PartInd - 1,x, y, ...]
        TextureIm_tmp = TextureIm_tmp*0
    generated_image = np.zeros((1200, 800, 3)).astype(np.uint8)
    for i in range(4):
        for j in range(6):
            generated_image[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200),...] = TextureIm[(6 * i + j),...]
    return generated_image

def refresh_IUV(img_path,dense_path):
    im_all = []
    IUV_all = []
    for img,dense in zip(img_path,dense_path):
        im_all.append(cv2.imread(img))
        IUV_all.append(cv2.imread(dense))
    IUV_map_new = Mulit_ImageToIUV(im_all,IUV_all)
    return IUV_map_new

def get_index(body_loc_all,find_body):
    if sum(find_body) == 0:
        return -1
    dist_all = []
    for i in body_loc_all:
        if sum(i) == 0:
            dist = 1000000
        else:
            dist = 0
        for j in range(len(i)):
            dist += abs(find_body[j] - i[j])
        dist_all.append(dist)
    index = dist_all.index(min(dist_all))
    return index

basic_img = './video/video_06.png'
basic_pose = './video/video_06_IUV.png'
basic_IUV = ImageToIUV(cv2.imread(basic_img),cv2.imread(basic_pose))
IUV_map = ImageToIUV(cv2.imread(basic_img),cv2.imread(basic_pose))

source_file_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/body_loc.txt'
target_file_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/body_loc.txt'
source_body_loc = txt_read(source_file_path)
target_body_loc = txt_read(target_file_path)

source_pose_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/org'
target_pose_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/org'
target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/img'

_,name_all_source = Get_List(source_pose_path)
_,name_all_target_pose = Get_List(target_pose_path)
_,name_all_target_img = Get_List(target_img_path)
name_all_source.sort()
name_all_target_pose.sort()
name_all_target_img.sort()
# 刷新比例，有redio的新信息被刷新
redio = 0.1
#
# fps = 30
# img_size = (1152, 1152)
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# video_path = 'video_iuv_refresh_face_keep.avi'
# videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)

for index,name in enumerate(name_all_source):
    source_pose = cv2.imread(os.path.join(source_pose_path,name))
    source_loc = source_body_loc[index]

    find_index = get_index(target_body_loc,source_loc)
    if find_index == -1:
        source_img = IUVToImage(IUV_map,source_pose)
    else:
        img_path = [os.path.join(target_img_path, name_all_target_img[i]) for i in
                    range(max(find_index - 5, 0), min(max(find_index - 5, 0) + 5, len(name_all_target_img)))]
        dense_path = [os.path.join(target_pose_path, name_all_target_pose[i]) for i in
                      range(max(find_index - 5, 0), min(max(find_index - 5, 0) + 5, len(name_all_target_pose)))]
        IUV_map_new = refresh_IUV(img_path,dense_path)
        # 填补空白区域
        x, y = np.where((IUV_map[..., 0] + IUV_map[..., 1] + IUV_map[..., 2]) == 0)
        IUV_map[x, y, ...] = IUV_map_new[x, y, ...]
        # 替换一部分区域
        x, y = np.where((IUV_map_new[..., 0] + IUV_map_new[..., 1] + IUV_map_new[..., 2]) != 0)
        rand_index = random.sample(range(len(x)),int(len(x)*redio))
        x = [x[i] for i in rand_index]
        y = [y[i] for i in rand_index]
        IUV_map[x, y, ...] = IUV_map_new[x, y, ...]
        # # 替换所有的头部位置
        # head_part = IUV_map_new[(200 * 4):(200 * 5 + 200), (200 * 3):(200 * 3 + 200), ...]
        # x,y = np.where((head_part[..., 0] + head_part[..., 1] + head_part[..., 2]) != 0)
        # IUV_map[x+200*4,y+200*3] = head_part[x,y]
        # 保持头部位置不变
        head_part = basic_IUV[(200 * 4):(200 * 5 + 200), (200 * 3):(200 * 3 + 200), ...]
        x,y = np.where((head_part[..., 0] + head_part[..., 1] + head_part[..., 2]) != 0)
        IUV_map[x+200*4,y+200*3] = head_part[x,y]

        source_img = IUVToImage(IUV_map, source_pose)
    # source_img = img_process(source_img, 1152)
    # videoWriter.write(source_img)
    print(index*1.0/len(name_all_source))
# videoWriter.release()
    # cv2.imshow('img',source_img)
    # cv2.waitKey(1)
