from basic_lib import Get_List,ImageToIUV,IUVToImage
import os
import shutil
from PIL import Image
import cv2
import numpy as np
# path_img = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/img'
# path_pose = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/all'
# _,pose_all = Get_List(path_pose)
# _,img_all = Get_List(path_img)
# pose_all.sort()
# img_all.sort()
#
#
# for name_pose,name_img in zip(pose_all,img_all):
#     img = Image.open(os.path.join(path_img,name_img))
#     pose = Image.open(os.path.join(path_pose,name_pose))
#     img = img.resize(pose.size, Image.ANTIALIAS)
#     img.save(os.path.join(path_img,name_img))
#     print(name_img)

# L_Leg = {'data':[5,10,14,8,12],'name':'L_Leg','value':120}
# for key in L_Leg:
#     print(key)

# path = '/media/kun/Dataset/Pose/DataSet/new_data/femal'
# _,img_all = Get_List(os.path.join(path,'img'))
# _,pose_all = Get_List(os.path.join(path,'DensePose'))
# img_all.sort()
# pose_all.sort()
#
# img = cv2.imread(os.path.join(path, 'img', img_all[0]))
# pose = cv2.imread(os.path.join(path, 'DensePose', pose_all[0]))
# IUV_map = ImageToIUV(img, pose)
# cv2.imshow('a',IUV_map)
# cv2.waitKey(0)
# for name1,name2 in zip(img_all,pose_all):
#     img = cv2.imread(os.path.join(path,'img',name1))
#     pose = cv2.imread(os.path.join(path, 'DensePose', name2))
#     IUV_map_new = ImageToIUV(img,pose)
#     x,y = np.where((IUV_map[...,0] + IUV_map[...,1] + IUV_map[...,2]) == 0)
#     IUV_map[x,y] = IUV_map_new[x,y]
#     print(name1)
# cv2.imwrite('./final.png',IUV_map)
# w = 1152
# h = 1080
# pose_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePose'
# _,name_all = Get_List(pose_path)
# name_all.sort()

# def rotate(image, angle, center=None, scale=1.0):
#     # 获取图像尺寸
#     (h, w) = image.shape[:2]
#
#     # 若未指定旋转中心，则将图像中心设为旋转中心
#     if center is None:
#         center = (w / 2, h / 2)
#
#     # 执行旋转
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(image, M, (w, h))
#
#     # 返回旋转后的图像
#     return rotated
#
# img = cv2.imread('./video/video_1.png')
# img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# cv2.imshow('b',img)
# cv2.waitKey(0)
# cv2.imshow('a',cv2.flip(img,-1))
# cv2.waitKey(0)
# cv2.imshow('a',cv2.flip(img,0))
# cv2.waitKey(0)
# cv2.imshow('a',cv2.flip(img,1))
# cv2.waitKey(0)
# cv2.imshow('a',rotate(img,90))
# cv2.waitKey(0)
# cv2.imshow('a',rotate(img,180))
# cv2.waitKey(0)
# cv2.imshow('a',rotate(img,270))
# cv2.waitKey(0)
# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('/media/kun/Dataset/Pose/DataSet/new_data/femal/feamal.avi',
#                               fourcc, fps, (w,h))
#
# for name in name_all:
#     pose = cv2.imread(os.path.join(pose_path,name))
#     out = IUVToImage(IUV_map,pose)
#     videoWriter.write(out)
#     print(name)
# videoWriter.release()

path_org = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/tmp/org'
path_img = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/img'
path_forground = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/forground'
path_dense_mask = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/dense_mask'
path_texture = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/texture_trip'
path_openpose = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/all'
path_all = [path_org,path_openpose]
name_all_path = []
for path in path_all:
    _,name_all = Get_List(path)
    name_all.sort()
    name_all_path.append(name_all)

for index in range(len(name_all_path[-1])):
    openpose = cv2.imread(os.path.join(path_all[-1],name_all_path[-1][index]))
    for path,name_all in zip(path_all[:-1],name_all_path[:-1]):
        img = cv2.imread(os.path.join(path,name_all[index]))
        img = cv2.resize(img,(openpose.shape[1],openpose.shape[0]),interpolation = cv2.INTER_NEAREST)
        # cv2.imshow('a',img[...,0]*10)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join('/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/org',name_all[index]),img)
    print(1.0*index/len(name_all_path[-1]))