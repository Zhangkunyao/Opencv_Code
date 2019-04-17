# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
from basic_lib import Get_List,ImageToIUV,IUVToImage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import random
from PIL import Image

# target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/back_ground.png'
# target_img = cv2.imread(target_img_path)
# size_target = target_img.shape
#
# img_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/img'
# _,name_all = Get_List(img_root_path)
# name_all.sort()
#
# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('/media/kun/Dataset/Pose/DataSet/new_data/0001_cut/0001_cut_densepose.avi',
#                               fourcc, fps, (size_target[1],size_target[0]))
# if not videoWriter.isOpened():
#     print("video error")
#     exit(0)
#
# for i in range(len(name_all)):
#     img_path = os.path.join(img_root_path,name_all[i])
#     img = cv2.imread(img_path)
#     videoWriter.write(img)
#     print(1.0*i/len(name_all))
# videoWriter.release()
# print('finish')


# txt_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/x_loc.txt'
# dense_pose_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/org'
# _,name_all = Get_List(dense_pose_path)
# name_all.sort()
#
# file = open(txt_path, 'w')
#
# for i,name in enumerate(name_all):
#     img_path = os.path.join(dense_pose_path,name)
#     IUV = cv2.imread(img_path)
#     I = IUV[...,0]
#     # loc_x = []
#     # loc_y = []
#     # loc_id = []
#     # tmp = np.zeros(I.shape)
#     # x_loc_final=0
#     # for PartInd in range(1, 25):
#     #     x, y = np.where(I == PartInd)
#     #     if len(x)!=0:
#     #         loc_x.append(x.mean())
#     #         loc_y.append(y.mean())
#     #         loc_id.append(PartInd)
#     # if len(loc_x) == 0:
#     #     x_loc_final=0
#     #     print("error")
#     # else:
#     #     index = loc_y.index(max(loc_y))
#     #     x_loc_final = loc_x[index]
#     #     PartInd = loc_id[index]
#     PartInd = 6
#     x, y = np.where(I == PartInd)
#     if len(x) == 0:
#         x_loc_final = IUV.shape[1]/2
#     else:
#         x_loc_final = x.mean()
#     # tmp[I>0]=128
#     # tmp[x, y] = 255
#     # cv2.imshow('a',tmp.astype(np.uint8))
#     # cv2.waitKey(0)
#     file.write(str(int(x_loc_final)))
#     file.write('\n')
#     print(i/len(name_all))
# file.close()
# densepose_img = cv2.imread('./video/video_06_IUV.png')
# I = densepose_img[:,:,0]
# tmp=np.zeros(densepose_img.shape)
# img_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/save_result'
# video_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/video_iuv_refresh.avi'
# cap0 = cv2.VideoCapture(video_path)
# _,name_pose = Get_List(img_path)
# name_pose.sort()
# for name in name_pose:
#     img = cv2.imread(os.path.join(img_path,name))
#     ret0, frame0 = cap0.read()
#     frame0 = cv2.resize(frame0, (512, 512), interpolation=cv2.INTER_CUBIC)
#     img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
#     result = np.concatenate([frame0,img],axis=1)
#     cv2.imshow('1',result.astype(np.uint8))
#     # cv2.waitKey(10)
#
#
#     # frame1 = cv2.resize(frame1, (640,480), interpolation=cv2.INTER_CUBIC)
#     # out = np.concatenate([frame0,frame1],axis=1)
#     # # if ret0:
#     # #     cv2.imshow('frame0', frame0)
#     # #     cv2.setWindowTitle('frame0','On Top')
#     # # if ret1:
#     # cv2.imshow('frame1', out)
#     # # cv2.moveWindow('frame1', x=frame0.shape[1], y=0)
#     # # cv2.moveWindow('frame1', x=0, y=0)
#     #
#     key = cv2.waitKey(delay=10)
#     if key == ord("q"):
#         break
#     if key == ord(" "):
#         cv2.waitKey(delay=0)


# pose_test = cv2.imread('./video/机械哥_bilibili_IUV.png')
# body = [[1,2],[0,0,255]]
# head = [[23,24],[0,255,0]]
# R_Arm = [[3,16,18,20,22],[255,0,0]]
# L_Arm = [[4,15,17,19,21],[255,255,0]]
# R_Leg = [[6,9,13,7,11],[0,255,255]]
# L_Leg = [[5,10,14,8,12],[255,0,255]]
# dict_all = {'body':body,'head':head,'R_Arm':R_Arm,'L_Arm':L_Arm,'R_Leg':R_Leg,'L_Leg':L_Leg}
# path = '/media/kun/Dataset/Pose/DataSet/new_data/bilibili_3/DensePoseProcess/org'
# _,name_all =  Get_List(path)
# name_all.sort()
# for name in name_all:
#     # bilibili_3_000000002390_rendered.png
#     pose = cv2.imread(os.path.join(path,name))
#     I = pose[:,:,0]
#     out = np.zeros(pose.shape).astype(np.uint8)
#     for PartInd in range(1, 25):
#         x, y = np.where(I == PartInd)
#         for colour in dict_all:
#             idx = dict_all[colour][0]
#             if PartInd in idx:
#                 out[x,y,0]=dict_all[colour][1][0]
#                 out[x, y, 1] = dict_all[colour][1][1]
#                 out[x, y, 2] = dict_all[colour][1][2]
#     cv2.imshow('a',out)
#     key = cv2.waitKey(1)
#     if key == ord(' '):
#         print(name)
#         cv2.waitKey(0)


# path = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/img/'
# _,name_all = Get_List(path)
# name_all.sort()
# for name in name_all:
#     img = cv2.imread(os.path.join(path,name))
#     img = img*0
#     img[...,1]=255
#     cv2.imshow('img',img)
#     cv2.imwrite('basic_zero.png',img)
    # key = cv2.waitKey(1)
    # if key ==ord(' '):
    #     print(name)

# a=[i for i in range(10)]
# a = np.array(a)
# plt.plot(a,'ro')
# plt.show()
# pose_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/org'
# texture_1_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/tmp3/'
# texture_2_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/tmp4/'
# _,tex_name_1 = Get_List(texture_1_path)
# _,tex_name_2 = Get_List(texture_2_path)
# tex_name_1.sort()
# tex_name_2.sort()
# index = 0
# video_path = './trip_low_refresh.avi'
# fps = 30
# img_size = (640, 960)
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
#
# for name_1,name_2 in zip(tex_name_1,tex_name_2):
#     index +=1
#     # if index<1000:
#     #     continue
#     tex_img_1 = cv2.imread(os.path.join(texture_1_path,name_1))
#     tex_img_2 = cv2.imread(os.path.join(texture_2_path, name_2))
#     result = np.concatenate([tex_img_1, tex_img_2], axis=1)
#     result = cv2.resize(result, (640,960), interpolation=cv2.INTER_CUBIC)
#     videoWriter.write(result)
#     # cv2.imshow('a',result)
#     # cv2.waitKey(1)
#     print(index)
# videoWriter.release()

# path_1 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/grub_cut'
# path_2 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/WSHP'
# path_3 = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/img'
# _,gb_name_all = Get_List(path_1)
# _,wshp_all = Get_List(path_2)
# wshp_all.sort()
# gb_name_all.sort()
# for i in range(len(wshp_all)):
#     wshp = cv2.resize(cv2.imread(os.path.join(path_2,wshp_all[i])), (320, 480), interpolation=cv2.INTER_CUBIC)
#     gb = cv2.resize(cv2.imread(os.path.join(path_1, gb_name_all[i]),cv2.IMREAD_GRAYSCALE), (320, 480), interpolation=cv2.INTER_CUBIC)
#     img = cv2.resize(cv2.imread(os.path.join(path_3, wshp_all[i])), (320, 480), interpolation=cv2.INTER_CUBIC)
#
#     wshp = wshp[...,0] + wshp[...,1] + wshp[...,2]
#     wshp[wshp>0] = 255
#     # x,y = tmp>0
#     # wshp[...,0] = img[...,0]
#     # wshp[..., 0][x,y] = 0
#     # wshp[...,1] = img[...,1]
#     # wshp[..., 1][x,y] = 255
#     # wshp[...,2] = img[...,2]
#     # wshp[..., 2][x,y] = 0
#     zero_map = np.zeros(np.shape(img), dtype=np.uint8)
#     # zero_map[...,1] = 255
#     one_map = np.ones(np.shape(wshp), dtype=np.uint8)*255
#
#
#     gb = cv2.add(img, zero_map, mask=one_map - gb)
#     wshp = cv2.add(img,zero_map, mask=one_map - wshp)
#     out = np.concatenate([gb, wshp], axis=1)
#     cv2.imshow('a',out)
#     key = cv2.waitKey(delay=1)
#     if key == ord("q"):
#         break
#     if key == ord(" "):
#         cv2.waitKey(delay=0)

# path_1 = '/media/kun/UbuntuData/Kun/GAN_Realation/kun/save_result/function3_correct/371/'
# path_2 = '/media/kun/Dataset/img_align_celeba'
# path_3 = '/media/kun/Dataset/LSUN/bedroom_train_lmdb/img'
# _,name_all = Get_List(path_2)
#
# while 1:
#     index = random.randint(0, len(name_all) - 1)
#     img = cv2.imread(os.path.join(path_2,name_all[index]))
#     index+=1
#     cv2.imshow('a',img)
#     key = cv2.waitKey(0)
#     if key == ord('s'):
#         break
# cv2.imwrite('.//media/kun/Ubuntu 18.04 LTS amd64/img/'+str(index)+'.png',img)

# root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess'
# pose_path = os.path.join(root,'org')
# save_path = os.path.join(root,'dense_mask')
# kernel = np.ones((5,5),np.uint8)
# _,name_all = Get_List(pose_path)
# for name in name_all:
#     pose = cv2.imread(os.path.join(pose_path,name))
#     save_name = os.path.join(save_path,name)
#     pose = pose[...,0] + pose[...,1] + pose[...,2]
#     pose = pose[...,np.newaxis]
#     pose = np.repeat(pose,3,2)
#     pose[pose > 0] = 255
#     # wshp[wshp < 0] = 0
#     wshp = cv2.morphologyEx(pose, cv2.MORPH_CLOSE, kernel)
#     cv2.imwrite(save_name,wshp)

# path = './video/video.png'
# img = Image.open(path).convert('L')
# img = np.array(img)
# cv2.imshow('a',img)
# cv2.waitKey(0)
# img.show()
# img = cv2.imread('./video/keypoints_pose_25.png')
# cv2.imshow('a', cv2.resize(img, (img.shape[1]//2, img.shape[0]//3), interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)

# pose = cv2.imread('./video/tmp/video_06_IUV.png')
# IUV = cv2.imread('./video/tmp/video_06_unwrap.png')
#
# img_1 = IUVToImage(IUV,pose)
# img_1 = cv2.resize(img_1, (img_1.shape[1]//8, img_1.shape[0]//8), interpolation=cv2.INTER_CUBIC)
# pose = cv2.resize(pose, (pose.shape[1]//8, pose.shape[0]//8), interpolation=cv2.INTER_CUBIC)
# img_2 = IUVToImage(IUV,pose)
# result = np.concatenate([img_2,img_1],axis=1)
# result = cv2.resize(result, (result.shape[1]*4, result.shape[0]*4), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('1',result)
# cv2.waitKey(0)
body = {'data':[1,2],'name':'body','value':20}
head = {'data':[23,24],'name':'head','value':40}
R_Arm = {'data':[3,16,18,20,22],'name':'R_Arm','value':60}
L_Arm = {'data':[4,15,17,19,21],'name':'L_Arm','value':80}
R_Leg = {'data':[6,9,13,7,11],'name':'R_Leg','value':100}
L_Leg = {'data':[5,10,14,8,12],'name':'L_Leg','value':120}

sub_part = [body,head,R_Arm,L_Arm,R_Leg,L_Leg]
#
# path_1 = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/tmp/org/video_3000000000077_IUV.png'
# # path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePose/机械哥_bilibili_000000000059_rendered_IUV.png'
# # img = cv2.imread(path)
# # img = img[...,0]
# img_1 = cv2.imread(path_1)
# img_1 = img_1[...,0]*10
# # cv2.imshow('a',img*10)
# # cv2.waitKey(0)
# # img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# # for part in sub_part:
# #     x,y = np.where(img == part['value'])
# #     img[x,y] = 0
# # img = img *255
# # cv2.imshow('a',img*10)
# cv2.imshow('a',np.array(img_1))
# # img_1 = Image.fromarray(img_1)
# # img_1 = img_1.resize((img_1.size[0]*2,img_1.size[1]*2))
# img_1 = cv2.resize(img_1,(img_1.shape[1]*2,img_1.shape[0]*2),interpolation = cv2.INTER_NEAREST)

# img_1 = cv2.imread('/home/kun/Documents/1.png')
# img_2 = cv2.imread('/home/kun/Documents/2.png')
# img_3 = cv2.imread('/home/kun/Documents/3.png')
#
# tmp_1 = img_1[...,0] + img_1[...,1] + img_1[...,2]
# tmp_2 = img_2[...,0] + img_2[...,1] + img_2[...,2]
# tmp_3 = img_3[...,0] + img_3[...,1] + img_3[...,2]
#
# x,y = np.where(tmp_1 == 255)
#
# img_1[x,y] = 0
# img_2[x,y] = 0
# img_3[x,y] = 0
#
# cv2.imshow('a',img_1)
# cv2.imshow('b',img_2)
# cv2.imshow('c',img_3)
# cv2.waitKey(0)

def img_process_cv(img,flg_pose,load_size,flag_L=False):
    h, w, _ = img.shape
    if flag_L:
        result = np.zeros((load_size,load_size)).astype(np.uint8)
    else:
        result = np.zeros((load_size, load_size, 3)).astype(np.uint8)
    if not flg_pose:
        result[...,1] = 255
    if h >= w:
        w = int(w*load_size/h)
        h = load_size
        img = cv2.resize(img,(w,h),interpolation = cv2.INTER_AREA)
        bias = int((load_size - w)/2)
        result[0:h,bias:bias+w,...] = img[0:h,0:w,...]
    if w > h:
        h = int(h * load_size / w)
        w = load_size
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
        bias = int((load_size - h)/2)
        result[bias:bias + h, 0:w, ...] = img[0:h, 0:w, ...]
    return result

def img_process_trans(img, org_size):
    roi_region = None
    loadsize = img.shape[0]
    w = org_size[0]
    h = org_size[1]
    if h >= w:
        w = int(w * loadsize / h)
        h = loadsize
        bias = int((loadsize - w) / 2)
        img = np.array(img)
        roi_region = img[0:h, bias:bias + w, ...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        bias = int((loadsize - h) / 2)
        img = np.array(img)
        roi_region = img[bias:bias + h, 0:w, ...]
    w = org_size[0]
    h = org_size[1]
    return cv2.resize(roi_region, (w, h), interpolation = cv2.INTER_AREA)

path_img = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/texture_trip/'
path_mask = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/dense_mask/'
_,img_name_all = Get_List(path_img)
_,mask_name_all = Get_List(path_mask)
img_name_all.sort()
mask_name_all.sort()

for name1,name2 in zip(img_name_all,mask_name_all):

    img = cv2.imread(os.path.join(path_img,name1))
    mask = cv2.imread(os.path.join(path_mask,name2))
    mask = mask[...,0]
    tmp = img*0
    tmp[...,1] = 255

    final_out = img*0
    x,y = np.where(mask == 0)
    img[x,y] = 0

    out_all = {}
    for part in sub_part:
        out = img * 0
        out[...,1] = 255
        x,y = np.where(mask == part['value'])
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        out[x,y] = img[x,y]
        out = out[xmin:xmax, ymin:ymax]
        out = img_process_cv(out, flg_pose=False, load_size=128)
        out_all[part['name']] = out
        out_all[part['name']+'_coor'] = [xmin,xmax,ymin,ymax]
        # cv2.imshow(part['name'],out)

    for part in sub_part:
        coor = out_all[part['name']+'_coor']
        xmin = coor[0]
        xmax = coor[1]
        ymin = coor[2]
        ymax = coor[3]

        cut_boundry_x = 12#int((xmax - xmin)*1.0) - (xmax - xmin)
        cut_boundry_x = cut_boundry_x if cut_boundry_x%2 == 0 else cut_boundry_x+1

        cut_boundry_y = 12#int((ymax - ymin) * 1.0) - (ymax - ymin)
        cut_boundry_y = cut_boundry_y if cut_boundry_y % 2 == 0 else cut_boundry_y + 1

        sub_out = out_all[part['name']]
        sub_out = img_process_trans(sub_out,(ymax-ymin+cut_boundry_x,xmax-xmin+cut_boundry_y))

        bigger_mask = mask[xmin:xmax, ymin:ymax]
        bigger_mask = cv2.copyMakeBorder(bigger_mask, cut_boundry_y//2, cut_boundry_y//2, cut_boundry_x//2, cut_boundry_x//2, cv2.BORDER_REFLECT)
        # x, y = np.where(bigger_mask != part['value'])
        # bigger_mask[x,y] = 0
        # x, y = np.where(bigger_mask > 0)
        # bigger_mask[x, y] = 255
        # bigger_mask = np.array(bigger_mask).astype(np.uint8)
        # sub_out = cv2.add(sub_out, np.zeros(np.shape(sub_out), dtype=np.uint8), mask=bigger_mask)
        # cv2.imshow(part['name']+'trans', sub_out)
        x, y = np.where(bigger_mask != part['value'])
        sub_out[x,y] = [0,255,0]
        x, y = np.where(mask[xmin:xmax, ymin:ymax] == part['value'])
        final_out[x+xmin,y+ymin] = sub_out[x+cut_boundry_y//2,y+cut_boundry_y//2]

# cv2.imshow('a', tmp)
# cv2.waitKey(0)

    cv2.imshow('INTER_NEAREST',cv2.resize(final_out,(final_out.shape[1]//2,final_out.shape[0]//2)))
    cv2.waitKey(1)