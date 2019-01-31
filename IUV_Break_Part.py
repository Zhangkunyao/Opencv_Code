# coding=utf-8
import cv2
import numpy as np
import os
from basic_lib import Get_List,ImageToIUV,IUVToImage
from PIL import Image
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


body = {'data':[1,2],'name':'body'}
head = {'data':[23,24],'name':'head'}
R_Arm = {'data':[3,16,18,20,22],'name':'R_Arm'}
L_Arm = {'data':[4,15,17,19,21],'name':'L_Arm'}
R_Leg = {'data':[6,9,13,7,11],'name':'R_Leg'}
L_Leg = {'data':[5,10,14,8,12],'name':'L_Leg'}

sub_part = [body,head,R_Arm,L_Arm,R_Leg,L_Leg]
target_img = cv2.imread('/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png')
target_h = target_img.shape[0]
kernel = np.ones((5, 5), np.uint8)
kernel_big = np.ones((6, 6), np.uint8)
data_root = '/media/kun/Dataset/Pose/DataSet/new_data/bilibili_3/DensePoseProcess/'

print(data_root)
pose_org_root = os.path.join(data_root,'org')
pose_pro_root = os.path.join(data_root,'pose')
iuv_root = os.path.join(data_root,'IUV_map')
img_root = os.path.join(data_root,'img')
save_root = os.path.join(data_root,'part')

_,name_pose_org_all =  Get_List(pose_org_root)
_,name_pose_pro_all =  Get_List(pose_pro_root)
_,name_img_all =  Get_List(img_root)
_,name_IUV_all =  Get_List(iuv_root)

name_pose_org_all.sort()
name_pose_pro_all.sort()
name_img_all.sort()
name_IUV_all.sort()

for index in range(len(name_IUV_all)):

    # bilibili_3_000000002390_rendered.png
    pose_org = cv2.imread(os.path.join(pose_org_root,name_pose_org_all[index]))
    pose_pro = cv2.imread(os.path.join(pose_pro_root,name_pose_pro_all[index]))
    texture = cv2.imread(os.path.join(iuv_root,name_IUV_all[index]))
    img = cv2.imread(os.path.join(img_root,name_img_all[index]))

    scale = target_h*1.0/pose_org.shape[0]
    pose_org = cv2.resize(pose_org, (int(scale*pose_org.shape[1]), target_h),
                          interpolation=cv2.INTER_CUBIC)
    scale = target_h*1.0/pose_pro.shape[0]
    pose_pro = cv2.resize(pose_pro, (int(scale*pose_pro.shape[1]), target_h),
                          interpolation=cv2.INTER_CUBIC)
    scale = target_h*1.0/texture.shape[0]
    texture = cv2.resize(texture, (int(scale*texture.shape[1]), target_h),
                          interpolation=cv2.INTER_CUBIC)
    scale = target_h*1.0/img.shape[0]
    img = cv2.resize(img, (int(scale*img.shape[1]), target_h),
                          interpolation=cv2.INTER_CUBIC)

    shape_w_all = [img.shape[1],texture.shape[1],pose_pro.shape[1]]
    shape_h_all = [img.shape[0], texture.shape[0], pose_pro.shape[0]]
    h_min = min(shape_h_all)-1
    w_min = min(shape_w_all)-1
    I = pose_org[:,:,0]

    for part in sub_part:
        out = None
        tmp = I>255
        for PartInd in part['data']:
            tmp = tmp|(I == PartInd)
        tmp = tmp*255
        tmp = tmp[...,np.newaxis]
        tmp = np.repeat(tmp,3,2).astype(np.uint8)
        tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
        tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel_big)
        # if part['name'] == 'body':
        #     cv2.imshow('tmp',tmp)
        #     cv2.waitKey(0)

        tmp = np.array(tmp[...,0]+tmp[...,1]+tmp[...,2])
        x,y = np.where(tmp>0)

        x[x >= h_min] = h_min-1
        y[y >= w_min] = w_min-1

        # if part['name'] == 'body' and len(x)!=0 and len(y)!=0 and (max(x) - min(x))!=0 and (max(y) - min(y))!=0:
        #     tmp2 = tmp[min(x):max(x), min(y):max(y)]
        #     cv2.imshow('tmp2', tmp2)
        #     cv2.waitKey(0)

        save_path = os.path.join(save_root,part['name'])
        if len(x)!=0:
            box_xmin = min(x)
            box_xmax = max(x)
            box_ymin = min(y)
            box_ymax = max(y)
            h = box_xmax - box_xmin+1
            w = box_ymax - box_ymin+1

            if min(w, h) < 5:
                img_out = np.zeros((256, 256, 3)).astype(np.uint8)
                IUV_map_out = np.zeros((256, 256, 3)).astype(np.uint8)
                pose_out = np.zeros((256, 256, 3)).astype(np.uint8)
                img_out[..., 1] = 255
                IUV_map_out[..., 1] = 255
            else:
                img_out = np.zeros((h,w,3)).astype(np.uint8)
                IUV_map_out = np.zeros((h, w, 3)).astype(np.uint8)
                pose_out = np.zeros((h, w, 3)).astype(np.uint8)
                img_out[..., 1] = 255
                IUV_map_out[..., 1] = 255

                img_out[x-box_xmin,y-box_ymin] = img[x,y,...]
                IUV_map_out[x-box_xmin,y-box_ymin] = texture[x,y,...]
                pose_out[x - box_xmin, y - box_ymin] = pose_pro[x, y, ...]
        else:
            img_out = np.zeros((256, 256,3)).astype(np.uint8)
            IUV_map_out = np.zeros((256, 256,3)).astype(np.uint8)
            pose_out = np.zeros((256, 256,3)).astype(np.uint8)
            img_out[..., 1] = 255
            IUV_map_out[..., 1] = 255

        cv2.imwrite(os.path.join(save_path,'pose',name_pose_pro_all[index]),pose_out)
        cv2.imwrite(os.path.join(save_path, 'img', name_img_all[index]), img_out)
        cv2.imwrite(os.path.join(save_path, 'IUV_map', name_IUV_all[index]), IUV_map_out)
    print(index*1.0/len(name_IUV_all))
    # out = np.array(out.rotate(90))
    # cv2.imshow('head',out)
    # cv2.waitKey(1)
    # img_process()
    # if key == ord(' '):
    #     print(name)
    #     cv2.waitKey(0)