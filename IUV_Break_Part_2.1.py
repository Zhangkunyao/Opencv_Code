# coding=utf-8
import cv2
import numpy as np
import os
from basic_lib import Get_List,ImageToIUV,IUVToImage
from PIL import Image
import time
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


body = {'data':[1,2],'name':'body','value':20}
head = {'data':[23,24],'name':'head','value':40}
R_Arm = {'data':[3,16,18,20,22],'name':'R_Arm','value':60}
L_Arm = {'data':[4,15,17,19,21],'name':'L_Arm','value':80}
R_Leg = {'data':[6,9,13,7,11],'name':'R_Leg','value':100}
L_Leg = {'data':[5,10,14,8,12],'name':'L_Leg','value':120}

# sub_part = [body,head,R_Arm,L_Arm,R_Leg,L_Leg]
sub_part = [body,head,R_Arm,L_Arm,R_Leg,L_Leg]
kernel = np.ones((5, 5), np.uint8)
# kernel_big = np.ones((6, 6), np.uint8)
data_root = '/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess'
print(data_root)
pose_root = os.path.join(data_root,'dense_mask')
image_root = os.path.join(data_root,'img')
# save_root = os.path.join(data_root,'dense_mask')

_,pose_all =  Get_List(pose_root)
_,img_all =  Get_List(image_root)
pose_all.sort()
img_all.sort()

for index in range(1000,len(img_all),1):
    pose_org = cv2.imread(os.path.join(pose_root,pose_all[index]))
    pose_org = pose_org[...,0]
    img = cv2.imread(os.path.join(image_root, img_all[index]))

    x,y = np.where([pose_org])
    final_out = np.zeros(img.shape).astype(np.uint8)
    final_out[...,1] = 255
    for part in sub_part:
        x,y = np.where(pose_org == part['value'])
        final_out[x,y] = img[x,y]
    final_out = cv2.resize(final_out,(final_out.shape[1]//2,final_out.shape[0]//2))
    pose_org = cv2.resize(pose_org,(pose_org.shape[1]//2,pose_org.shape[0]//2))

    cv2.imshow('a',final_out)
    cv2.imshow('b', pose_org)
    cv2.waitKey(0)
    # cv2.imwrite(os.path.join(save_root,name_all[index]), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # print(index*1.0/len(name_all))
    # cv2.imshow('a',out*2)
    # cv2.waitKey(1)