import numpy
import cv2
import matplotlib.pyplot as plt
from basic_lib import Get_List
import os
import numpy as np
from PIL import Image

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7

def text_save(filename, data):
    file = open(filename,'a')
    file.write(data)
    file.write('\n')
    file.close()

IUV_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePose'
Img_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/img'

IUV_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/pose/'
img_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/img/'
txt_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/img.sh'

_,IUV_ALL = Get_List(IUV_path_root)
index = 0

for name in IUV_ALL:
    IUV_path = os.path.join(IUV_path_root,name)
    img_name = name[:-8] + '.png'
    img_path = os.path.join(Img_path_root,img_name)
    IUV_save_path = os.path.join(IUV_save_root,img_name)

    if os.path.exists(img_path) and os.path.exists(IUV_path):
        cmd = 'cp -r ' + img_path + ' ' + img_save_root
        text_save(txt_save_root,cmd)
        cmd = 'cp -r ' + IUV_path + ' ' + IUV_save_path
        text_save(txt_save_root, cmd)
cmd = 'bash '+txt_save_root
os.system(cmd)