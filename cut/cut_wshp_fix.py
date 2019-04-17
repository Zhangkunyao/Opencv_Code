#coding=utf-8
from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,get_muliti_bbox
import cv2
import matplotlib.pyplot as plt
import sys
# resize to 1080
def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i))
        file.write('\t')
    file.write('\n')
    file.close()

def refresh(lis,data):
    for i in range(len(lis)-1):
        lis[len(lis)-i-1] = lis[len(lis)-i-2]
    lis[0] = data
    return lis

def get_all_loc(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

def DenseposeProcess(img,target_shape_h,save_path):
    scale = 1.0 * target_shape_h / img.size[1]
    IUV = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)

    plt.figure(figsize=[IUV.size[0] / 100.0, IUV.size[1] / 100.0])
    IUV = np.array(IUV)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(IUV)
    plt.contour(IUV[:, :, 0] / 256., 15, linewidths=5)
    plt.contour(IUV[:, :, 1] / 256., 15, linewidths=5)
    plt.contour(IUV[:, :, 2] / 256., 15, linewidths=3)
    plt.savefig(save_path)
    plt.close()

target_img_h = 1080
kernel = np.ones((5,5),np.uint8)
# path_root = '/media/kun/Dataset/Kun/Dataset/densepose/new_video/Long_Time/video_3/'
path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06'

pose_root = os.path.join(path_root,'DensePose')
img_root = os.path.join(path_root,'img')
WSHP_root = os.path.join(path_root,'WSHP')

if not os.path.isdir(os.path.join(path_root,'DensePoseProcess')):
    os.mkdir(os.path.join(path_root,'DensePoseProcess'))
if not os.path.isdir(os.path.join(path_root,'DensePoseProcess','img')):
    os.mkdir(os.path.join(path_root,'DensePoseProcess','img'))
if not os.path.isdir(os.path.join(path_root, 'DensePoseProcess','org')):
    os.mkdir(os.path.join(path_root, 'DensePoseProcess','org'))


save_img_root = os.path.join(path_root,'DensePoseProcess','img')
save_pose_root = os.path.join(path_root,'DensePoseProcess','org')
save_openpose_root = os.path.join(path_root,'DensePoseProcess','openpose','img')
txt_path = os.path.join(path_root,'DensePoseProcess','loc.txt')
loc_all = get_all_loc(txt_path)

_,pose_names = Get_List(pose_root)
pose_names.sort()

_,img_names = Get_List(img_root)
img_names.sort()


x_min_filter = [0 for i in range(10)]
x_max_filter = [0 for i in range(10)]
y_min_filter = [0 for i in range(10)]
y_max_filter = [0 for i in range(10)]

for i in range(len(img_names)):
    pose_name = pose_names[i]
    img_name = pose_name[:-8]+'.jpg'

    img_path = os.path.join(img_root, img_name[:-4]+'.png')
    pose_path = os.path.join(pose_root, pose_name)
    wshp_path = os.path.join(WSHP_root, img_name[:-4]+'.png')

    wshp = np.array(Image.open(wshp_path))
    img = np.array(Image.open(img_path))
    pose = np.array(Image.open(pose_path))
    wshp = wshp[...,0] + wshp[...,1] + wshp[...,2] + pose[...,0] + pose[...,1] + pose[...,2]
    wshp = wshp[...,np.newaxis]
    wshp = np.repeat(wshp,3,2)
    wshp[wshp > 0] = 255
    wshp = cv2.morphologyEx(wshp, cv2.MORPH_CLOSE, kernel)

    bbox_loc = loc_all[i]

    xmin = bbox_loc[0]
    xmax = bbox_loc[1]
    ymin = bbox_loc[2]
    ymax = bbox_loc[3]
    if xmax > xmin and ymax > ymin:
        img = img[ymin:ymax, xmin:xmax]
        wshp = wshp[ymin:ymax, xmin:xmax]
    zero_idx = wshp == 0

    img[zero_idx[..., 0], 0] = 0
    img[zero_idx[..., 1], 1] = 255
    img[zero_idx[..., 2], 2] = 0

    img = Image.fromarray(img)

    img.save(os.path.join(save_img_root,img_name))

    print(1.0*i/len(img_names))
print('finished')
