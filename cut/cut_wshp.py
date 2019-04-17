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

    bbox_loc = get_muliti_bbox(wshp)
    w = bbox_loc['xmax'] - bbox_loc['xmin']
    h = bbox_loc['ymax'] - bbox_loc['ymin']
    xmin = bbox_loc['xmin'] - w * 0.1
    xmax = bbox_loc['xmax'] + w * 0.1
    ymin = bbox_loc['ymin'] - h * 0.05
    ymax = bbox_loc['ymax'] + h * 0.05

    if i == 0:
        x_min_filter = [xmin for i in x_min_filter]
        x_max_filter = [xmax for i in x_max_filter]
        y_min_filter = [ymin for i in y_min_filter]
        y_max_filter = [ymax for i in y_max_filter]
    x_min_filter = refresh(x_min_filter, xmin)
    x_max_filter = refresh(x_max_filter, xmax)
    y_min_filter = refresh(y_min_filter, ymin)
    y_max_filter = refresh(y_max_filter, ymax)

    xmin = max(int(np.array(x_min_filter).mean()),0)
    xmax = min(int(np.array(x_max_filter).mean()),img.shape[1])
    ymin = max(int(np.array(y_min_filter).mean()),0)
    ymax = min(int(np.array(y_max_filter).mean()),img.shape[0])

    text_save(txt_path,[xmin,xmax,ymin,ymax])
    if xmax>xmin and ymax>ymin:
        tmp = img[ymin:ymax,xmin:xmax]
        tmp = Image.fromarray(tmp)
        scale = 1.0 * target_img_h / tmp.size[1]
        tmp = tmp.resize((int(tmp.size[0] * scale), target_img_h), Image.ANTIALIAS)

        tmp.save(os.path.join(save_openpose_root, img_name))

        zero_idx = wshp == 0
        img[zero_idx[..., 0], 0] = 0
        img[zero_idx[..., 1], 1] = 255
        img[zero_idx[..., 2], 2] = 0

        pose[zero_idx[..., 0], 0] = 0
        pose[zero_idx[..., 1], 1] = 0
        pose[zero_idx[..., 2], 2] = 0

        img = img[ymin:ymax, xmin:xmax]
        pose = pose[ymin:ymax, xmin:xmax]
    else:
        zero_idx = wshp == 0
        img[zero_idx[..., 0], 0] = 0
        img[zero_idx[..., 1], 1] = 255
        img[zero_idx[..., 2], 2] = 0

        pose[zero_idx[..., 0], 0] = 0
        pose[zero_idx[..., 1], 1] = 0
        pose[zero_idx[..., 2], 2] = 0

    img = Image.fromarray(img)
    pose = Image.fromarray(pose)

    img.save(os.path.join(save_img_root,img_name))
    pose.save(os.path.join(save_pose_root,pose_name))

    print(1.0*i/len(img_names))
print('finished')
