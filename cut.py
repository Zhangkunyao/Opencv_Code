from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,get_bbox
import random
import cv2


def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        file.write(str(i))
        file.write('\t')
    file.write('\n')
    file.close()
kernel = np.ones((5,5),np.uint8)
pose_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/pose'
img_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/img'
WSHP_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePose/'

save_img_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/cut/img'
save_pose_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/cut/pose'
txt_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/cut/loc.txt'

_,img_names = Get_List(pose_root)
img_names.sort()
for i in range(len(img_names)):
    name = img_names[i]
    # name = 'video_06_000000000299_rendered.png'
    img_path = os.path.join(img_root, name)
    pose_path = os.path.join(pose_root, name)
    name_wshp = name[:-4]+'_IUV.png'
    wshp_path = os.path.join(WSHP_root, name_wshp)

    wshp = np.array(Image.open(wshp_path))
    img = np.array(Image.open(img_path))
    pose = np.array(Image.open(pose_path))
    wshp = wshp[...,0] + wshp[...,1] + wshp[...,2]
    wshp = wshp[...,np.newaxis]
    wshp = np.repeat(wshp,3,2)
    wshp[wshp > 0] = 255
    # wshp[wshp < 0] = 0
    wshp = cv2.morphologyEx(wshp, cv2.MORPH_CLOSE, kernel)

    out = cv2.bitwise_and(img,wshp)
    zero_idx = wshp == 0
    img[zero_idx[...,0],0] = 0
    img[zero_idx[..., 1], 1] = 255
    img[zero_idx[..., 2], 2] = 0

    pose[zero_idx[...,0],0] = 0
    pose[zero_idx[..., 1], 1] = 0
    pose[zero_idx[..., 2], 2] = 0

    bbox_loc = get_bbox(wshp)
    xmin = bbox_loc['xmin']
    xmax = bbox_loc['xmax']
    ymin = bbox_loc['ymin']
    ymax = bbox_loc['ymax']

    text_save(txt_path,[xmin,xmax,ymin,ymax])
    if xmax>xmin and ymax>ymin:
        img = img[ymin:ymax,xmin:xmax]
        pose = pose[ymin:ymax, xmin:xmax]

    img = Image.fromarray(img)
    pose = Image.fromarray(pose)

    img.save(os.path.join(save_img_root,name))
    pose.save(os.path.join(save_pose_root,name))
    print(i/len(img_names))
print('finished')
