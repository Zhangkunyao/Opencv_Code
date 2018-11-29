from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,get_bbox
import random

pose_root = '/home/kun/Documents/DataSet/video_06/cut/pose'
img_root = '/home/kun/Documents/DataSet/video_06/cut/img'

_,img_names = Get_List(pose_root)
img_names.sort()
for i in range(len(img_names)):
    name = img_names[i]
    # name = 'video_06_000000000299_rendered.png'
    # img_path = os.path.join(img_root, name)
    pose_path = os.path.join(pose_root, name)
    CRF_path = os.path.join(CRF_root, name)

    # img = np.array(Image.open(img_path))
    pose = np.array(Image.open(pose_path))
    CRF = np.array(Image.open(CRF_path))

    bbox_loc = get_bbox(CRF)
    xmin = bbox_loc['xmin']
    xmax = bbox_loc['xmax']
    ymin = bbox_loc['ymin']
    ymax = bbox_loc['ymax']

    text_save(txt_path,[xmin,xmax,ymin,ymax])
    if xmax>xmin and ymax>ymin:
        # img = img[ymin:ymax,xmin:xmax]
        pose = pose[ymin:ymax, xmin:xmax]

    # img = Image.fromarray(img)
    pose = Image.fromarray(pose)

    # img.save(os.path.join(save_img_root,name))
    pose.save(os.path.join(save_pose_root,name))

    print(i/len(img_names))
print('finished')
