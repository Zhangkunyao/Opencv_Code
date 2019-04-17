import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage
from PIL import Image


img_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/uv_unwrap'
_,name_all = Get_List(img_root)
name_all.sort()
for name_index in range(len(name_all)):
    name = name_all[name_index]
    img = cv2.imread(os.path.join(img_root,name))
    img = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('a',img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('a'):
        name_index+=10