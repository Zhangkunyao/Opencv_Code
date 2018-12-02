from skimage import io,color
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

file_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/txt'
path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePoseProcess/pose'
file = open(file_path,'r')
listall = file.readlines()
listall = [i.rstrip('\n').split('/')[-1] for i in listall]

for i in listall:
    path=os.path.join(path_root,i)
    os.remove(path)


# kernel = np.ones((5,5),np.uint8)
# img = io.imread('./video/video.png')
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#
# cv2.imshow("img", closing)
# cv2.waitKey(0)

