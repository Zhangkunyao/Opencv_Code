import numpy
import cv2
import matplotlib.pyplot as plt
from basic_lib import Get_List
import os
import numpy as np
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



IUV_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePose'
Img_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/img'
IUV_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/pose'
Img_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/img.txt'

_,IUV_ALL = Get_List(IUV_path_root)
index = 0

# IUV_path = os.path.join(IUV_path_root,IUV_ALL[0])
# IUV = cv2.imread(IUV_path)


for name in IUV_ALL:
    IUV_path = os.path.join(IUV_path_root,name)
    img_name = name[:-8] + '.png'
    img_path = os.path.join(Img_path_root,img_name)
    IUV_save_path = os.path.join(IUV_save_root,img_name)

    IUV = cv2.imread(IUV_path)
    fig = plt.figure(figsize=[IUV.shape[1] / 100.0, IUV.shape[0] / 100.0])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow( IUV[:,:,::-1] )
    plt.contour( IUV[:,:,1]/256.,15, linewidths = 5 )
    plt.contour( IUV[:,:,2]/256.,15, linewidths = 5 )
    plt.savefig(IUV_save_path)
    plt.close()
    img_path = 'cp -r ' + img_path + ' /media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/img'
    text_save(Img_save_root,img_path)
    index = index+1
    print(index/len(IUV_ALL))



# plt.show()

# tmp = cv2.imread('./tmp.png')
# # tmp = rotate(tmp,-180)
# #
# im[...,0] = im[...,0]*0.7 + tmp[...,0]*0.3
# im[...,1] = im[...,1]*0.7 + tmp[...,1]*0.3
# im[...,2] = im[...,2]*0.7 + tmp[...,2]*0.3
# cv2.imshow('a',im)
# cv2.waitKey(0)