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


def DenseposeProcess(img,target_shape):
    IUV_size = img.size
    scale = 1.0 * target_shape[1] / img.size[1]
    if scale * img.size[0] > size_target[0] * 2:
        IUV = img.resize((int(img.size[0]), int(img.size[1] * scale)), Image.ANTIALIAS)
    else:
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
    plt.savefig(os.path.join(path_root, 'tmp.png'))
    plt.close()
    final = Image.open(os.path.join(path_root, 'tmp.png')).convert('RGB')
    final = final.resize((IUV_size[0], IUV_size[1]), Image.ANTIALIAS)
    final = np.array(final)[:, :, ::-1]
    return final

path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/'
IUV_path_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/org'
IUV_save_root = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/pose'

target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'
target_img = Image.open(target_img_path).convert('RGB')
size_target = target_img.size

_,IUV_ALL = Get_List(IUV_path_root)
IUV_ALL.sort()
index = 0

for name in IUV_ALL:
    IUV_path = os.path.join(IUV_path_root,name)
    IUV_save_path = os.path.join(IUV_save_root,name)

    IUV = Image.open(IUV_path).convert('RGB')
    IUV_size = IUV.size

    scale = 1.0*size_target[1]/IUV.size[1]
    if scale*IUV.size[0]>size_target[0]*2:
        IUV = IUV.resize((int(IUV.size[0]), int(IUV.size[1] * scale)), Image.ANTIALIAS)
    else:
        IUV = IUV.resize((int(IUV.size[0]*scale),int(IUV.size[1]*scale)),Image.ANTIALIAS)

    fig = plt.figure(figsize=[IUV.size[0] / 100.0, IUV.size[1] / 100.0])
    IUV = np.array(IUV)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow( IUV)
    plt.contour(IUV[:, :, 0] / 256., 15, linewidths=5)
    plt.contour( IUV[:,:,1]/256.,15, linewidths = 5 )
    plt.contour( IUV[:,:,2]/256.,15, linewidths = 3 )
    plt.savefig(IUV_save_path)
    plt.close()
    # final = Image.open(os.path.join(path_root,'tmp.png')).convert('RGB')
    # final = final.resize((IUV_size[0], IUV_size[1]), Image.ANTIALIAS)
    # final = np.array(final)[:,:,::-1]
    # cv2.imwrite(IUV_save_path,final,[int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    index = index+1
    print(index/len(IUV_ALL))
