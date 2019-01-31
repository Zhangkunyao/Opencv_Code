# -*-coding:utf8-*-#
import numpy as np
import cv2
import os
from basic_lib import Get_List
# GCD_BGD（=0），背景；
#
# GCD_FGD（=1），前景；
#
# GCD_PR_BGD（=2），可能是背景；
#
# GCD_PR_FGD（=3），可能是前景。
def get_all_loc(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

kernel_small = np.ones((5,5),np.uint8)
kernel_big = np.ones((50,50),np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

data_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06'
txt_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess'
save_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/grub_cut'
mask_root = os.path.join(data_root,'DensePose')
img_root = os.path.join(data_root,'img')
txt_root = os.path.join(txt_root,'loc.txt')

loc_all = get_all_loc(txt_root)
_,name_all = Get_List(mask_root)
name_all.sort()
for index,name in enumerate(name_all):
    # if index<4000:
    #     continue
    tmp = loc_all[index]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}
    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    rect = (point['xmin'], point['ymin'], w, h)


    img_name = name[:-8]+'.png'
    img_name = os.path.join(img_root,img_name)
    mask_name = os.path.join(mask_root,name)
    img = cv2.imread(img_name)
    mask = cv2.imread(mask_name)
    mask = mask[...,0] + mask[...,1] + mask[...,2]
    mask[mask>0] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)

    # 确定前景
    forground = cv2.erode(mask,kernel_big)
    # 确定可能区域
    prob_forground = cv2.dilate(mask,kernel_big)
    prob_forground = prob_forground - forground

    mask[forground > 0] = 1
    mask[prob_forground > 0] = 3


    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask = mask[:, :, np.newaxis]*255
    mask = np.repeat(mask, 3, 2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    # img = np.concatenate([img,mask],axis=1)

    # img = cv2.resize(img, (int(img.shape[1]*2/3),int(img.shape[0]*2/3)), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a',img)
    # cv2.waitKey(1)
    cv2.imwrite(os.path.join(save_path,name),mask)
    print(index*1.0/len(name_all))
cv2.destroyAllWindows()