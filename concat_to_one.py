# -*- coding: utf-8 -*-
import cv2
import numpy as np
from basic_lib import Get_List
import random
import os

path_1 = '/media/kun/UbuntuData/Kun/GAN_Realation/kun/save_result/function3_correct/371/'
path_2 = '/media/kun/UbuntuData/Kun/GAN_Realation/kun/save_result/function3_g_max/'
path_3 = '/media/kun/Dataset/LSUN/bedroom_train_lmdb/img'
path_4 = '/media/kun/Dataset/GAN_Relation/save_result/function3_cifar_10/231'


path = path_3
_,name_all = Get_List(path,True)

img_all = []
for i in range(10):
    index = random.randint(0,len(name_all)-1)
    tmp = cv2.imread(os.path.join(path,name_all[index]))
    tmp = cv2.resize(tmp, (64, 64), interpolation=cv2.INTER_CUBIC)
    for j in range(1,11,1):
        index = random.randint(0, len(name_all) - 1)
        tmp2 = cv2.imread(os.path.join(path,name_all[index]))
        tmp2 = cv2.resize(tmp2, (64, 64), interpolation=cv2.INTER_CUBIC)
        tmp = np.concatenate((tmp,tmp2),1)
    img_all.append(tmp)
img = img_all[0]

for i in img_all[1:]:
    img = np.concatenate((img,i),0)
cv2.imshow('a',img)
cv2.waitKey(0)