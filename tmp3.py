# -*-coding:utf8-*-#
__author__ = 'play4fun'
"""
create time:15-10-25 下午12:20
实 上我是怎么做的呢 我们使用图像编  件打开 入图像 
添加一个 图层 
使用笔刷工具在  的地方使用白色绘制 比如头发  子 球等 
 使 用 色笔刷在不  的地方绘制 比如 logo 草地等 。
 然后将其他地方用灰 色填充 保存成新的掩码图像。
 在 OpenCV 中导入 个掩模图像 根据新的 掩码图像对原来的掩模图像  编 。
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# GCD_BGD（=0），背景；
#
# GCD_FGD（=1），前景；
#
# GCD_PR_BGD（=2），可能是背景；
#
# GCD_PR_FGD（=3），可能是前景。
kernel_small = np.ones((5,5),np.uint8)
kernel_big = np.ones((50,50),np.uint8)
img = cv2.imread('./video/video_06.png')
mask = cv2.imread('./video/video_06_IUV.png')
mask = mask[...,0] + mask[...,1] + mask[...,2]
mask[mask>0] = 1
mask = cv2.dilate(mask,kernel_small)

# 确定前景
forground = cv2.erode(mask,kernel_big)
# 确定前景
prob_forground = cv2.dilate(mask,kernel_big)
prob_forground = prob_forground - forground

mask[forground > 0] = 1
mask[prob_forground > 0] = 3
rect = (142,10,804,1060)

# 建立背景模型和前景模型，大小为1*65
bgdModel = np.zeros((1, 65), np.float64)

fgdModel = np.zeros((1, 65), np.float64)


# 把蒙版中白色地方置为1，作为确定前景。黑色地方置为0，作为确定背景

cv2.grabCut(img, mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)

# 把2变为0，把3变为1

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 将蒙版与原图做点对点乘积

img = img * mask[:, :, np.newaxis]

img2 = cv2.resize(img, (int(img.shape[1]*2/3),int(img.shape[0]*2/3)), interpolation=cv2.INTER_CUBIC)
cv2.imshow('a',img2)
cv2.waitKey(0)

cv2.destroyAllWindows()