import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage
#
# pose = cv2.imread('./video/video_06_IUV.png')
# img = cv2.imread('./video/video_06.png')
# iuv_map = ImageToIUV(img,pose)
# path = '/home/kun/Documents/DataSet/my/DensePose'
# _,name_all = Get_List(path)
# name_all.sort()
# for name in name_all:
#     pose = cv2.imread(os.path.join(path,name))
#     pose = np.array(pose)
#     # pose.transpose([1,0,2])
#     out = IUVToImage(iuv_map,pose)
#     # out = cv2.resize(out, (int(iuv_map.shape[1]/2),int(iuv_map.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('out',out)
#     cv2.waitKey(1)
#
# U = pose[:, :, 1]
# V = pose[:, :, 2]
# I = pose[:, :, 0]
#
# result = np.zeros(pose.shape).astype(np.uint8)
# x,y = np.where(I==2)
# result[x,y,0] = 2
#
# tmp_u = U[x,y]
# tmp_v = V[x,y]
#
# point = [(i,j) for i,j in zip(tmp_u,tmp_v)]
# point_set = set(point)
#
# for data in point_set:
#     index_data = point.index(data)
#     result[x[index_data],y[index_data],1] =data[0]
#     result[x[index_data], y[index_data], 2] =data[1]
# path_1 = '/media/kun/Dataset/Pose/DataSet/new_data/bilibili_3/DensePoseProcess/normal_result/bilibili_3_000000000997_rendered.png'
# path_2 = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/normal_result/机械哥_bilibili_000000001461_rendered.png'
# img_1 = cv2.imread(path_1)
# tmp = img_1[...,1]
# x,y = np.where(tmp>253)
# img_1[x,y,1] = 0
# img_2 = cv2.imread(path_2)
# tmp = img_2[...,1]
# x,y = np.where(tmp>253)
# img_2[x,y,1] = 0
# out = np.concatenate([img_1,img_2],1)
# out = cv2.resize(out, (int(out.shape[1]/2),int(out.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('a',out)
# cv2.waitKey(0)

# video_path = '/home/kun/Documents/video_3.flv'
# img_path = '/home/kun/Documents/tmp_meitu_2.jpg'
# img_base = cv2.imread(img_path)
# cap0 = cv2.VideoCapture(video_path)
# ret0, frame0 = cap0.read()
# img_base = cv2.resize(img_base, (int(frame0.shape[1]),int(frame0.shape[0])), interpolation=cv2.INTER_CUBIC)
#
# top_left = [500*2,806*2]
# bottom_right = [525*2,947*2]
# w = bottom_right[1] - top_left[1]
# h = bottom_right[0] - top_left[0]
# while cap0.isOpened():
#     ret0, frame0 = cap0.read()
#     frame0[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1],...] = img_base[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1],...]
#     frame0 = cv2.resize(frame0, (int(frame0.shape[1]/2),int(frame0.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('frame1', frame0)
#     key = cv2.waitKey(delay=1)
#     if key == ord("q"):
#         break
#     if key == ord(" "):
#         cv2.waitKey(delay=0)
#     # cv2.imwrite('/home/kun/Documents/tmp.png',frame0)
# # When everything done, release the capture
# cap0.release()
# cv2.destroyAllWindows()

root = '/media/kun/Dataset/Pose/DataSet/new_video/video_1/img'
_,img_all = Get_List(root)
img_all.sort()
for name in img_all:
    path = os.path.join(root,name)
    img = cv2.imread(path)
    cv2.imshow('a',cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC))
    key = cv2.waitKey(1)
    if key == ord(' '):
        print(name)