import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage
from PIL import Image
import shutil
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

# a = cv2.imread('./video/video_1.png')
# b = Image.fromarray(a[...,::-1])
# b.show()


# root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/IUV_map_video_1'
# _,img_all = Get_List(root)
# img_all.sort()
# for name in img_all:
#     path = os.path.join(root,name)
#     img = cv2.imread(path)
#     cv2.imshow('a',cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC))
#     key = cv2.waitKey(1)
#     if key == ord(' '):
#         print(name)

# img_root = '/home/kun/Documents/DataSet/video_06/DensePoseProcess/IUV_map/'
# _,name_all = Get_List(img_root)
# name_all.sort()
# # img = cv2.imread(os.path.join(img_root,name_all[0]))
# # print('a')
# # video_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/video_local.avi'
# # fps = 30
# # img_size = (img.shape[1], img.shape[0])
# # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
# # i=0
# for name in name_all:
#     img = cv2.imread(os.path.join(img_root,name))
#     cv2.imshow('a',img)
#     key = cv2.waitKey(delay=1)
#     if key == ord("q"):
#         break
#     if key == ord(" "):
#         print(name)
#         cv2.waitKey(delay=0)
#     img = cv2.resize(img, (1152,1080), interpolation=cv2.INTER_CUBIC)
#     videoWriter.write(img)
#     print(i/len(name_all))
#     i+=1
# videoWriter.release()
# path = '/media/kun/Dataset/Pose/DataSet/test_video/20190223/一分钟教你学会高尔夫-击地篇小贴士(V2调色降噪)[超清版]/哔哩哔哩-一分钟教你学会高尔夫-击地篇小贴士(V2调色降噪)[超清版].flv'
# cap0 = cv2.VideoCapture(path)
# # ret = cap0.set(3, 320)
# # ret = cap0.set(4, 240)
# # ret = cap1.set(3, 320)
# # ret = cap1.set(4, 240)
# # 900 1100
# index_star = 900
# index_end = 1100
# index = 0
# while cap0.isOpened():
#
#     ret0, frame0 = cap0.read()
#     # cv2.imshow('frame1', cv2.resize(frame0, (320,240),interpolation=cv2.INTER_CUBIC))
#     # key = cv2.waitKey(delay=5)
#     # if key == ord("q"):
#     #     break
#     # if key == ord(" "):
#     #     print(index)
#     #     cv2.waitKey(delay=0)
#     if index>index_star and index<index_end:
#         cv2.imwrite('/media/kun/Dataset/Pose/DataSet/test_video/test_img/j'+str(index)+'.jpg',frame0)
#     if index > index_end:
#         break
#     index+=1
#
#
# # When everything done, release the capture
# cap0.release()
# cv2.destroyAllWindows()

# def get_all_loc(file_path):
#     file = open(file_path, 'r')
#     listall = file.readlines()
#     listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
#     for i in range(len(listall)):
#         for j in range(len(listall[i])):
#             listall[i][j] = int(listall[i][j])
#     file.close()
#     return listall
#
# def text_save(filename, data):
#     file = open(filename,'a')
#     for i in data:
#         file.write(str(i))
#         file.write('\t')
#     file.write('\n')
#     file.close()
#
# path_root = '/media/kun/Dataset/Pose/test_data/'
# img_path = os.path.join(path_root,'img')
# pose_path = os.path.join(path_root,'DensePoseProcess','org')
# openpose_path = os.path.join(path_root,'openpose','expend')
# map_path = os.path.join(path_root,'DensePoseProcess','IUV_map')
# _,img_all = Get_List(img_path)
# _,pose_all = Get_List(pose_path)
# _,openpose_all = Get_List(openpose_path)
# _,map_all = Get_List(map_path)
#
# map_all.sort()
# pose_all.sort()
# img_all.sort()
# openpose_all.sort()
# body_loc = get_all_loc(os.path.join(path_root,'DensePoseProcess','body_loc.txt'))
# loc = get_all_loc(os.path.join(path_root,'DensePoseProcess','loc.txt'))
#
#
#
# save_img = os.path.join(path_root,'tmp','img')
# save_pose = os.path.join(path_root,'tmp','DensePoseProcess','org')
# save_map = os.path.join(path_root,'tmp','DensePoseProcess','IUV_map')
# save_openpose = os.path.join(path_root,'tmp','DensePoseProcess','expend')
# save_body_loc = os.path.join(path_root,'tmp','DensePoseProcess','body_loc.txt')
# save_loc = os.path.join(path_root,'tmp','DensePoseProcess','loc.txt')
#
# for index in range(len(img_all)):
#     if img_all[index][0] == 'i':
#         shutil.copy(os.path.join(img_path,img_all[index]), os.path.join(save_img,img_all[index]))
#         shutil.copy(os.path.join(pose_path, pose_all[index]), os.path.join(save_pose, pose_all[index]))
#         shutil.copy(os.path.join(openpose_path, openpose_all[index]), os.path.join(save_openpose, openpose_all[index]))
#         shutil.copy(os.path.join(map_path, map_all[index]), os.path.join(save_map, map_all[index]))
#         text_save(save_body_loc,body_loc[index])
#         text_save(save_loc, loc[index])

# img = cv2.imread('/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/texture_trip/video_3000000004059_IUV.png')
# mask = cv2.imread('/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/dense_mask/video_3000000004059_IUV.png')
# mask = mask[...,0]
#
# delt_x = int(img.shape[1]*1.01) - img.shape[1]
# delt_y = int(img.shape[0]*1.01) - img.shape[0]
#
# mask = cv2.copyMakeBorder(mask, delt_y, delt_y, delt_x,delt_x, cv2.BORDER_REPLICATE)
# x,y = np.where(mask>0)
# mask[x,y] = 255
# img = cv2.resize(img,(img.shape[1]+delt_x*2,img.shape[0]+delt_y*2))
# out = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
#
#
# cv2.imshow('mask',cv2.resize(mask,(mask.shape[1]//2,mask.shape[0]//2)))
# cv2.imshow('out',cv2.resize(out,(out.shape[1]//2,out.shape[0]//2)))
# cv2.imshow('img',cv2.resize(img,(img.shape[1]//2,img.shape[0]//2)))
# cv2.waitKey(0)

body = {'data':[1,2],'name':'body','value':20}
head = {'data':[23,24],'name':'head','value':40}
R_Arm = {'data':[3,16,18,20,22],'name':'R_Arm','value':60}
L_Arm = {'data':[4,15,17,19,21],'name':'L_Arm','value':80}
R_Leg = {'data':[6,9,13,7,11],'name':'R_Leg','value':100}
L_Leg = {'data':[5,10,14,8,12],'name':'L_Leg','value':120}

sub_part = [body,head,R_Arm,L_Arm,R_Leg,L_Leg]

def img_process_cv(img,flg_pose,load_size,flag_L=False):
    h, w, _ = img.shape
    if flag_L:
        result = np.zeros((load_size,load_size)).astype(np.uint8)
    else:
        result = np.zeros((load_size, load_size, 3)).astype(np.uint8)
    if not flg_pose:
        result[...,1] = 255
    if h >= w:
        w = int(w*load_size/h)
        h = load_size
        img = cv2.resize(img,(w,h),interpolation = cv2.INTER_AREA)
        bias = int((load_size - w)/2)
        result[0:h,bias:bias+w,...] = img[0:h,0:w,...]
    if w > h:
        h = int(h * load_size / w)
        w = load_size
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
        bias = int((load_size - h)/2)
        result[bias:bias + h, 0:w, ...] = img[0:h, 0:w, ...]
    return result

def img_process_trans(img, org_size):
    roi_region = None
    loadsize = img.shape[0]
    w = org_size[0]
    h = org_size[1]
    if h >= w:
        w = int(w * loadsize / h)
        h = loadsize
        bias = int((loadsize - w) / 2)
        img = np.array(img)
        roi_region = img[0:h, bias:bias + w, ...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        bias = int((loadsize - h) / 2)
        img = np.array(img)
        roi_region = img[bias:bias + h, 0:w, ...]
    w = org_size[0]
    h = org_size[1]
    return cv2.resize(roi_region, (w, h), interpolation = cv2.INTER_AREA)

img = cv2.imread('/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/texture_trip/video_3000000004059_IUV.png')
mask = cv2.imread('/media/kun/Dataset/Pose/DataSet/data_rebuilt/video_3/DensePoseProcess/dense_mask/video_3000000004059_IUV.png')

mask = mask[...,0]
tmp = img*0
tmp[...,1] = 255

# cv2.imshow('img',img)
# cv2.imshow('mask',mask)
# cv2.waitKey(0)

final_out = img*0
final_out[...,1] = 255
x,y = np.where(mask == 0)
img[x,y] = 0

out_all = {}
for part in sub_part:
    out = img * 0
    out[...,1] = 255
    x,y = np.where(mask == part['value'])
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    out[x,y] = img[x,y]
    out = out[xmin:xmax, ymin:ymax]
    out = img_process_cv(out, flg_pose=False, load_size=128)
    out_all[part['name']] = out
    out_all[part['name']+'_coor'] = [xmin,xmax,ymin,ymax]

cut_boundry_x = int(img.shape[0]*1.05) - img.shape[0]
cut_boundry_x = cut_boundry_x if cut_boundry_x%2 == 0 else cut_boundry_x+1

cut_boundry_y = int(img.shape[1]*1.05) - img.shape[1]
cut_boundry_y = cut_boundry_y if cut_boundry_y % 2 == 0 else cut_boundry_y + 1


for part in sub_part:
    coor = out_all[part['name']+'_coor']
    xmin = coor[0]
    xmax = coor[1]
    ymin = coor[2]
    ymax = coor[3]

    sub_out = out_all[part['name']]
    sub_out = img_process_trans(sub_out,(ymax-ymin+cut_boundry_x,xmax-xmin+cut_boundry_y))
    # cv2.imshow('sub_out1', sub_out)

    bigger_mask = mask[xmin:xmax, ymin:ymax]*0
    x, y = np.where(mask[xmin:xmax, ymin:ymax] == part['value'])
    bigger_mask[x,y] = 255

    bigger_mask = cv2.copyMakeBorder(bigger_mask, cut_boundry_y//2, cut_boundry_y//2, cut_boundry_x//2, cut_boundry_x//2, cv2.BORDER_CONSTANT,0)
    x, y = np.where(bigger_mask >0)
    final_out[x+xmin - cut_boundry_x//2,y + ymin - cut_boundry_y//2] = sub_out[x, y]


cv2.imshow('final_out',cv2.resize(final_out,(final_out.shape[1]//2,final_out.shape[0]//2)))
cv2.waitKey(0)