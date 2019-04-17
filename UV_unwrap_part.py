import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage
from PIL import Image
def refresh(lis,data):
    for i in range(len(lis)-1):
        lis[len(lis)-i-1] = lis[len(lis)-i-2]
    lis[0] = data
    return lis

def init(img_path,pose_path,delt):
    iuv_map = np.zeros((1200, 800, 3)).astype(np.uint8)
    img = cv2.imread(img_path)
    pose = cv2.imread(pose_path)
    uv_map_new = ImageToIUV(img, pose)
    uv_map_all = [uv_map_new for i in range(delt)]
    x_tmp, y_tmp = np.where((uv_map_new[..., 0] + uv_map_new[..., 1] + uv_map_new[..., 2]) != 0)
    iuv_map[x_tmp, y_tmp, ...] = uv_map_new[x_tmp, y_tmp, ...]
    return iuv_map,uv_map_all

# 利用t时刻和之前 10 帧时间的图像得到展开图
time_length = 10
path_img = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/img'
path_pose = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePose'
save_root = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/uv_unwrap_part'

_,name_saved = Get_List(save_root)
_,name_all = Get_List(path_pose)
name_all.sort()

# init
uv_map_past_all = []
uv_map_future_all = []
for i in range(int(time_length/2)):
    name_img = name_all[0][:-8] + '.png'
    img = cv2.imread(os.path.join(path_img, name_img))
    pose = cv2.imread(os.path.join(path_pose, name_all[0]))
    uv_map_past_all.append(ImageToIUV(img, pose))

    name_img = name_all[i][:-8] + '.png'
    img = cv2.imread(os.path.join(path_img, name_img))
    pose = cv2.imread(os.path.join(path_pose, name_all[i]))
    uv_map_future_all.append(ImageToIUV(img, pose))

basic_img = './video/video_06.png'
basic_pose = './video/video_06_IUV.png'
basic_back_img = './video/video_06_back.png'
basic_back_pose = './video/video_06_back_IUV.png'
basic_IUV = ImageToIUV(cv2.imread(basic_img),cv2.imread(basic_pose))
basic_IUV_back = ImageToIUV(cv2.imread(basic_back_img),cv2.imread(basic_back_pose))
tmp = basic_IUV[..., 0] + basic_IUV[..., 1] + basic_IUV[..., 2]
x,y = np.where((tmp) == 0)
basic_IUV[x,y,...] = basic_IUV_back[x,y,...]
uv_map_old = basic_IUV.copy()



index_test = 0
for name_index in range(len(name_all)):
    past_index = max(name_index-1,0)
    future_index = min(name_index+int(time_length/2),len(name_all)-1)
    if name_all[name_index] in name_saved:
        continue
    # 现在
    name_img = name_all[name_index][:-8] + '.png'
    img = cv2.imread(os.path.join(path_img, name_img))
    pose = cv2.imread(os.path.join(path_pose, name_all[name_index]))
    uv_map_now = ImageToIUV(img, pose)
    # a = Image.fromarray(uv_map_now).crop((0, 200, 200, 400))
    # a.show()
    # 将来
    name_img = name_all[future_index][:-8] + '.png'
    img = cv2.imread(os.path.join(path_img, name_img))
    pose = cv2.imread(os.path.join(path_pose, name_all[future_index]))
    uv_map_future = ImageToIUV(img, pose)


    tmp = uv_map_now.copy()
    for map_future,map_past in zip(uv_map_future_all,uv_map_past_all):
        x, y = np.where((uv_map_now[..., 0] + uv_map_now[..., 1] + uv_map_now[..., 2]) == 0)
        uv_map_now[x,y,...] = map_future[x,y,...]
        uv_map_now[x, y, ...] = map_past[x, y, ...]

    refresh(uv_map_past_all,tmp)
    refresh(uv_map_future_all, uv_map_future)
    x, y = np.where((uv_map_now[..., 0] + uv_map_now[..., 1] + uv_map_now[..., 2]) != 0)
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    x, y = np.where((uv_map_now[..., 0] + uv_map_now[..., 1] + uv_map_now[..., 2]) == 0)
    x[x > x_max] = 0
    x[x < x_min] = 0
    y[y > y_max] = 0
    y[y < y_min] = 0
    uv_map_now[x,y,...] = uv_map_old[x,y,...]
    uv_map_old = uv_map_now
    # uv_map_final[x,y,...] =  uv_map_now[x,y,...]
    # cv2.imshow('a', uv_map_now)
    # cv2.waitKey(1)
    cv2.imwrite(os.path.join(save_root, name_all[name_index]),uv_map_now)
    print(str(name_index*1.0/len(name_all)) + '    ' + str(name_index))