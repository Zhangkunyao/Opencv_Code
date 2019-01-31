import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage

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

delt = 1000
path_img = '/home/kun/Documents/DataSet/video_06/img'
path_pose = '/home/kun/Documents/DataSet/video_06/DensePose'
test_pose = '/media/kun/Dataset/Pose/DataSet/new_data/video_27/DensePose'
_,name_all = Get_List(path_pose)
name_all.sort()
_,name_test = Get_List(test_pose)
name_test.sort()
# 初始化
name_img = name_all[800][:-8] + '.png'
img_path = os.path.join(path_img, name_img)
pose_path = os.path.join(path_pose, name_all[800])
iuv_map, uv_map_all = init(img_path, pose_path, delt)

# 尝试使用delt长的时间去合成一张完整的uv展开图
# 选取平滑滤波的方式，每次更新一张图片，最后一张图片的信息全部抹去，替换成倒数第二张图片的信息。
index_test = 0
for name_index in range(800+delt,len(name_all)):
    if name_index<3500:
        result = cv2.resize(iuv_map,(int(iuv_map.shape[1]*2/3),int(iuv_map.shape[0]*2/3)),interpolation=cv2.INTER_CUBIC)
        iuv_index = name_index - delt
        # 新的图片
        name_img = name_all[iuv_index][:-8] + '.png'
        img = cv2.imread(os.path.join(path_img, name_img))
        pose = cv2.imread(os.path.join(path_pose, name_all[iuv_index]))
        uv_map_new = ImageToIUV(img, pose)
        # 删除旧图
        uv_map_old = uv_map_all[-1]
        x, y = np.where((uv_map_old[..., 0] + uv_map_old[..., 1] + uv_map_old[..., 2]) != 0)
        iuv_map[x, y, ...] = 0
        uv_map_old = uv_map_all[-2]
        x, y = np.where((uv_map_old[..., 0] + uv_map_old[..., 1] + uv_map_old[..., 2]) != 0)
        iuv_map[x, y, ...] = uv_map_old[x, y, ...]
        # 刷新列表
        refresh(uv_map_all, uv_map_new)
        # 刷新新图
        x, y = np.where((iuv_map[..., 0] + iuv_map[..., 1] + iuv_map[..., 2]) == 0)
        tmp = np.zeros((1200, 800)).astype(np.uint8)
        tmp[x, y, ...] = 1
        uv_map_new[..., 0] = uv_map_new[..., 0] * tmp
        uv_map_new[..., 1] = uv_map_new[..., 1] * tmp
        uv_map_new[..., 2] = uv_map_new[..., 2] * tmp
        x, y = np.where((uv_map_new[..., 0] + uv_map_new[..., 1] + uv_map_new[..., 2]) != 0)
        iuv_map[x, y, ...] = uv_map_new[x, y, ...]
    else:
        pose = cv2.imread(os.path.join(test_pose, name_test[index_test]))
        result = IUVToImage(iuv_map,pose)
        x, y = np.where((result[..., 0] + result[..., 1] + result[..., 2]) == 0)
        result[x,y,1]=255
        index_test += 1
    cv2.imshow('a', result)
    cv2.waitKey(1)
    print(name_index)