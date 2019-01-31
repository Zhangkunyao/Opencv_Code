import os
import numpy as np
import cv2
from basic_lib import Get_List,ImageToIUV,IUVToImage

kernel = np.ones((5,5),np.uint8)
# 下一张图片的IUV_map是前面所有的集合。
path_img = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/img'
path_pose = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/org'
save_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/DensePoseProcess/IUV_map'
_,name_all = Get_List(path_pose)
name_all.sort()
# _,name_test = Get_List(test_pose)
# name_test.sort()
# 初始化
img_path = os.path.join(path_img, name_all[800])
pose_path = os.path.join(path_pose, name_all[800])

img = cv2.imread(img_path)
pose = cv2.imread(pose_path)
iuv_map = ImageToIUV(img, pose)

# 尝试使用delt长的时间去合成一张完整的uv展开图
# 选取平滑滤波的方式，每次更新一张图片，最后一张图片的信息全部抹去，替换成倒数第二张图片的信息。
index_test = 0
for name_index in range(1,len(name_all)):
    # 新的图片
    img = cv2.imread(os.path.join(path_img, name_all[name_index-1]))
    pose = cv2.imread(os.path.join(path_pose, name_all[name_index-1]))
    uv_map_new = ImageToIUV(img, pose)
    # 更新
    x, y = np.where((uv_map_new[..., 0] + uv_map_new[..., 1] + uv_map_new[..., 2]) != 0)
    iuv_map[x, y, ...] = uv_map_new[x, y, ...]

    pose_next = cv2.imread(os.path.join(path_pose, name_all[name_index]))
    result = IUVToImage(iuv_map,pose_next)

    pose_next[pose_next > 0] = 255
    pose_next = cv2.morphologyEx(pose_next, cv2.MORPH_CLOSE, kernel)
    x, y = np.where((pose_next[..., 0] + pose_next[..., 1] + pose_next[..., 2]) == 0)
    result[x,y,1]=255
    # result = cv2.resize(result, (int(result.shape[1] * 2 / 3), int(result.shape[0] * 2 / 3)),
    #                    interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a', result)
    # cv2.waitKey(1)
    save_name = os.path.join(save_path,name_all[name_index])
    # cv2.imwrite(save_name,result,[int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(name_index)