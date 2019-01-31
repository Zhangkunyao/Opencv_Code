import cv2
import os
from basic_lib import Get_List
from PIL import Image
import numpy as np

# 剪切图片生成video 为openpose做准备
def img_process(img,loadsize):
    try :
        h, w ,_= img.shape
    except:
        print("hah")
    result = np.zeros((loadsize,loadsize,3))
    if h >= w:
        w = int(w*loadsize/h)
        h = loadsize
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - w)/2)
        img = np.array(img)
        result[0:h,bias:bias+w,...] = img[0:h,0:w,...]
    if w > h:
        h = int(h * loadsize / w)
        w = loadsize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        bias = int((loadsize - h)/2)
        img = np.array(img)
        result[bias:bias+h,0:w,...] = img[0:h,0:w,...]
    result = result.astype(np.uint8)
    return result

def get_all_loc(file_path):
    file = open(file_path, 'r')
    listall = file.readlines()
    listall = [i.rstrip('\n').split('\t')[:-1] for i in listall]
    for i in range(len(listall)):
        for j in range(len(listall[i])):
            listall[i][j] = int(listall[i][j])
    file.close()
    return listall

target_img_path = '/media/kun/Dataset/Pose/DataSet/new_data/video_06/back_ground.png'
target_img = Image.open(target_img_path).convert('RGB')
size_target = target_img.size

img_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/img'
name_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/img'
txt_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/DensePoseProcess/loc.txt'
loc_all_source = get_all_loc(txt_root_path)

_,name_all = Get_List(name_path)
name_all.sort()

# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# videoWriter = cv2.VideoWriter('/media/kun/Dataset/Pose/DataSet/new_data/机械哥_bilibili/机械哥_bilibili_cut.avi',
#                               fourcc, fps, (size_target[0],size_target[0]))
# if not videoWriter.isOpened():
#     print("video error")
#     exit(0)
img_last = None
for i in range(len(name_all)):
    img_path = os.path.join(img_root_path,name_all[i])
    img = cv2.imread(img_path)
    # 定位
    tmp = loc_all_source[i]
    point = {'xmin': tmp[0], 'xmax': tmp[1], 'ymin': tmp[2], 'ymax': tmp[3]}
    w = point['xmax'] - point['xmin']
    h = point['ymax'] - point['ymin']
    xmin = point['xmin']
    xmax = point['xmax']
    ymin = point['ymin']
    ymax = point['ymax']

    if xmax>xmin and ymax>ymin:
        img = img[ymin:ymax, xmin:xmax, ...]
    img = img_process(img,size_target[0])
    img = cv2.resize(img, (int(img.shape[1]*2/3), int(img.shape[0]*2/3)), interpolation=cv2.INTER_CUBIC)
    if img_last is None:
        img_last = img
    print(1.0*i/len(name_all))
    cv2.imshow('a',img-img_last)
    cv2.waitKey(1)
    img_last = img
# videoWriter.release()
print('finish')


