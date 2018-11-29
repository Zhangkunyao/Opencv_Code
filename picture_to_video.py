import cv2
import os
from basic_lib import Get_List
from PIL import Image
import numpy as np

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
    return Image.fromarray(result)

img_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001/cut/img'
pose_root_path = '/media/kun/Dataset/Pose/DataSet/new_data/0001/cut/normal_result'
_,img_all = Get_List(img_root_path)
_,pose_all = Get_List(pose_root_path)

img_all.sort()
pose_all.sort()
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('测试_gp.avi', fourcc, fps, (512,256))
if not videoWriter.isOpened():
    print("video error")
    exit(0)

for i in range(len(img_all)):
    img = cv2.imread(os.path.join(img_root_path, img_all[i]))
    pose = cv2.imread(os.path.join(pose_root_path, pose_all[i]))
    img = img_process(img,256)
    pose = img_process(pose, 256)
    img = np.concatenate([pose, img], axis=1)
    videoWriter.write(img)
    print(1.0*i/len(img_all))
videoWriter.release()
print('finish')


