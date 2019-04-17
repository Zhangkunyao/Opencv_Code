# coding=utf-8
import json
import os
import cv2
import numpy as np
import sys

'''
max min
判断在地上的坐标才是有效的
最大值的y坐标表示距离图片底部最近的距离
最小值通过对小于中心值的点做聚类并且与最大值到中心的距离相等。比例缩放和平移
计算 source ankle的平均值、最大最小值，
ankle positions
对最大和最小的脚踝位置计算高度，
'''
list_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder",
             "LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip",
             "LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe",
             "LSmallToe","LHeel","RBigToe","RSmallToe","RHeel","Background"]

middle_Candidate = ['MidHip','RHip','LHip','LKnee','RKnee']
low_Candidate = ['LAnkle','RAnkle','LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']
top_Candidate = ['Nose','REye','LEye','REar','LEar']
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
kernel_smal = np.ones((5,5),np.uint8)
def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    return [dirList, fileList]

def get_pose(img,pose_img):
    save_path = os.path.join(path_root,'tmp.png')
    if os.path.exists(save_path):
        os.remove(save_path)
    mask = (pose_img - img)
    mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smal)
    mask = 1 * (mask > 0)

    img_copy = img.copy()
    pose_img[:, :, 0] = pose_img[:, :, 0] * mask
    pose_img[:, :, 1] = pose_img[:, :, 1] * mask
    pose_img[:, :, 2] = pose_img[:, :, 2] * mask
    img_copy[:, :, 0] = img_copy[:, :, 0] * mask
    img_copy[:, :, 1] = img_copy[:, :, 1] * mask
    img_copy[:, :, 2] = img_copy[:, :, 2] * mask
    pose = (pose_img - 0.5 * img_copy)*1.5
    pose[pose>255]=255
    pose.astype(np.uint8)

    pose = cv2.resize(pose, (pose.shape[1], pose.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, pose)
    return cv2.imread(save_path)

def expend(img,size):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))  # 十字形结构
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Lowerwhite = np.array([0, 0, 100])
    Upperwhite = np.array([180, 200, 255])
    mask = cv2.inRange(HSV, Lowerwhite, Upperwhite)  # 提取彩色部分
    mask = mask[..., np.newaxis]
    mask = np.repeat(mask, 3, axis=2)

    WhiteThings = cv2.bitwise_and(img, mask)
    Clour = cv2.bitwise_and(img, 255 - mask)
    Clour = cv2.dilate(Clour, kernel)
    Clour = cv2.bitwise_and(Clour, 255 - mask)

    Clour[..., 0] = Clour[..., 0] + WhiteThings[..., 0]
    Clour[..., 1] = Clour[..., 1] + WhiteThings[..., 1]
    Clour[..., 2] = Clour[..., 2] + WhiteThings[..., 2]
    return Clour

def text_save(filename, data):
    file = open(filename,'a')
    file.write(str(data[0]))
    file.write('\t')
    file.write(str(data[1]))
    file.write('\n')
    file.close()

path_root = '/media/kun/Dataset/Pose/test_data/tmp/openpose/'#sys.argv[1]
body_path = os.path.join(path_root,"body")
face_path = os.path.join(path_root,'face')
hand_path = os.path.join(path_root,'hand')
save_pose_path = os.path.join(path_root, 'expend')
img_root = os.path.join(path_root, 'img')
# 文件路径确定
_,body_name_list = Get_List(body_path)
body_name_list.sort()
_,face_name_list = Get_List(face_path)
face_name_list.sort()
_,hand_name_list = Get_List(hand_path)
hand_name_list.sort()

_,aredy_save=Get_List(save_pose_path)
aredy_save.sort()
# 视频读取
_,img_name_all = Get_List(img_root)
img_name_all.sort()
for name_index in range(len(img_name_all)):
    if body_name_list[name_index] in aredy_save:
        print('have')
        continue
    body_name = os.path.join(body_path, body_name_list[name_index])
    face_name = os.path.join(face_path, face_name_list[name_index])
    hand_name = os.path.join(hand_path, hand_name_list[name_index])
    img_name = os.path.join(img_root, img_name_all[name_index])
    # 图像部分
    body_img = cv2.imread(body_name)
    # h,w,_ = body_img.shape
    # face_img = cv2.resize(cv2.imread(face_name), (w,h), interpolation=cv2.INTER_CUBIC)
    # hand_img = cv2.resize(cv2.imread(hand_name), (w,h), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(cv2.imread(img_name), (w,h), interpolation=cv2.INTER_CUBIC)
    face_img = cv2.imread(face_name)
    hand_img = cv2.imread(hand_name)
    img = cv2.imread(img_name)
    all = [body_img.shape,face_img.shape,hand_img.shape,img.shape]
    if len(set(all)) != 1:
        print('error')
        out = np.zeros(img.shape).astype(np.uint8)
        cv2.imwrite(os.path.join(save_pose_path, body_name_list[name_index]), out,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        continue
    pose = get_pose(img, body_img)
    # cv2.imshow('a',pose)
    # cv2.waitKey(0)
    hand = get_pose(img, hand_img)
    face = get_pose(img, face_img)
    expend_img = expend(face, 10)

    out = expend_img + (hand-pose)
    cv2.imwrite(os.path.join(save_pose_path,body_name_list[name_index]), out,[int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    print(1.0*name_index/len(face_name_list))

print("finish video")
print("finish all")





