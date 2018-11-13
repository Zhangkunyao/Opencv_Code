from PIL import Image
import os
import numpy as np
from basic_lib import Get_List,mkdir


_,img_names = Get_List('/media/kun/Dataset/Pose/DataSet/0001/img')
_,pose_names = Get_List('/media/kun/Dataset/Pose/DataSet/0001/pose')

for i in range(len(img_names)):
    img_name = img_names[i]
    pose_name = pose_names[i]
    img_path = os.path.join('/media/kun/Dataset/Pose/DataSet/0001/img', img_name)
    pose_path = os.path.join('/media/kun/Dataset/Pose/DataSet/0001/pose',pose_name)

    img = np.array(Image.open(img_path))
    pose = np.array(Image.open(pose_path))

    img[700:1080, 0:479] = img[0:380, 0:479]
    pose[700:1080, 0:479] = pose[0:380, 0:479]
    # cv2.imshow('Image', img)
    # cv2.imshow('pose', pose)
    # cv2.waitKey(0)
    # # os.remove(pose_path)
    img = Image.fromarray(img)
    pose = Image.fromarray(pose)
    img.save(img_path)
    # # os.remove(img_path)
    pose.save(pose_path)
    print(i/len(img_names))
print('finished')

#
#
# # 加载图像并显示
# image = cv2.imread('./1929.jpg')
# cv2.imshow("Original", image)
#
# # # 第一次尝试把嘴的部位裁剪出来
# mouth = image[700:1080, 0:479]
#
# #
# # # 第二次尝试把嘴的部位裁剪出来
# # mouth = image[85:350, 285:420]
# # cv2.imshow("Mouth2", mouth)
# # cv2.waitKey(0)
# #
# # # 第三次尝试把嘴的部位裁剪出来
# # mouth = image[85:250, 85:220]
# cv2.imshow("cut", image)
# cv2.waitKey(0)

