from picture_to_video import Get_List
import cv2
import os
import numpy as np
# path_root = "../data/target_pose"
# _,img_names = Get_List(path_root)
# img_names.sort()

img = cv2.imread('./0001.png ')
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel)
out = np.concatenate([img,erosion],axis=1)
cv2.imshow(out)
cv2.waitKey(0)

# for i in range(len(img_names)):
#     img = cv2.imread(os.path.join(path_root, img_names[i]))
#     videoWriter.write(img)
#     print(1.0*i/len(img_names))