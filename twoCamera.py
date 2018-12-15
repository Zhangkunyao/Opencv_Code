# -*- coding: utf-8 -*-
# @Time    : 2017/8/15 00:19
# @Author  : play4fun
# @File    : two_camera.py
# @Software: PyCharm

"""
two_camera.py:
"""

import cv2
import numpy as np

cap0 = cv2.VideoCapture('/home/kun/Documents/DataSet/video/Densepose/wshp.avi')
cap1 = cv2.VideoCapture('/home/kun/Documents/DataSet/video/Densepose/densepose.avi')
# ret = cap0.set(3, 320)
# ret = cap0.set(4, 240)
# ret = cap1.set(3, 320)
# ret = cap1.set(4, 240)
index = 0
while cap0.isOpened() and cap1.isOpened():
    if index<50:
        ret0, frame0 = cap0.read()
        index = index+1
        continue
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = frame0[:,256:,:]
    frame1 = frame1[:,256:,:]
    # frame0 = cv2.resize(frame0, (320,240),interpolation=cv2.INTER_CUBIC)
    # frame1 = cv2.resize(frame1, (320,240), interpolation=cv2.INTER_CUBIC)
    out = np.concatenate([frame0,frame1],axis=1)
    # if ret0:
    #     cv2.imshow('frame0', frame0)
    #     cv2.setWindowTitle('frame0','On Top')
    # if ret1:
    cv2.imshow('frame1', out)
    # cv2.moveWindow('frame1', x=frame0.shape[1], y=0)
    # cv2.moveWindow('frame1', x=0, y=0)

    key = cv2.waitKey(delay=10)
    if key == ord("q"):
        break
    if key == ord(" "):
        cv2.waitKey(delay=0)

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
