# -*- coding: utf-8 -*-

import cv2
import numpy as np
import cv2
import os
from basic_lib import Get_List

video_path = '/media/kun/Dataset/Pose/DataSet/result_hand_face/video/0001.flv'
video_save = '/media/kun/Dataset/Pose/DataSet/result_hand_face/video/0001_cut.avi'

cap0 = cv2.VideoCapture(video_path)
ret0, frame0 = cap0.read()

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(video_save,fourcc, fps, (frame0.shape[1],frame0.shape[0]))
index = 0
star_flg = False
while cap0.isOpened():
    ret0, frame0 = cap0.read()

    if star_flg:
        videoWriter.write(frame0)
        frame0 = cv2.putText(frame0, "start write!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    key = cv2.waitKey(delay=1)
    if key == ord("q"):
        break
    if key == ord(" "):
        cv2.waitKey(delay=0)
    if key == ord("s"):
        star_flg=not star_flg

    cv2.imshow('frame1', frame0)
# When everything done, release the capture
cap0.release()
videoWriter.release()
cv2.destroyAllWindows()
