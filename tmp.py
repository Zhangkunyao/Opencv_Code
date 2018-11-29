from skimage import io,color
import numpy as np
import cv2
from matplotlib import pyplot as plt

kernel = np.ones((5,5),np.uint8)
img = io.imread('./video/video.png')
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("img", closing)
cv2.waitKey(0)

