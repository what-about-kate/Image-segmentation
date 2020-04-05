# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('kitten.jpg',0)

# Initiate STAR detector
#orb = cv2.ORB()

# find the keypoints with ORB
#kp = orb.detect(img)

# compute the descriptors with ORB
#kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()


img2 = cv2.imread('coins2.jpg',0)
orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
#orb = cv2.ORB_create()
kp2 = orb.detect(img2)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), \
        flags=cv2.DrawMatchesFlags_DEFAULT)

#plt.figure()
plt.imshow(img2_kp)
plt.show()
