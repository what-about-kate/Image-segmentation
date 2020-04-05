import cv2

image = cv2.imread('coins2.jpg')
spatialRadius = 35
colorRadius = 60
pyramidLevels = 3
res = cv2.pyrMeanShiftFiltering(image, spatialRadius, colorRadius, pyramidLevels)
cv2.imwrite('meanshift_coins2.jpg', res)
