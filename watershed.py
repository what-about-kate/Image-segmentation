# First we load image, convert it to grayscale and threshold it with a suitable value 
# Taken Otsu's binarization, so it would find the best threshold value
import cv2
import numpy as np

img = cv2.imread('mashrooms.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Now we have to create the marker
# Marker is the image with same size as the original one

# Foreground region:
# Erode main objects a little bit to be sure remaining region belongs to foreground
fg = cv2.erode(thresh, None, iterations = 2)

# Background region:
# Dilate the thresholded image so that background region is reduced
# But we are sure remaining black region is 100% background. Set it to 128
bgt = cv2.dilate(thresh, None, iterations = 3)
ret,bg = cv2.threshold(bgt, 1, 128, 1)

# Now we add both fg and bg
marker = cv2.add(fg, bg)

# Convert it into 32SC1
marker32 = np.int32(marker)

# Finally apply watershed
cv2.watershed(img, marker32)

# Convert result back into uint8 image
m = cv2.convertScaleAbs(marker32)

#ret,thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#res = cv2.bitwise_and(img, img, mask = thresh)

# Save the result
cv2.imwrite('mashrooms_watershed.jpg', m)

# https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
