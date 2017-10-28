import cv2
import numpy as np

img = cv2.imread('skeleton.png',0)
# kernel = np.ones((7,7),np.uint8)
# img = cv2.medianBlur(img, 5)
img[img>50] = 255
cv2.imwrite('filtered.png', img)
kernel = np.asarray([
	[0,0,1,0,0],
	[0,1,1,1,0],
	[1,1,1,1,1],
	[0,1,1,1,0],
	[0,0,1,0,0]],dtype=np.uint8)
# kernel = np.asarray([
# 	[1,1,1],
# 	[1,1,1],
# 	[1,1,1]],dtype=np.uint8)
print(kernel)
dilation = cv2.dilate(img,kernel,iterations = 9)

cv2.imwrite('dilated.png',dilation)	
# cv2.imwrite('closing.png',output)