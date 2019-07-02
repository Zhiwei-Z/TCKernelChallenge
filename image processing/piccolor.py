import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('armor/armorplate0.jpg')
b,g,r = cv2.split(img)
img_revised = cv2.merge([r,g,b])
color_set = {'blue':[6, 83, 237]}

for color in color_set:
    hsv = cv2.cvtColor(img_revised, cv2.COLOR_BGR2HSV)
    target = np.uint8([[color_set[color]]])
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    low = hsv_target[0][0][0] - 10
    upper = hsv_target[0][0][0] + 10
    lower_bound = np.array([low, 50, 50])
    upper_bound = np.array([upper, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    img_revised[mask != 0] = [0, 0, 0]
print lower_bound
print upper_bound
    
plt.subplot(122);plt.imshow(img_revised) # expect true color
plt.show()


#{red:253,100,103, orange:253,174,118, pink:250,130,169, blue:78,204,255}
'''
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
lower_red = np.array([110,50,50])
upper_red = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
img2[mask != 0] = [0, 0, 0]
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

target_red = np.uint8([[[253, 100, 103]]])
hsv_tred = cv2.cvtColor(target_red, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,50,50])
upper_red = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
img[mask != 0] = [255, 0, 0]
cv2.imshow("open_cv", img)
cv2.waitKey()


b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()

cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
