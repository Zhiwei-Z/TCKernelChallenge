import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pics/DH134_E6_P1_24Hours.jpeg')
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()
