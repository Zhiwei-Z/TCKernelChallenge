import cv2
import sys
import numpy as np
from sklearn.cluster import KMeans



tracker = cv2.TrackerCSRT_create()

img = cv2.imread('pics/DH134_E6_P1_24Hours.jpeg')
height = np.size(img, 0)
width = np.size(img, 1)
img = cv2.resize(img, (width//9, height//9))  
bbox = cv2.selectROI(img, False)

ok = tracker.init(img, bbox)

main_circle = img[int(bbox[1]):int(bbox[1] + bbox[3]),int(bbox[0]):int(bbox[0] + bbox[2])]


img_revised = cv2.cvtColor(main_circle, cv2.COLOR_BGR2RGB)

color_set = {'yellow1':[196, 163, 94], 'yellow2':[166, 137, 79], 'yellow3':[162,152,115], 'yellow4':[143, 138,106], 'dark_yellow':[104, 102,81], 'brown':[81, 47, 45], 'brown1':[93, 62,51], 'brown2':[67, 39, 36], 'brown3': [49, 21, 17]}

#applying mask to get points
for color in color_set:
    hsv = cv2.cvtColor(img_revised, cv2.COLOR_BGR2HSV)
    target = np.uint8([[color_set[color]]])
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    color_range = hsv_target[0][0]
    lower_bound = np.array([color_range[0] - 10, 100 , 100])
    upper_bound = np.array([color_range[0] + 10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
num = 1
for contour in contours:
    left = min(contour, key = lambda x: x[0][0])[0][0]
    right = max(contour, key = lambda x:x[0][0])[0][0]
    top = min(contour, key = lambda x: x[0][1])[0][1]
    bot = max(contour, key = lambda x: x[0][1])[0][1]
    if (right - left) * (bot - top) > 35:
        current_crop = main_circle[top - 3:bot + 3, left - 3:right + 3]
        cv2.imwrite('secondday/secondday%d.png' % num, current_crop)
        num += 1

    # #k-means grouping
    # X = np.array(list(zip(blue_points_j, blue_points_i)))

    # # Number of clusters
    # kmeans = KMeans(n_clusters=2)
    # # Fitting the input data
    # kmeans = kmeans.fit(X)
    # # Getting the cluster labels
    # labels = kmeans.predict(X)
    # # Centroid values
    # centroids = kmeans.cluster_centers_

    
    # print(centroids) # From sci-kit learn
    
    # for i in centroids:
    #     cv2.circle(frame,(int(i[0] + bbox[0]), int(i[1] + bbox[1] + bbox[3]/2)), 10,(255, 0, 0), -1)
    # # Display result
    # #out.write(frame);
cv2.imshow("Tracking", main_circle)

# Exit if ESC pressed
k = cv2.waitKey(1000000000) & 0xff
if k == 27 : cv2.destroyAllWindows()
#out.release()


