import cv2
import numpy as np
import time
def existence_mask(frame, threshhold):
    resized = cv2.resize(frame, (60,60))
    #lur =  cv2.GaussianBlur(resized,(21, 21),0)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    target = np.uint8([[[25,77,249]]])
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    #print(hsv_target)
    low = hsv_target[0][0][0] - 10
    upper = hsv_target[0][0][0] + 10
    lower_bound = np.array([low, 219, 239])
    upper_bound = np.array([upper, 239, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    blue_points = np.count_nonzero(mask)
    if blue_points/24300 >= threshhold:
        return 1
    return 0

#Alternative, using contour

def existence_contour(frame, threshhold):
    resized = cv2.resize(frame, (60,60))
    # blur =  cv2.GaussianBlur(resized,(21, 21),0)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    target = np.uint8([[[25,77,249]]])
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    #print(hsv_target)
    low = hsv_target[0][0][0] - 10
    upper = hsv_target[0][0][0] + 10
    lower_bound = np.array([low, 219, 239])
    upper_bound = np.array([upper, 239, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) >= threshhold:
        return 1
    return 0

cap1 = cv2.VideoCapture(0) #camera 1
cap2 = cv2.VideoCapture(1) #camera 2
cap3 = cv2.VideoCapture(2) #camera 3
cap4 = cv2.VideoCapture(3) #camera 4
cap5 = cv2.VideoCapture(4) #camera 5

cap1.set(cv2.CAP_PROP_FPS, 30) #camera1's fps
cap2.set(cv2.CAP_PROP_FPS, 30) #camera2's fps
cap3.set(cv2.CAP_PROP_FPS, 30) #camera3's fps
cap4.set(cv2.CAP_PROP_FPS, 30) #camera4's fps
cap5.set(cv2.CAP_PROP_FPS, 30) #camera5's fps

n = 1
running_sum = [0,0,0,0,0]
result = [0,0,0,0,0]
while(True):
    # Capture camera1 frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    ret5, frame5 = cap5.read()
    n += 1
    running_sum[0] += existence_contour(frame1, 4);
    running_sum[1] += existence_contour(frame2, 4);
    running_sum[2] += existence_contour(frame3, 4);
    running_sum[3] += existence_contour(frame4, 4);
    running_sum[4] += existence_contour(frame5, 4);
    if n == 10:
        for i in range(5):
            if running_sum[i] >=6:
                print("Camera%d has detected an enemy"%i)
            else:
                print("Camera%d is safe"%i)
            result[i] = (running_sum[i] + 5)//10
    #write result to txt if needed
    # Display the resulting frame
    # cv2.imshow('frame1',frame)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()