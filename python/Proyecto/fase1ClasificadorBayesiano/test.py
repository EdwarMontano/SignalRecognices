import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    lower_green = np.array([60, 50, 50])
    upper_green = np.array([90, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    lower_red_dark = np.array([0,100,100])
    upper_red_dark = np.array([10,255,255])
    lower_red_light = np.array([170,100,100])
    upper_red_light = np.array([180,255,255])
    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # mask = cv2.inRange(hsv, lower_red_light, upper_red_light)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    mean = np.mean(res)
    std = np.std(res)
    h, s, v = cv2.split(res)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR_FULL)
    b,g,r = cv2.split(bgr)
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_std = np.std(b)
    g_std = np.std(g)
    r_std = np.std(r)
    mask_mean = np.mean(mask)
    mask_std = np.std(mask)
    print(mask.shape)
    print(mask_mean,mask_std,sep="----")
    # print(b.shape)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('b',b)
    cv2.imshow('g',g)
    cv2.imshow('r',r)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()