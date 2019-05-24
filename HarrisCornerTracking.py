import cv2
import numpy as np
import time

corner_track_params = dict(blockSize = 2, 
                           ksize = 3, 
                           k = 0.04)

lk_params = dict(winSize = (15, 15), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def get_corners(frame_gray):
    dst = cv2.cornerHarris(frame_gray, **corner_track_params)
    dst = cv2.dilate(dst, None)
    ret, dst1 = cv2.threshold(dst,0.9*dst.max(),255,0)
    dst1 = np.uint8(dst1)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(frame_gray, np.float32(centroids), (5,5), (-1,-1), criteria)
    corners = corners.reshape(-1, 1, 2)
    return corners

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
prev_corners = get_corners(prev_frame_gray)
mask = np.zeros_like(prev_frame)
count = 0
while cap.isOpened():
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    next_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, prev_corners, None, **lk_params)
    
    
    for i, (new, prev) in enumerate(zip(next_corners, prev_corners)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)
        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)
        
    img = cv2.add(frame, mask)
    #time.sleep(1/45)  used for normal videos, not for web camera streaming
    count += 1
    cv2.imshow('Tracking Corners', img)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break
    elif k == ord('c'):
        mask = np.zeros_like(frame)
        prev_corners = next_corners.copy()
        prev_frame_gray = frame_gray.copy()
    if count == 10000:
        ret, prev_frame = cap.read()
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_corners = get_corners(prev_frame_gray)
        mask = np.zeros_like(prev_frame)
    else:
        prev_corners = next_corners.copy()
        prev_frame_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
