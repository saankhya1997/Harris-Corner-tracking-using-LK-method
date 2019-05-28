import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
%matplotlib inline

def create_gaussian_window(window_size):
    ax, ay = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
    ad = np.sqrt(ax*ax+ay*ay)
    sigma, mu = 1.0, 0.0
    gaussian_window = np.exp(-( (ad-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return gaussian_window

def findCorners(img_gray, gaussian_window, k, threshold):
    dy, dx = np.gradient(img_gray)
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx*dy
    Sxx = signal.convolve2d(Ixx, gaussian_window)
    Syy = signal.convolve2d(Iyy, gaussian_window)
    Sxy = signal.convolve2d(Ixy, gaussian_window)
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    r = det - k*(trace**2)
    cv2.normalize(r, r, 0, 1, cv2.NORM_MINMAX)
    loc = np.where(r > threshold)
    return loc

k = 0.05
threshold = 0.75
window_size = 7
gaussian_window = create_gaussian_window(window_size)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    new_img = frame.copy()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    loc = findCorners(img_gray, gaussian_window, k, threshold)
    for pt in zip(*loc[::-1]):
        cv2.circle(new_img, pt, 2, (0, 0, 255), -1)
    cv2.imshow('frame', new_img)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
