import cv2
import numpy as np
import time
from scipy import signal

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

k = 0.06
threshold = 0.5
window_size = 7
gaussian_window = create_gaussian_window(window_size)

Gx = np.array([[-1, 1],
               [-1, 1]])
Gy = np.array([[-1, -1],
               [1, 1]])
Gt1 = np.array([[-1, -1],
                [-1, -1]])
Gt2 = np.array([[1, 1],
                [1, 1]])

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
prev_frame_gray = prev_frame_gray/255.
#prev_corners = get_corners(prev_frame_gray)
prev_corners = findCorners(prev_frame_gray, gaussian_window, k, threshold)
prev_corners = np.array(prev_corners)
prev_corners = prev_corners.reshape(-1)
mask = np.zeros_like(prev_frame)
print(prev_corners.shape)
count = 0
while True:
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = frame_gray/255.
    Ix = (signal.convolve2d(prev_frame_gray, Gx) + signal.convolve2d(frame_gray, Gx))/2
    Iy = (signal.convolve2d(frame_gray, Gy) + signal.convolve2d(prev_frame_gray, Gy))/2
    It = signal.convolve2d(prev_frame_gray, Gt1) + signal.convolve2d(frame_gray, Gt2)
    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    A = np.zeros((2, 2))
    b = np.zeros((2, 1))
    mask = np.zeros_like(prev_frame)
    new_corners = np.zeros_like(prev_corners)
    
    for i in range(prev_corners.shape[0]//2):
        y = prev_corners[i]
        x = prev_corners[i+prev_corners.shape[0]//2]
        A[0, 0] = np.sum(Ix[y-2:y+3, x-2:x+3]**2)
        A[0, 1] = np.sum(Ix[y-2:y+3, x-2:x+3]*Iy[y-2:y+3, x-2:x+3])
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(Iy[y-2:y+3, x-2:x+3]**2)
        A_inv = np.linalg.pinv(A)
        b[0, 0] = -np.sum(Ix[y-2:y+3, x-2:x+3]*It[y-2:y+3, x-2:x+3])
        b[1, 0] = -np.sum(Iy[y-2:y+3, x-2:x+3]*It[y-2:y+3, x-2:x+3])
        c = np.matmul(A_inv, b)
        u[y, x] = c[0]
        v[y, x] = c[1]
        new_corners[i] = y + v[y, x]
        new_corners[i + prev_corners.shape[0]//2] = x + u[y, x]

    for i in range(prev_corners.shape[0]//2):
        f, g = prev_corners[i], prev_corners[i+prev_corners.shape[0]//2]
        h, j = new_corners[i], new_corners[i+prev_corners.shape[0]//2]
        mask = cv2.line(mask, (f, g), (h, j), (0, 255, 0), 3)
        frame = cv2.circle(frame, (h, j), 8, (0, 0, 255), -1)
    img = cv2.add(frame, mask)
    count += 1
    cv2.imshow('frame', img)
        
    k = cv2.waitKey(1)
    if count == 50:
        ret, prev_frame = cap.read()
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_frame_gray = prev_frame_gray/255.
        #prev_corners = get_corners(prev_frame_gray)
        prev_corners = findCorners(prev_frame_gray, gaussian_window, k, threshold)
        prev_corners = np.array(prev_corners)
        prev_corners = prev_corners.reshape(-1)
        mask = np.zeros_like(prev_frame)
        count = 0
    else:
        prev_corners = new_corners.copy()
        prev_frame_gray = frame_gray.copy()
        prev_frame = frame.copy()
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
