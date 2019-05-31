import cv2
import numpy as np
import time
from scipy import signal

def expand(img):
    w, h = img.shape
    nw = int(w)*2
    nh = int(h)*2
    new_img = np.zeros((nw, nh))
    new_img[::2, ::2] = img
    G = create_gaussian_window(5)
    for i in range(2, new_img.shape[0]-2, 2):
        for j in range(2, new_img.shape[1]-2, 2):
            m = new_img[i-2:i+3, j-2:j+3]
            new_img[i, j] = np.sum(m*G)
    return new_img

def expand_level(img, level):
    if level == 0:
        return img
    else:
        i = 0
        new_img = img.copy()
        while i < level:
            new_img = expand(new_img)
            i += 1
        return new_img
    
def reduce(img):
    w, h = img.shape
    nw = int(w//2)
    nh = int(h//2)
    new_img = np.zeros((nw, nh))
    G = create_gaussian_window(5)
    for i in range(2, img.shape[0]-2, 2):
        for j in range(2, img.shape[1]-2, 2):
            m = img[i-2:i+3, j-2:j+3]
            new_img[i//2, j//2] = np.sum(m*G)
    return new_img

def reduce_level(img, level):
    if level == 0:
        return img
    else:
        i = 0
        new_img = img.copy()
        while i < level:
            new_img = reduce(new_img)
            i += 1
        return new_img

def create_gaussian_window(window_size):
    ax, ay = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
    ad = np.sqrt(ax*ax+ay*ay)
    sigma, mu = 1.5, 0.0
    gaussian_window = 1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-((ad-mu)**2/(2.0*sigma**2))))
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

k = 0.055
threshold = 0.7
window_size = 11
gaussian_window = create_gaussian_window(window_size)
level = 0

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
Gt1 = np.array([[-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1]])
Gt2 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

#cap = cv2.VideoCapture('C:/Users/USER/Downloads/videoplayback (2).mp4')
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_gray1 = prev_frame_gray1/255.
prev_frame_gray = reduce_level(prev_frame_gray1, level)
#prev_corners = get_corners(prev_frame_gray)
prev_corners = findCorners(prev_frame_gray1, gaussian_window, k, threshold)
prev_corners = np.array(prev_corners)
prev_corners = prev_corners.reshape(-1)
prev_corners = np.int32(prev_corners/(2**level))
print(prev_corners.shape)
mask = np.zeros_like(prev_frame)
count = 0
while True:
    
    ret, frame = cap.read()
    frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray1 = frame_gray1/255.
    frame_gray = reduce_level(frame_gray1, level)
    Ix = (signal.convolve2d(prev_frame_gray, Gx) + signal.convolve2d(frame_gray, Gx))/2
    Iy = (signal.convolve2d(frame_gray, Gy) + signal.convolve2d(prev_frame_gray, Gy))/2
    It = signal.convolve2d(prev_frame_gray, Gt1) + signal.convolve2d(frame_gray, Gt2)
    u = np.zeros((Ix.shape[0]+100, Ix.shape[1]+100))
    v = np.zeros((Ix.shape[0]+100, Ix.shape[1]+100))
    A = np.zeros((2, 2))
    b = np.zeros((2, 1))
    #mask = np.zeros_like(prev_frame)
    new_corners = np.zeros_like(prev_corners)
    
    for i in range(prev_corners.shape[0]//2):
        y = prev_corners[i]
        x = prev_corners[i+prev_corners.shape[0]//2]
        A[0, 0] = np.sum(Ix[y-3:y+4, x-3:x+4]**2)
        A[0, 1] = np.sum(Ix[y-3:y+4, x-3:x+4]*Iy[y-3:y+4, x-3:x+4])
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(Iy[y-3:y+4, x-3:x+4]**2)
        A_inv = np.linalg.pinv(A)
        b[0, 0] = -np.sum(Ix[y-3:y+4, x-3:x+4]*It[y-3:y+4, x-3:x+4])
        b[1, 0] = -np.sum(Iy[y-3:y+4, x-3:x+4]*It[y-3:y+4, x-3:x+4])
        c = np.matmul(A_inv, b)
        u[y, x] = c[0]
        v[y, x] = c[1]
        new_corners[i] = y + u[y, x]
        new_corners[i + prev_corners.shape[0]//2] = x + v[y, x]
    
    new_corners = np.int32(new_corners * (2**level))
    prev_corners = np.int32(prev_corners * (2**level))
    
    for i in range(prev_corners.shape[0]//2):
        f, g = prev_corners[i], prev_corners[i+prev_corners.shape[0]//2]
        h, j = new_corners[i], new_corners[i+prev_corners.shape[0]//2]
        mask = cv2.line(mask, (g, f), (j, h), (0, 255, 0), 3)
        frame = cv2.circle(frame, (j, h), 8, (0, 0, 255), -1)
    img = cv2.add(frame, mask)
    count += 1
    cv2.imshow('frame', img)
        
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('q'):
        ret, prev_frame = cap.read()
        prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_frame_gray1 = prev_frame_gray1/255.
        prev_frame_gray = reduce_level(prev_frame_gray1, level)
        #prev_corners = get_corners(prev_frame_gray)
        prev_corners = findCorners(prev_frame_gray1, gaussian_window, k, threshold)
        prev_corners = np.array(prev_corners)
        prev_corners = prev_corners.reshape(-1)
        prev_corners = np.int32(prev_corners/(2**level))
        mask = np.zeros_like(prev_frame)
        count = 0

    if count == 50:
        ret, prev_frame = cap.read()
        prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_frame_gray1 = prev_frame_gray1/255.
        prev_frame_gray = reduce_level(prev_frame_gray1, level)
        #prev_corners = get_corners(prev_frame_gray)
        prev_corners = findCorners(prev_frame_gray1, gaussian_window, k, threshold)
        prev_corners = np.array(prev_corners)
        prev_corners = prev_corners.reshape(-1)
        prev_corners = np.int32(prev_corners/(2**level))
        mask = np.zeros_like(prev_frame)
        count = 0
    else:
        new_corners = np.int32(new_corners/(2**level))
        prev_corners = new_corners.copy()
        prev_frame_gray = frame_gray.copy()
        prev_frame = frame.copy()
        
cap.release()
cv2.destroyAllWindows()
