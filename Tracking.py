import cv2
import numpy as np
import time
from scipy import signal
    
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
    
def add_noise(img):
    row,col = img.shape
    mean = 0
    var = 0.6
    sigma = var**0.5
    noise = np.random.normal(mean,sigma,(row,col))
    noise = noise.reshape(row,col)
    noisy_img = img + noise
    return noisy_img

def create_gaussian_window(window_size):
    ax, ay = np.meshgrid(np.linspace(-1,1,window_size), np.linspace(-1,1,window_size))
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

k = 0.06
threshold = 0.85
window_size = 7
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


#width = 400
#height =400

#writer = cv2.VideoWriter('C:/Users/USER/Downloads/Computer-Vision-with-Python/Endocap.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (width, height))

cap = cv2.VideoCapture('C:/Users/USER/Downloads/Endoscopy1.mp4')
#cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
s = prev_frame.shape
print(s)
prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_gray1 = cv2.resize(prev_frame_gray1, (400, 400))
#prev_frame_gray1 = add_noise(prev_frame_gray1)
prev_frame_gray1 = prev_frame_gray1/255.
s = prev_frame_gray1.shape
print(s)
prev_frame_gray = reduce_level(prev_frame_gray1, level)
#prev_corners = get_corners(prev_frame_gray)
prev_corners = findCorners(prev_frame_gray1, gaussian_window, k, threshold)
prev_corners = np.array(prev_corners)
prev_corners = prev_corners.reshape(-1)
prev_corners = np.int32(prev_corners/(2**level))
mask = np.zeros_like(prev_frame)
count = 0

while True:
    
    ret, frame = cap.read()
    frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray1 = cv2.resize(frame_gray1, (400, 400))
    #frame_gray1 = add_noise(frame_gray1)
    frame_gray1 = frame_gray1/255.
    frame_gray = reduce_level(frame_gray1, level)
    Ix = (signal.convolve2d(prev_frame_gray, Gx) + signal.convolve2d(frame_gray, Gx))/2
    Iy = (signal.convolve2d(frame_gray, Gy) + signal.convolve2d(prev_frame_gray, Gy))/2
    It = signal.convolve2d(prev_frame_gray, Gt1) + signal.convolve2d(frame_gray, Gt2)
    u = np.zeros((Ix.shape[0]+1000, Ix.shape[1]+1000))
    v = np.zeros((Ix.shape[0]+1000, Ix.shape[1]+1000))
    A = np.zeros((2, 2))
    b = np.zeros((2, 1))
    mask = np.zeros_like(prev_frame)                                #Comment this
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
        new_corners[i] = y - u[y, x]
        new_corners[i + prev_corners.shape[0]//2] = x - v[y, x]
    
    new_corners = np.int32(new_corners * (2**level))
    prev_corners = np.int32(prev_corners * (2**level))
    avg_x = 0
    avg_y = 0
    x_max = 0
    x_min = 1000
    y_max = 0
    y_min = 1000
    
    v = 0
    for i in range(prev_corners.shape[0]//2):
        f, g = prev_corners[i], prev_corners[i+prev_corners.shape[0]//2]
        h, j = new_corners[i], new_corners[i+prev_corners.shape[0]//2]
        #mask = cv2.line(mask, (g, f), (j, h), (0, 255, 0), 3)
        #frame = cv2.circle(frame, (j, h), 8, (0, 0, 255), -1)
        
        if 50<=j<=420 and 30<=h<=320:
            avg_x += j
            avg_y += h
            x_max = max(x_max, j)
            y_max = max(y_max, h)
            x_min = min(x_min, j)
            y_min = min(y_min, h)
            v += 1

    #avg_x = int(avg_x/(prev_corners.shape[0]//2))
    #avg_y = int(avg_y/(prev_corners.shape[0]//2))
    if v != 0:
        avg_x = int(avg_x/v)
        avg_y = int(avg_y/v)
        frame = cv2.rectangle(frame, (avg_x-x_min//2, avg_y-y_min//2), (avg_x+x_max//2, avg_y+y_max//2), (0, 0, 255), 3)
        #frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
    img = cv2.add(frame, mask)
    count += 1
    #img = cv2.rectangle(img, (50, 30), (420, 320), (0,255,0), 3)
    #writer.write(img)
    cv2.imshow('image', img)
    #cv2.imshow('part', img[100:400,100:400])
        
    k = cv2.waitKey(1)
    
    if k == 27:
        break
    
    elif k == ord('q'):
        ret, prev_frame = cap.read()
        prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #prev_frame_gray1 = add_noise(prev_frame_gray1)
        prev_frame_gray1 = cv2.resize(prev_frame_gray1, (400, 400))
        prev_frame_gray1 = prev_frame_gray1/255.
        prev_frame_gray = reduce_level(prev_frame_gray1, level)
        #prev_corners = get_corners(prev_frame_gray)
        prev_corners = findCorners(prev_frame_gray1, gaussian_window, k, threshold)
        prev_corners = np.array(prev_corners)
        prev_corners = prev_corners.reshape(-1)
        prev_corners = np.int32(prev_corners/(2**level))
        mask = np.zeros_like(prev_frame)
        count = 0

    if count == 5:
        
        ret, prev_frame = cap.read()
        prev_frame_gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #prev_frame_gray1 = add_noise(prev_frame_gray1)
        prev_frame_gray1 = cv2.resize(prev_frame_gray1, (400, 400))
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
#writer.release()
cv2.destroyAllWindows()
