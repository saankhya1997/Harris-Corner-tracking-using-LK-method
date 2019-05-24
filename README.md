# Harris-Corner-tracking-using-LK-method

Harris Corners Detection is a method used to detect corners in an image. An error function is set up which is the correlation of the window of a fixed size and the squared difference of the change in intensity for a small shift. Corners can tell us a lot about features of an image. Hence, it is necessary to detect corners in an image or video.

The process requires calculating gradients in both x and y direction. The equation simplifies to a matrix which has two eigen values lambda1 and lambda2 in the two directions. If there is change in gradient in no directions, it is a plain region. If there is change in gradient by a huge amount in one direction but not much in the other direction, an edge is detected. Also, if gradients change by huge amounts in both the direction, a corner is detected. 

Harris Corners are detected and tracked using Lucas Kanade method for object tracking (optical flow). In the code, I have used a web camera. If you want to play your own video, you can replace the '0' in Video Capture method of cv2 with the path of the video.

You can stop the video by the 'Escape' key and clear the tracking in the video by pressing 'c'. 

Also, Harris Corners are detected after every 1000 frames(You can change the count as you want). If you want to detect them at any instant and then track them, press 'q'. 
