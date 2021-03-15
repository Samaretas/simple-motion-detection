import cv2
import numpy as np
from motion_detection import MotionDetection

frames = list()
colored_frames = list()

# cap= cv2.VideoCapture('moving-mouse.mp4')
cap= cv2.VideoCapture('resized.avi')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    colored_frames.append(frame)
    frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), dtype='int8'))
 
cap.release()
cv2.destroyAllWindows()          

print(f"read {len(frames)} frames, of size {frames[0].shape}")
# cv2.imshow("Frame 0", colored_frames[0])
# cv2.waitKey(0)

detector = MotionDetection()
detector.diamond_search_motion_estimation(frames[0], frames[1], colored_frames[0])

