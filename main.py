from cv2 import cv2
import numpy as np
from motion_detection import MotionDetection

def read_video_stream(name):
    frames = list()
    colored_frames = list()
    cap= cv2.VideoCapture(name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        colored_frames.append(frame)
        frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), dtype='uint8'))
    cap.release()
    return frames, colored_frames


frames, colored_frames = read_video_stream('rolling_resized.mp4')
print(f'read {len(frames)} frames, of size {frames[0].shape}')

detector = MotionDetection()
detector.diamond_search_motion_estimation(frames[30], frames[31], colored_frames[30])

