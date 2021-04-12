import os
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from motion_detection import MotionDetection


def read_video_stream(name):
    frames = list()
    colored_frames = list()
    cap = cv2.VideoCapture(name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        colored_frames.append(frame)
        frames.append(np.array(cv2.cvtColor(
            frame, cv2.COLOR_RGB2GRAY), dtype='uint8'))
    cap.release()
    return frames, colored_frames


frames, colored_frames = read_video_stream('rolling_resized.mp4')
# frames, colored_frames = read_video_stream('Rolling-resized1over4.m4v')
print(f'read {len(frames)} frames, of size {frames[0].shape}')

detector = MotionDetection()

for i in range(len(frames)-1):
    motion_map = detector.diamond_search_motion_estimation(frames[i], frames[i+1])

    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(colored_frames[i], cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(colored_frames[i+1], cv2.COLOR_BGR2RGB))
    plt.title('Original')

    plt.subplot(2, 2, (3,4)), plt.imshow(motion_map, 'gray')
    plt.title('Motion map')
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    plt.savefig('./plots/frames'+str(i)+'-'+str(i+1)+'.png')
