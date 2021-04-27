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


# frames, colored_frames = read_video_stream('rolling.mp4')
frames, colored_frames = read_video_stream('rolling_resized.mp4')
# frames, colored_frames = read_video_stream('Rolling-resized1over4.m4v')
print(f'read {len(frames)} frames, of size {frames[0].shape}')

detector = MotionDetection()

for i in range(3, len(frames)-1, 2):
    prev_frame_idx = i
    next_frame_idx = i+1
    print(f"frames {prev_frame_idx}-{next_frame_idx}")

    prev_frame = ((frames[prev_frame_idx]+frames[prev_frame_idx-1])/2)

    motion_map = detector.diamond_search_motion_estimation(prev_frame, frames[next_frame_idx])
    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(colored_frames[prev_frame_idx], cv2.COLOR_BGR2RGB))
    plt.title(f'Original {prev_frame_idx}')
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(colored_frames[next_frame_idx], cv2.COLOR_BGR2RGB))
    plt.title(f'Original {next_frame_idx}')
    plt.subplot(2, 2, (3,4)), plt.imshow(motion_map, 'gray')
    plt.title(f'Motion map {prev_frame_idx}-{next_frame_idx}')
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    plt.savefig('./plots/frames'+str(prev_frame_idx)+'-'+str(next_frame_idx)+'.png')
