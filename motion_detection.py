import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter


def timing(func):
    def wrap(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Function executed in {int(end-start)}s")
    return wrap


class MotionDetection:

    def __init__(self):
        self.block_size = 5
        self.large_search_pattern_offsets = [
            (0, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
            (0, -2),
            (1, -1)
        ]
        self.small_search_pattern_offsets = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ]
        self.max_offset_l = 2
        self.max_offset_s = 1

    def block_difference(self, ba, bb):
        return np.sum(np.abs(ba-bb))

    @timing
    def diamond_search_motion_estimation(self, prev_frame, now_frame, original=None):
        """
        I assume here to get as input numpy arrays.
        """
        shape = prev_frame.shape
        motion_map = np.zeros_like(prev_frame, dtype='int16')

        # notice here the approach at borders, it must be corrected
        for row in range(shape[0]-self.block_size):
            for col in range(shape[1]-self.block_size):
                # iterating over all image positions
                motion_distance = -1

                this_block = prev_frame[row:row +
                                        self.block_size, col:col+self.block_size]
                match_position = (row, col)

                # find the best large offset \w large diamond search
                stopping_condition = False
                while(not stopping_condition):
                    motion_distance += 1
                    min_diff = float('inf')
                    best_pos = match_position
                    for offset in self.large_search_pattern_offsets:
                        (row2, col2) = (
                            match_position[0]+offset[0], match_position[1]+offset[1])
                        row2 = min(max(row2, 0), shape[0]-self.block_size-1)
                        col2 = min(max(col2, 0), shape[1]-self.block_size-1)
                        block = now_frame[row2:row2 +
                                          self.block_size, col2:col2+self.block_size]
                        diff = self.block_difference(this_block, block)
                        if(diff < min_diff):
                            min_diff = diff
                            best_pos = (row2, col2)
                    stopping_condition = (match_position == best_pos)
                    match_position = (best_pos)
                    motion_map[match_position] += motion_distance

                # small offset search, small diamond
                min_diff = float('inf')
                for offset in self.small_search_pattern_offsets:
                    (row2, col2) = (
                        match_position[0]+offset[1], match_position[1]+offset[0])
                    row2 = min(max(row2, 0), shape[0]-self.block_size-1)
                    col2 = min(max(col2, 0), shape[1]-self.block_size-1)
                    block = now_frame[row2:row2 +
                                      self.block_size, col2:col2+self.block_size]
                    diff = self.block_difference(this_block, block)
                    if(diff < min_diff):
                        min_diff = diff
                        best_pos = (row2, col2)
                if(best_pos == match_position):
                    motion_distance += 1
                match_position = (best_pos)

                # if(not row%100 and not col%100):
                #     print(f"analyzing position  [{row},{col}] matches with [{match_position[0]},{match_position[1]}]")

                # Compute the amount of movement
                # For now, just add 1 to all terminating points
                motion_map[match_position] += motion_distance
                # motion_map[match_position] += 2**motion_distance
                # motion_map[match_position] += 3**motion_distance

        print("Motion computation finished")
        # Spread movement to all the surroundings
        map_max = np.max(motion_map)
        # bw = gaussian_filter(bw, sigma=.5, mode='constant')
        # gfilter = cv2.getGaussianKernel(self.block_size, ((self.block_size-1)/6))
        thresh = map_max/3
        def thresher(x): return (x*(255/map_max)) if x > thresh else 0
        vthresher = np.vectorize(thresher)
        bw = vthresher(motion_map)
        # bw = (bw*(255/map_max)).astype('uint8', copy=False)
        bw = bw.astype('uint8', copy=False)
        bw2 = bw
        gauss_rounds = 5
        for _ in range(gauss_rounds):
            bw2 = cv2.GaussianBlur(
                bw2, (self.block_size, self.block_size), ((self.block_size-1)/6))
        # map_max = np.max(bw2)
        # bw2 = (bw2*(255/map_max)).astype('uint8', copy=False)
        # np.savetxt('motion_map', bw, fmt='%d')
        # np.savetxt('motion_map_gauss', bw2, fmt='%d')
        # _, bw = cv2.threshold(bw, 178, 255, cv2.THRESH_BINARY)
        # with gaussian convolution
        # bw = bw.astype('int8')

        if not original is None:
            plt.subplot(2, 2, (1,2)), plt.imshow(
                cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.subplot(2, 2, 3), plt.imshow(bw, 'gray')
            plt.subplot(2, 2, 4), plt.imshow(bw2, 'gray')
            plt.title('Saliency map')
            plt.show()
