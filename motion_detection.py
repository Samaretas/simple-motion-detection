import numpy as np
from cv2 import cv2
import time
from scipy.ndimage.filters import gaussian_filter


def timing(func):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"Function executed in {int(end-start)}s")
        return ret
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
    def diamond_search_motion_estimation(self, prev_frame, now_frame):
        """
        I assume here to get as input numpy arrays.
        """
        shape = prev_frame.shape
        motion_map = np.zeros_like(prev_frame, dtype='int16')

        # notice here the approach at borders, at the moment we neglect right and bottom leftovers
        for row in range(shape[0]-self.block_size):
            for col in range(shape[1]-self.block_size):
                # iterating over all image positions
                shift_distance = -1
                
                this_block = prev_frame[row:row + self.block_size, col:col+self.block_size]
                match_position = (row, col)

                # find the best large offset \w large diamond search
                stopping_condition = False
                while(not stopping_condition):
                    shift_distance += 1
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
                    motion_map[match_position] += 1

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
                if(best_pos != match_position):
                    shift_distance += 1
                match_position = (best_pos)

                # Compute the amount of movement
                # For now, just add 1 to all terminating points
                motion_map[match_position] += 1



        print("Motion computation finished")
        # Normalize motion map
        map_max = np.max(motion_map)
        motion_map = motion_map/map_max
        return motion_map.astype('uint8', copy=True)