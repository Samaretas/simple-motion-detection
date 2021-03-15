import numpy as np
import cv2


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


    def diamond_search_motion_estimation(self, prev_frame, now_frame, original=None):
        """
        I assume here to get as input numpy arrays.
        """
        shape = prev_frame.shape
        motion_map = np.zeros_like(prev_frame)

        # notice here the approach at borders, it must be corrected
        for row in range(shape[0]-self.block_size):
            for col in range(shape[1]-self.block_size):
                # iterating over all image positions

                this_block = prev_frame[row:row+self.block_size, col:col+self.block_size]
                match_position = (row, col)

                # find the best large offset \w large diamond search
                best_offset_l = None
                while(best_offset_l != (0, 0)):
                    min_diff = float('inf')
                    for offset in self.large_search_pattern_offsets:
                        (row2, col2) = (match_position[0]+offset[0], match_position[1]+offset[1])
                        row2 = min(max(row2,0), shape[0]-self.block_size-1)
                        col2 = min(max(col2,0), shape[1]-self.block_size-1)
                        block = now_frame[row2:row2+self.block_size, col2:col2+self.block_size]
                        diff = self.block_difference(this_block, block)
                        if(diff < min_diff):
                            min_diff = diff
                            best_offset_l = offset
                    match_position = (
                        match_position[0]+best_offset_l[0], match_position[1]+best_offset_l[1])

                # small offset search, small diamond
                best_offset_s = None
                min_diff = float('inf')
                for offset in self.small_search_pattern_offsets:
                    (row2,col2) = (match_position[0]+offset[1], match_position[1]+offset[0])
                    row2 = min(max(row2,0), shape[0]-self.block_size-1)
                    col2 = min(max(col2,0), shape[1]-self.block_size-1)
                    block = now_frame[row2:row2+self.block_size, col2:col2+self.block_size]
                    diff = self.block_difference(this_block, block)
                    if(diff < min_diff):
                        min_diff = diff
                        best_offset_s = offset
                match_position = (match_position[0]+best_offset_s[0], match_position[1]+best_offset_s[1])

                if(not row%71 and not col%71):
                    print(f"analyzing position  [{row},{col}] matches with [{match_position[0]},{match_position[1]}]")
                # start = cv2.circle(original, (row, col), 2,
                #                    (255, 0, 0), thickness=-1)
                # end = cv2.circle(
                #     original, (match_position[0], match_position[1]), 2, (0, 0, 255), thickness=-1)
                # cv2.imshow('starting point', start)
                # cv2.imshow('matching point', end)
                # cv2.waitKey(0)
