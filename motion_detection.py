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

    def block_difference(self, ba, bb):
        shape = ba.shape
        diff = 0
        for row in range(shape[0]):
            for x in range(shape[0]):
                diff += abs(ba[y][x]-bb[y][x])
        return diff

    def detect_motion(self, prev_frame, now_frame, original=None):
        shape = prev_frame.shape
        np.zeros_like(prev_frame)

        for y in range(shape[1]-self.block_size):
            for x in range(shape[0]-self.block_size):
                match_psition = (y, x)

                best_offset_l = None
                while(best_offset_l != (0, 0)):
                    # large diamond search
                    min_diff = 1000000
                    # orig_block = prev_frame[y:y+self.block_size][x:x+self.block_size]
                    orig_block = [prev_frame[yt][x:x+self.block_size] for yt in range(y,y+self.block_size)]
                    # print(f'original block size :{orig_block.shape}') 
                    print(f'original block :{orig_block}') 
                    for offset in self.large_search_pattern_offsets:
                        evaluating_position = (
                            match_psition[0]+offset[0], match_psition[1]+offset[1])
                        an_block = now_frame[y:y+self.block_size][x:x+self.block_size]
                        diff = self.block_difference(orig_block, an_block)
                        if(diff < min_diff):
                            min_diff = diff
                            best_offset_l = offset
                    match_psition = (
                        match_psition[0]+best_offset_l[0], match_psition[1]+best_offset_l[1])

                # Once we found the best large offset we have to look in the smaller area
                best_offset_s = None
                min_diff = 1000000
                for offset in self.small_search_pattern_offsets:
                    evaluating_position = (
                        match_psition[0]+offset[1], match_psition[1]+offset[0])
                    diff = self.block_difference(
                        prev_frame[y: y+self.block_size][x: x +
                                                         self.block_size],
                        now_frame[evaluating_position[0]: evaluating_position[0] +
                                  self.block_size][evaluating_position[1]: evaluating_position[1]+self.block_size]
                    )
                    if(diff < min_diff):
                        min_diff = diff
                        best_offset_s = offset
                match_psition = (
                    match_psition[0]+best_offset_s[0], match_psition[1]+best_offset_s[1])

                start = cv2.circle(original, (x, y), 2,
                                   (255, 0, 0), thickness=-1)
                end = cv2.circle(
                    original, (match_psition[1], match_psition[0]), 2, (0, 0, 255), thickness=-1)
                cv2.imshow('starting point', start)
                cv2.imshow('matching point', end)
                cv2.waitKey(0)
