import cv2
import numpy as np
import argparse
from stereo_match_sad import stereo_match

def execute_stereo_matching(img_left_path, img_right_path, window_size, max_disparity, disparity_map_path):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    disparity_map = stereo_match(img_left, img_right, window_size, max_disparity)
    cv2.imwrite(disparity_map_path, disparity_map)
    return 

parser = argparse.ArgumentParser()
parser.add_argument("--img_left_path", help="")
parser.add_argument("--img_right_path", help="")
parser.add_argument("--window_size", help="", type=int)
parser.add_argument("--max_disparity", help="", type=int)
parser.add_argument("--disparity_map_path", help="")


if __name__ == "__main__":
    args = parser.parse_args()
    execute_stereo_matching(args.img_left_path, args.img_right_path, args.window_size, args.max_disparity, args.disparity_map_path)
