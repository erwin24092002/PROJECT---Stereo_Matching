import numpy as np 
import cv2 
import tqdm

def compute_census(img_left_gray, img_right_gray, kernel):
    h, w = img_left_gray.shape  # assume that both images are same size

    kernel_half = int(kernel / 2)
    shape = (h, w, kernel*kernel-1)

    left_census = np.zeros(shape)
    right_census = np.zeros(shape)

    for y in tqdm.tqdm(range(kernel_half, h - kernel_half), desc="compute census"):
        for x in range(kernel_half, w - kernel_half):

            index = 0
            for v in range(-kernel_half, kernel_half+1):
                for u in range(-kernel_half, kernel_half+1):
                    if (not (v == 0 and u == 0)):

                        if (int(img_left_gray[y + v, x + u]) >= int(img_left_gray[y + 0, x + 0])):
                            left_census[y,x,index] = 1
                        else:
                            left_census[y, x, index] = 0

                        if (int(img_right_gray[y + v, (x + u)]) >= int(img_right_gray[y + 0, (x + 0)])):
                            right_census[y,x,index] = 1
                        else:
                            right_census[y, x, index] = 0

                        index = index + 1

    return left_census, right_census

def stereo_match_census(img_left, img_right, window_size, max_disparity):
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    left_census, right_census = compute_census(img_left_gray, img_right_gray, window_size)

    height, width = img_left_gray.shape
    disp_map = np.zeros((height, width), np.uint8)

    half_window = int(window_size / 2)

    for y in tqdm.tqdm(range(half_window, height - half_window), desc="stereo matching"):
        # print('\rProcessing.. %d%% complete' %(y/(height-half_window)*100), end="", flush=True)
        for x in range(half_window, width - half_window):
            left_window = left_census[y - half_window:y + half_window + 1,
                                            x - half_window:x + half_window + 1]

            best_match = -1
            min_diff = float('inf')

            for d in range(max_disparity):
                right_window = right_census[y - half_window:y + half_window + 1,
                                                  x + d - half_window:x + d + half_window + 1]

                hamming_distance = np.sum(np.abs(left_window-right_window))

                if hamming_distance < min_diff:
                    min_diff = hamming_distance
                    best_match = d

                if x + d + half_window + 1 >= width:
                    break

            disp_map[y, x] = best_match / max_disparity * 255

    return disp_map