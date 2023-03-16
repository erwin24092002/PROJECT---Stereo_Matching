import numpy as np 
import cv2 
import tqdm

def stereo_match(img_left, img_right, window_size, max_disparity):
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    height, width = img_left_gray.shape
    disp_map = np.zeros((height, width), np.uint8)

    half_window = int(window_size / 2)

    for y in tqdm.tqdm(range(half_window, height - half_window)):
        # print('\rProcessing.. %d%% complete' %(y/(height-half_window)*100), end="", flush=True)
        for x in range(half_window, width - half_window):
            left_window = img_left_gray[y - half_window:y + half_window + 1,
                                            x - half_window:x + half_window + 1]

            best_match = -1
            min_diff = float('inf')

            for d in range(max_disparity):
                right_window = img_right_gray[y - half_window:y + half_window + 1,
                                                  x + d - half_window:x + d + half_window + 1]

                sad = np.sum(np.abs(left_window - right_window))

                if sad < min_diff:
                    min_diff = sad
                    best_match = d

                if x + d + half_window + 1 >= width:
                    break

            disp_map[y, x] = best_match / max_disparity * 255

    return disp_map