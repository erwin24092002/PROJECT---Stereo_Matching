import numpy as np 
import cv2 
import tqdm

def stereo_match(img_left, img_right, window_size, max_disparity):
    # Convert images to grayscale
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Get image dimensions and create disparity map
    height, width = img_left_gray.shape
    disp_map = np.zeros((height, width), np.uint8)

    # Calculate half of window size
    half_window = int(window_size / 2)

    # Loop over image pixels
    for y in tqdm.tqdm(range(half_window, height - half_window)):
        # print('\rProcessing.. %d%% complete' %(y/(height-half_window)*100), end="", flush=True)
        for x in range(half_window, width - half_window):
            # Define search window in right image
            search_window = img_right_gray[y - half_window:y + half_window + 1,
                                            x - max_disparity - half_window:x - max_disparity + half_window + 1]

            # Define template window in left image
            template_window = img_left_gray[y - half_window:y + half_window + 1,
                                            x - half_window:x + half_window + 1]

            # Initialize best match and minimum difference
            best_match = -1
            min_diff = float('inf')

            # Search for best match in search window
            for d in range(max_disparity):
                # Define candidate window in right image
                candidate_window = img_right_gray[y - half_window:y + half_window + 1,
                                                  x - d - half_window:x - d + half_window + 1]

                # Calculate sum of absolute differences between template and candidate windows
                template_window = np.resize(template_window, candidate_window.shape)
                sad = np.sum(np.abs(template_window - candidate_window))

                # Update best match and minimum difference if necessary
                if sad < min_diff:
                    min_diff = sad
                    best_match = d

            # Set disparity map value to best match
            disp_map[y, x] = best_match

    return disp_map