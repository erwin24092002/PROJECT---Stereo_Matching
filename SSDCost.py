import numpy as np
from PIL import Image


def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in range(kernel_half, h - kernel_half):
        print("\rProcessing.. %d%% complete" % (y / (h - kernel_half) * 100), end="", flush=True)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good match
                # because the squared differences of two pixels alone is not enough ot go on
                for v in range(-kernel_half, kernel_half+1):
                    for u in range(-kernel_half, kernel_half+1):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow
                        ssd_temp = int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset])
                        ssd += ssd_temp * ssd_temp

                        # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust

    # Convert to PIL and save it
    Image.fromarray(depth).save('depth.png')


if __name__ == '__main__':
    stereo_match("images/view1.png", "images/view5.png", 3, 32)  