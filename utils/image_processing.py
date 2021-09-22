import numpy as np
import cv2

def extract_triangle_from_image(triangle, image):
    """select only the portion of  prediction image within this triangle and covert rgb codes to class data"""

    # crop image to fit triangle and recompute triangle coordinates
    triangle = np.array(triangle, dtype=int)
    xy_min = np.amin(triangle, axis=0).astype(int)
    xy_max = np.amax(triangle, axis=0).astype(int)
    image_crop = image[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]
    tri_crop = triangle
    tri_crop[:, 0] = tri_crop[:, 0] - xy_min[0]  # revise x-coords
    tri_crop[:, 1] = tri_crop[:, 1] - xy_min[1]  # revise y-coords
    tri_crop = np.append(tri_crop, np.array(tri_crop[0, :], ndmin=2), axis=0)  # close triangle
    mask = np.zeros((image_crop.shape[0], image_crop.shape[1]))

    # create mask over triangular region
    cv2.fillConvexPoly(mask, tri_crop, 1)
    mask = mask.astype(np.bool)
    selection = np.zeros_like(image_crop)
    selection[mask] = image_crop[mask]  # extract predictions within mask region

    return selection, tri_crop


def maskrgb_to_class( mask, class_map):
    """ decode rgb mask to classes using class map"""
    h, w, channels = mask.shape[0], mask.shape[1], mask.shape[2]
    mask_out = -1 * np.ones((h, w), dtype=int)

    for k in class_map:
        matches = np.zeros((h, w, channels), dtype=bool)

        for c in range(channels):
            matches[:, :, c] = mask[:, :, c] == k[c]

        matches_total = np.sum(matches, axis=2)
        valid_idx = matches_total == channels
        mask_out[valid_idx] = class_map[k]

    return mask_out
