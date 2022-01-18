# Import required packages:
import cv2
import numpy
import numpy as np


def get_movement_image(mask_start, mask_end, frame, show=False):
    # Find contours:
    contours1, hierarchy1 = cv2.findContours(mask_start, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(mask_end, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Calculate image moments of the detected contour
    M1_ = [cv2.moments(c) for c in contours1]
    M2_ = [cv2.moments(c) for c in contours2]
    xy1 = [[round(m['m10'] / m['m00']), round(m['m01'] / m['m00'])] for m in M1_]
    xy2 = [[round(m['m10'] / m['m00']), round(m['m01'] / m['m00'])] for m in M2_]
    if len(M1_) != len(M2_):
        areas1 = [cv2.contourArea(c) for c in contours1]
        area1 = np.sum([cv2.contourArea(c) for c in contours1])

        areas2 = [cv2.contourArea(c) for c in contours2]
        area2 = np.sum([cv2.contourArea(c) for c in contours2])

        xy1_ = [[xy[0] * (areas1[i] / area1), xy[1] * (areas1[i] / area1)] for i, xy in enumerate(xy1)]
        xy1 = [np.sum(xy1_, axis=0).astype(int)]

        xy2_ = [[xy[0] * (areas2[i] / area2), xy[1] * (areas2[i] / area2)] for i, xy in enumerate(xy2)]
        xy2 = [np.sum(xy2_, axis=0).astype(int)]

    for i_countour, xy_pair1 in enumerate(xy1):
        x1 = xy_pair1[0]
        y1 = xy_pair1[1]
        min_distance = 1000000
        i_pair_min = 0
        for i, xy_pair2 in enumerate(xy2):
            x2 = xy_pair2[0]
            y2 = xy_pair2[1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance < min_distance:
                min_distance = distance
                i_pair_min = i
        x2 = xy2[i_pair_min][0]
        y2 = xy2[i_pair_min][1]
        xy2.remove(xy2[i_pair_min])
        if min_distance < mask_end.shape[0] * 0.25:
            frame = cv2.arrowedLine(frame, (x1, y1), (x2 + (x2 - x1), y2 + (y2 - y1)), (125, 0, 125), 3)
    if show is True:
        cv2.imshow("outline contour & centroid", frame)
        cv2.waitKey(0)

        # Destroy all created windows:
        cv2.destroyAllWindows()
    return frame