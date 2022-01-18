# Import required packages:
import cv2
import numpy
import numpy as np



def get_movement_image(path_start, path_end, show=False):
    # Load the image and convert it to grayscale:
    image1 = cv2.imread(path_start)
    image2 = cv2.imread(path_end)

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Apply cv2.threshold() to get a binary image
    ret1, thresh1 = cv2.threshold(gray_image1, 50, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(gray_image2, 50, 255, cv2.THRESH_BINARY)
    # Find contours:
    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        image1 = cv2.arrowedLine(image1, (x1, y1), (x2, y2),
                                 (125, 0, 125), 3)
    if show is True:
        cv2.imshow("outline contour & centroid", image1)
        cv2.waitKey(0)

        # Destroy all created windows:
        cv2.destroyAllWindows()
    return image1
# get_movement_image('gettyimages-583944852-640_adpp1.png', 'gettyimages-583944852-640_adpp300.png', True)
