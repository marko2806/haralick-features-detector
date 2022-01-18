# Import required packages:
import cv2
import numpy
import numpy as np


# Load the image and convert it to grayscale:
def get_movement_image(path_start, path_end):
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
    # Draw contours:
    # cv2.drawContours(image, contours, 0, (0, 255, 0), 2)

    M1_ = [cv2.moments(c) for c in contours1]
    M2_ = [cv2.moments(c) for c in contours2]
    print("size_h:", len(hierarchy2))
    if len(M1_) != len(M2_):
        print("Wrong object count")
        return
    # Calculate image moments of the detected contour
    xy1 = [[round(m['m10'] / m['m00']), round(m['m01'] / m['m00'])] for m in M1_]
    xy2 = [[round(m['m10'] / m['m00']), round(m['m01'] / m['m00'])] for m in M2_]

    for i_countour, xy_pair1 in enumerate(xy1):
        x1 = xy_pair1[0]
        y1 = xy_pair1[1]
        min_distance = 1000000
        i_pair_min = 0
        print("xy1before:", xy1)
        print("xy2before:", xy2)
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
        print("xy2removed:", xy2)
        image1 = cv2.arrowedLine(image1, (x1, y1), (x2, y2),
                                 (125, 0, 125), 3)
        cv2.circle(image1, (
            round(M1_[i_countour]['m10'] / M1_[i_countour]['m00']),
            round(M1_[i_countour]['m01'] / M1_[i_countour]['m00'])), 5,
                   (0, 255,), -1)
    # cv2.imshow("outline contour & centroid", image1)
    # cv2.waitKey(0)
    #
    # # Destroy all created windows:
    # cv2.destroyAllWindows()

# M1 = cv2.moments(contours1[0])
# M2 = cv2.moments(contours2[0])
# # Print center (debugging):
# center_x1 = round(M1['m10'] / M1['m00'])
# center_y1 = round(M1['m01'] / M1['m00'])
# print("center X1 : '{}'".format(round(M1['m10'] / M1['m00'])))
# print("center Y1 : '{}'".format(round(M1['m01'] / M1['m00'])))
#
# center_x2 = round(M2['m10'] / M2['m00'])
# center_y2 = round(M2['m01'] / M2['m00'])
# print("center X1 : '{}'".format(round(M2['m10'] / M2['m00'])))
# print("center Y1 : '{}'".format(round(M2['m01'] / M2['m00'])))
#
# # Draw a circle based centered at centroid coordinates
# image1 = cv2.arrowedLine(image1, (center_x1, center_y1), (center_x2, center_y2),
#                          (0, 255,), 3)
# cv2.circle(image1, (round(M1['m10'] / M1['m00']), round(M1['m01'] / M1['m00'])), 5, (0, 255, 0), -1)

# Show image:
# cv2.imshow("outline contour & centroid", image1)
#
# # Wait until a key is pressed:
# cv2.waitKey(0)
#
# # Destroy all created windows:
# cv2.destroyAllWindows()
#get_movement_image('gettyimages-583944852-640_adpp30.png' , 'gettyimages-583944852-640_adpp300.png')
