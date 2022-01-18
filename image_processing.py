from mahotas import gaussian_filter, label
import numpy as np
from matplotlib import pyplot as plt
import cv2
from cv2 import erode, dilate, morphologyEx

def_kernel = np.ones((80, 80), np.uint8)
small_kernel = np.ones((40, 40), np.uint8)
big_kernel = np.ones((80, 80), np.uint8)


def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def process_mask(mask):
    #mask = erosion(mask)
    #mask = opening(mask, small_kernel)
    #mask = closing(mask, small_kernel)
    #mask = erosion(mask, small_kernel)
    return mask


def erosion(mask, kernel=def_kernel):
    result = erode(mask, kernel, iterations=1)
    #show_result(result, 'Erosion', kernel)
    return result


def dilation(mask, kernel=def_kernel):
    result = dilate(mask, kernel, iterations=1)
    #show_result(result, 'Dilation', kernel)
    return result


def opening(mask, kernel=def_kernel):
    result = morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #show_result(result, 'Opening', kernel)
    return result


def closing(mask, kernel=def_kernel):
    result = morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #show_result(result, 'Closing', kernel)
    return result


def show_result(mask, title, kernel):
    plt.imshow(mask)
    plt.title(f'{title}, kernel: {kernel.shape}')
    plt.show()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.GaussianBlur(image, (3,3), 0)
    image = adjust_gamma(image, 0.5)
    #image = cv2.equalizeHist(image, image)
    #image = gaussian_filter(image, 3)
    image = cv2.resize(image, (640, 350))
    #threshed = (image >= image.mean())
    #labeled, n = label(threshed)
    #return labeled
    return image