import numpy as np
from matplotlib import pyplot as plt
import cv2
from cv2 import erode, dilate, morphologyEx

def_kernel = np.ones((80, 80), np.uint8)
small_kernel = np.ones((40, 40), np.uint8)
big_kernel = np.ones((80, 80), np.uint8)

def process_mask(mask):
    print('Postprocessing: ')
    # mask = erosion(mask)
    mask = opening(mask, small_kernel)
    mask = closing(mask, small_kernel)
    mask = erosion(mask, big_kernel)
    return mask


def erosion(mask, kernel=def_kernel):
    result = erode(mask, kernel, iterations=1)
    show_result(result, 'Erosion', kernel)
    return result


def dilation(mask, kernel=def_kernel):
    result = dilate(mask, kernel, iterations=1)
    show_result(result, 'Dilation', kernel)
    return result


def opening(mask, kernel=def_kernel):
    result = morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    show_result(result, 'Opening', kernel)
    return result


def closing(mask, kernel=def_kernel):
    result = morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    show_result(result, 'Closing', kernel)
    return result


def show_result(mask, title, kernel):
    plt.imshow(mask)
    plt.title(f'{title}, kernel: {kernel.shape}')
    plt.show()
