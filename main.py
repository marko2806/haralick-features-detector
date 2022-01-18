import argparse
import numpy as np
import cv2
import glob
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
import time
from model import Model
from data import getHaralickFeaturesForTrainingSet, getHaralickFeaturesForTestSet
from video_processing import processVideo
from image_processing import preprocess_image
from haralick import *
from logger import Logger
from cvat_annotation_parser import getMaskForFrame
from image_processing import process_mask as postprocess_mask
from evaluation import get_IOU, get_classification_metrics
from movement_calcualtion import get_movement_image


ap = argparse.ArgumentParser()
ap.add_argument("--verbose",     default=True)
ap.add_argument("--model-path",  default=None)
ap.add_argument("--save-model",  default=False)
ap.add_argument("--load-model",  default=False)
ap.add_argument("--log-results", default=False)
ap.add_argument("--log-path",    default=None)
ap.add_argument("--window-size", default=50)
ap.add_argument("--stride", default=25)

args = vars(ap.parse_args())

verbose = args["verbose"]
model_path = args["model_path"]
save_model_flag = args["save_model"]
load_model_flag = args["load_model"]
log_results = args["log_results"]
log_path = args["log_path"]
window_size = int(args["window_size"])
stride = int(args["stride"])


def getMaskAfterClassification(image, slidingWindowAreas, model, pools):
    if verbose:
        print("Performing segmentation")
    start_time = time.time()
    mask = np.zeros(shape=image.shape, dtype=np.uint8)
    image_list = [image] * len(slidingWindowAreas)
    subimages = pools.starmap(getHaralickForWindow, zip(slidingWindowAreas, image_list))
    predictions = model.make_prediction(np.array(subimages))
    counter = 0
    for start_x, end_x, start_y, end_y in slidingWindowAreas:
        mask[start_y:end_y, start_x:end_x] = np.logical_or(mask[start_y:end_y, start_x:end_x], predictions[counter])
        counter += 1
    inference_time = time.time() - start_time
    if verbose:
        print("Inference time: " + str(round(inference_time, 2)) + "s")

    return mask


def getSlidingWindowAreas(imageWidth, imageHeight, windowSize, stride=None):
    if verbose:
        print("Getting sliding window")
    if stride is None:
        stride = windowSize

    windowAreas = []
    for i in range(0, imageHeight, stride):
        for j in range(0, imageWidth, stride):
            windowAreas.append((j, j + windowSize,
                                i, i + windowSize))
    return windowAreas


ious = []
mdrs = []
fdrs = []
previousMask = []

def processFrame(frame, filePath, frameNumber):
    originalFrame = frame.copy()
    frame = preprocess_image(frame)
    slidingWindows = getSlidingWindowAreas(frame.shape[1], frame.shape[0], window_size, stride)
    with Pool(12) as p:
        mask1 = getMaskForFrame("labels" + filePath[6:-4] + ".xml", int(frameNumber))
        mask = getMaskAfterClassification(frame, slidingWindows, classifier, p)
    mask = postprocess_mask(mask)
    border = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    cv2.drawContours(originalFrame, contours, -1, (36, 255, 12), thickness=1)
    if len(previousMask) > 0:
        get_movement_image(previousMask.pop(), mask, originalFrame)
    previousMask.append(mask)
    cv2.waitKey()
    cv2.imwrite("./detections/" + filePath[6:-4] + "_" + str(int(frameNumber)) + ".png", originalFrame)
    iou = get_IOU(mask, mask1)
    ious.append(iou)


# izvodi samo glavna dretva
if __name__ == '__main__':
    classifier = Model()
    logger = Logger(log_path)
    if load_model_flag and model_path is not None:
        classifier.load_model(model_path, verbose)
    else:
        X_train, y_train = getHaralickFeaturesForTrainingSet(verbose)

        classifier.model = SVC(kernel="linear", C=1)
        #classifier.model = KNeighborsClassifier(n_neighbors=2)
        if verbose:
            print("Training classifier")
        classifier.train_model(X_train, y_train)
        if save_model_flag and model_path is not None:
            classifier.save_model(model_path, verbose)

    X_test, y_test = getHaralickFeaturesForTestSet(verbose)
    FDR, MDR, P, R, F1 = get_classification_metrics(y_test, classifier.make_prediction(X_test))
    if verbose:
        print("MDR: " + str(MDR))
        print("FDR: " + str(FDR))
        print("P: " + str(P))
        print("R: " + str(R))
        print("F1: " + str(F1))

    videos = glob.glob("videos/*")
    for video in videos:
        processVideo(video, processFrame, 100, verbose)
        print(np.mean(np.array(ious)))

    iou = np.mean(np.array(ious))

    if log_results and log_path is not None:
        logger.logSegmentationResults(iou, verbose)
        logger.logClassificationResults(P, R, F1, MDR, FDR, verbose)