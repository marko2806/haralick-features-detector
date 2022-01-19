import os
import random
import glob
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import mahotas
import matplotlib.pyplot as plt
from math import ceil


def getHaralickFeaturesForDataset():
    train_path_true = ".\\dataset\\train\\true\\"
    train_path_false = ".\\dataset\\train\\false\\"
    # test_path_true = ".\\dataset\\test\\true\\"
    # test_path_false = ".\\dataset\\test\\false\\"
    true_train_files = glob.glob(train_path_true + '*.png')
    false_train_files = glob.glob(train_path_false + '*.png')
    labels = np.zeros(shape=(len(true_train_files) + len(false_train_files)))
    for i in range(0, len(true_train_files)):
        labels[i] = 1

    X = []
    for image in true_train_files:
        imageArray = cv2.imread(image)
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)
    for image in false_train_files:
        imageArray = cv2.imread(image)
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)

    return np.array(X), labels


def getHaralickFeatures(subimage):
    return mahotas.features.haralick(subimage).mean(axis=0)


def performClassification(subimages, model):
    return model.predict(subimages)


def getMaskAfterClassification(image, slidingWindowAreas, model):
    mask = np.zeros(shape=image.shape)
    counter = 0
    subimages = []
    for start_x, end_x, start_y, end_y in slidingWindowAreas:
        if counter % 100 == 0:
            print(str(counter) + "/" + str(len(slidingWindowAreas)))
        counter += 1
        subimages.append(getHaralickFeatures(image[start_y:end_y, start_x:end_x]))
    predictions = performClassification(np.array(subimages), model)
    counter = 0
    for start_x, end_x, start_y, end_y in slidingWindowAreas:
        mask[start_y:end_y, start_x:end_x] = np.logical_or(mask[start_y:end_y, start_x:end_x], predictions[counter])
        counter += 1
    return mask


def getSlidingWindowAreas(image, windowWidth, windowHeight, stride=None):
    if stride is None:
        stride = windowWidth

    windowAreas = []
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            # start_x, end_x, start_y, end_y
            windowAreas.append((j, j + windowWidth,
                                i, i + windowHeight))
    return windowAreas


if __name__ == "__main__":

    print("jae")
    X, y = getHaralickFeaturesForDataset()
    classifier = SVC(kernel="linear")
    # classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X, y)

    # citanje slike
    image = cv2.imread("test_image2.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
    features = getHaralickFeatures(image)
    # shape 13,13
    slidingWindows = getSlidingWindowAreas(image, 100, 100, 25)
    mask = getMaskAfterClassification(image, slidingWindows, classifier)

    cv2.imshow("real", image)
    cv2.imshow("test", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
