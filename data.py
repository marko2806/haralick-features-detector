import cv2
import numpy as np
from haralick import getHaralickFeatures
from image_processing import preprocess_image
import glob


def getHaralickFeaturesForTrainingSet(verbose=False):
    if verbose:
        print("Fetching training set")
    train_path_true = ".\\dataset\\train\\true\\"
    train_path_false = ".\\dataset\\train\\false\\"

    true_train_files = glob.glob(train_path_true + '*.png')
    false_train_files = glob.glob(train_path_false + '*.png')

    labels = np.zeros(shape=(len(true_train_files) + len(false_train_files)))
    for i in range(0, len(true_train_files)):
        labels[i] = 1

    X = []
    if verbose:
        print("Processing crowd images")
    for image in true_train_files:
        imageArray = cv2.imread(image)
        imageArray = preprocess_image(imageArray)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)
    if verbose:
        print("Processing non-crowd images")
    for image in false_train_files:
        imageArray = cv2.imread(image)
        imageArray = preprocess_image(imageArray)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)
    if verbose:
        print("Finished processing images")
    return np.array(X), labels


def getHaralickFeaturesForTestSet(verbose=False):
    if verbose:
        print("Fetching test set")
    test_path_true = ".\\dataset\\test\\true\\"
    test_path_false = ".\\dataset\\test\\false\\"

    true_test_files = glob.glob(test_path_true + '*.png')
    false_test_files = glob.glob(test_path_false + '*.png')

    labels = np.zeros(shape=(len(true_test_files) + len(false_test_files)))
    for i in range(0, len(true_test_files)):
        labels[i] = 1

    X = []
    if verbose:
        print("Processing crowd images")
    for image in true_test_files:
        imageArray = cv2.imread(image)
        imageArray = preprocess_image(imageArray)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)
    if verbose:
        print("Processing non-crowd images")
    for image in false_test_files:
        imageArray = cv2.imread(image)
        imageArray = preprocess_image(imageArray)
        haralickFeatures = getHaralickFeatures(imageArray)
        X.append(haralickFeatures)
    if verbose:
        print("Finished processing images")
    return np.array(X), labels