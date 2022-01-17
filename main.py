import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import mahotas
from matplotlib import pyplot
from multiprocessing import Pool
import time
from model import Model
from data import *
from postprocessing import process_mask as postprocess_mask

# from video_processing import processVideo

ap = argparse.ArgumentParser(prog='main')
ap.add_argument("--verbose", default=False)
ap.add_argument("--model-path", default=None)
ap.add_argument("--save-model", default=False)
ap.add_argument("--log-results", default=False)
ap.add_argument("--log-path", default=False)
ap.add_argument("--save-result", default=False) # dodano za kasnije,

args = vars(ap.parse_args())

verbose = args["verbose"]
model_path = args["model_path"]
save_model_flag = args["save_model"]
log_results = args["log_results"]
log_path = args["log_path"]
save_result_image = args["save_result"]  # argument za spremanje maske

verbose = True
model_path = './model.joblib'
saving_path = './result_images/'


def logResults(filePath: str, iou: float, mdr: float, fdr: float):
    if verbose:
        print("Logging results to " + filePath)
    with open(filePath, "w") as file:
        file.write("IOU: " + str(iou) + "\n")
        file.write("MDR: " + str(mdr) + "\n")
        file.write("FDR: " + str(fdr) + "\n")


def preprocessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = mahotas.gaussian_filter(image, 1)
    image = cv2.resize(image, (640, 380))
    threshed = (image >= image.mean())
    labeled, n = mahotas.label(threshed)
    return labeled


def getHaralickFeatures(subimage):
    return mahotas.features.haralick(subimage).mean(axis=0)


def performClassification(subimages, model):
    return model.make_prediction(subimages)


def getHaralickForWindow(window, image):
    start_x, end_x, start_y, end_y = window
    return getHaralickFeatures(image[start_y:end_y, start_x:end_x])


def getMaskAfterClassification(image, slidingWindowAreas, model, pools):
    start_time = time.time()
    mask = np.zeros(shape=image.shape)
    image_list = [image] * len(slidingWindowAreas)
    subimages = pools.starmap(getHaralickForWindow, zip(slidingWindowAreas, image_list))
    predictions = performClassification(np.array(subimages), model)
    counter = 0
    for start_x, end_x, start_y, end_y in slidingWindowAreas:
        mask[start_y:end_y, start_x:end_x] = np.logical_or(mask[start_y:end_y, start_x:end_x], predictions[counter])
        counter += 1
    inference_time = time.time() - start_time
    if verbose:
        print("Inference time: " + str(round(inference_time, 2)) + "s")

    return mask


def getSlidingWindowAreas(image, windowWidth, windowHeight, stride=None):
    if verbose:
        print("Getting sliding window")
    if stride is None:
        stride = windowWidth

    windowAreas = []
    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            # start_x, end_x, start_y, end_y
            windowAreas.append((j, j + windowWidth,
                                i, i + windowHeight))
    return windowAreas


if __name__ == '__main__':
    # X_test, y_test = getHaralickFeaturesForTestSet(verbose)
    classifier = Model()
    if model_path is not None:
        classifier.load_model(model_path)
    else:
        X_train, y_train = getHaralickFeaturesForTrainingSet(verbose) # stavljeno unutar else-a, nema smisla da bude gore
        classifier.model = SVC(kernel="linear", degree=3)

        if verbose:
            print("Training classifier")
        classifier.train_model(X_train, y_train)

        if save_model_flag and model_path is not None:
            classifier.save_model(model_path)

    # y_predicted_test = classifier.make_prediction(X_test)
    # evaluate_classifier(y_predicted_test, y_test)

    '''
    videos = glob.glob("videos/*")
    print("Videos :" + str(videos))
    for video in videos:
        processVideo(video, classifier)
    if log_results and log_path is not None:
        logResults(log_path, 0.58, 0.21, 0.14)
    '''

    # citanje slike
    image = cv2.imread("sample_images/test_image2.png")
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # image = cv2.resize(image, (int(image.shape[1] * 0.75), int(image.shape[0] * 0.75)))
    image = preprocessImage(image)
    features = getHaralickFeatures(image)
    # shape 13,13
    slidingWindows = getSlidingWindowAreas(image, 40, 40, 20)

    with Pool(6) as p:
        mask = getMaskAfterClassification(image, slidingWindows, classifier, p)
    p.join() # dodano da skupi threadove
    pyplot.imshow(image)
    pyplot.show()
    pyplot.imshow(mask)
    pyplot.title('Sliding window result')
    pyplot.show()

    # Postprocesiranje, pyplota sve korake
    # new_mask = postprocess_mask(mask)
