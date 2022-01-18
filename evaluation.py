import numpy as np
from sklearn.metrics import confusion_matrix


def check_shape(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise Exception(
            "Masks don't have the same shape. Array 1 has a shape {}, Array 2 has a shape {}".format(arr1.shape,
                                                                                                     arr2.shape))


def get_IOU(mask1, mask2):
    check_shape(mask1, mask2)
    intersection_value = np.sum(np.logical_and(mask1, mask2))
    union_value = np.sum(np.logical_or(mask1, mask2))
    IoU = intersection_value / union_value
    print("IoU = {}".format(IoU))

    return IoU


def get_classification_metrics(true, pred):
    check_shape(true, pred)

    true = true.reshape((1, true.size))[0]
    pred = pred.reshape((1, pred.size))[0]

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    FDR = np.round(fp / (fp + tp), 2)
    MDR = np.round(fn / (fn + tn), 2)
    P = np.round(tp / (tp + fp), 2)
    R = np.round(tp / (tp + fn), 2)
    F1 = np.round( (2 * P * R)/(P + R) ,2)
    print("FDR = {}, MDR = {}".format(FDR, MDR))

    return FDR, MDR, P, R, F1