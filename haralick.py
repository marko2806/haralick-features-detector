from mahotas.features import haralick


def getHaralickFeatures(subimage):
    return haralick(subimage).mean(axis=0)


def getHaralickForWindow(window, image):
    start_x, end_x, start_y, end_y = window
    return getHaralickFeatures(image[start_y:end_y, start_x:end_x])