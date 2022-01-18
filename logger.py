class Logger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        #with open(self.filepath, "w") as file:
        #    file.write("Statistika: ")

    def logSegmentationResults(self, iou: float, verbose=False):
        if verbose:
            print("Logging results to " + self.filepath)
        with open(self.filepath, "w+") as file:
            file.write("Segmentation results: \n")
            file.write("IOU: " + str(iou) + "\n")

    def logClassificationResults(self, precision:float, recall:float, f1_score:float, mdr: float, fdr: float, verbose=False):
        # TODO promijeni writing mode u append
        if verbose:
            print("Logging results to " + self.filepath)
        with open(self.filepath, "w+") as file:
            file.write("Classification results: \n")
            file.write("P : " + str(precision) + "\n")
            file.write("R : " + str(recall) + "\n")
            file.write("F1: " + str(f1_score) + "\n")
            file.write("MDR: " + str(mdr) + "\n")
            file.write("FDR: " + str(fdr) + "\n")
