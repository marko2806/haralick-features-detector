import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def get_precision_recall_f1score(true, pred):
     check_shape(true, pred)
     true = true.reshape((1, true.size))[0]
     pred = pred.reshape((1, pred.size))[0]

     precision = np.round(precision_score(true, pred),2)
     recall = np.round(recall_score(true, pred),2)
     f1score = np.round(f1_score(true, pred),2)
     print("Precision = {}, Recall = {}, F1Score = {}".format(precision, recall, f1score))
    
     return precision, recall, f1score