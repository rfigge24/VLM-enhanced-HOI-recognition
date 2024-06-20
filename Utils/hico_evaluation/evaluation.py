"""
These are the functions to evaluate a model trained to do HOI classification on the HICO dataset.
These function are a distilled version of the evaluation code by github.com/ywchao/hico_benchmark. 
The original code was from 2015 and written in matlab, so these python functions were made for convenience.

"""

import numpy as np


def eval_vo(scores, labels, num_classes):
    """Evaluates the predictions of the HOI classifier on the HICO dataset by calculating the Average Precision for each class. 

    Args:
        scores: A numpy array of shape (num_classes, instances) filled with prediction scores [0-1].
        labels: A numpy array of shape (num_classes, instances) filled with the groundtruth labels [-1,0,1].
        num_classes: The number of classes to evaluate.

    Returns:
        A tuple containing: 
        (An array with the Average precision values for each class,
        A list with numpy arrays for each class containing the recall vallues of the Precision Recall plot,
        A list with numpy arrays for each class containing the corresponding precision values for the Precision Recall plot.)
    """    
    # init the metrics:
    #   average Precision:
    ap_values = np.zeros((num_classes))
    recall_list = []
    precision_list = []
    
    for i in range(num_classes):
        score = scores[i]
        label = labels[i]

        nonzero_indices = np.nonzero(label)

        score = score[nonzero_indices]
        label = label[nonzero_indices]

        rec, prec, ap = eval_pr_score_label(score, label, sum(label == 1))
        ap_values[i] = ap
        recall_list.append(rec)
        precision_list.append(prec)
    
    return ap_values, recall_list, precision_list



def eval_pr_score_label(score, label, npos):
    """This function handles the calculations of average precision for a single class.

    Args:
        score: A numpy array of shape (1,instances) containing the predicted scores for the instances
        label: A numpy array of shape (1,instances) containing the groundtruth labels for the instances
        npos: The total number of positive groundtruth labels for the class.

    Raises:
        ValueError: The labels/scores should not contain the ambiguous instances. 
        only instances labeled with -1 or 1 should be inputted to this function.

    Returns:
        A tuple containing: 
        (The Average Precision for the single class,
        A numpy array for this single class containing the recall vallue of the Precision Recall plot,
        A numpy array for this single class containing the corresponding precision value for the Precision Recall plot.)
    """    
    ulabel = np.unique(label)
    if not (len(ulabel) == 2 and ulabel[0] == -1 and ulabel[1] == 1):
        return (None,None,None) #return None if not all possible labels are there.

    # sort classification by decreasing score:
    si = score.argsort()[::-1]
    lb = label[si]

    # assign tp/fp
    nd = len(score)
    tp = np.zeros((nd,1))
    fp = np.zeros((nd,1))

    for d in range(nd):
        if lb[d] == 1:
            tp[d] = 1
        else:
            fp[d] = 1
    
    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp / (fp+tp)

    # compute average precision:
    #   The possibly zigzag pr-curve is smooth out by max(prec[rec >= t]), this makes the curve into steps.
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if not np.any(prec[rec >= t]):
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p/11
    
    return rec, prec, ap