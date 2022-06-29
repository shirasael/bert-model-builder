import numpy as np


def b_tp_label(preds, labels, true_label):
    '''Returns True Positives (TP): count of correct predictions of actual true_label'''
    return sum([preds == labels and preds == true_label for preds, labels in zip(preds, labels)])


def b_fp_label(preds, labels, true_label):
    '''Returns False Positives (FP): count of wrong predictions of actual not true_label'''
    return sum([preds != labels and preds == true_label for preds, labels in zip(preds, labels)])


def b_tn_label(preds, labels, false_label):
    '''Returns True Negatives (TN): count of correct predictions of actual false_label'''
    return sum([preds == labels and preds != false_label for preds, labels in zip(preds, labels)])


def b_fn_label(preds, labels, false_label):
    '''Returns False Negatives (FN): count of wrong predictions of actual false_label'''
    return sum([preds != labels and preds != false_label for preds, labels in zip(preds, labels)])


def b_tp(preds, labels):
    '''Returns True Positives (TP): count of correct predictions of actual true_label'''
    return sum([b_tp_label(preds, labels, label) for label in labels])


def b_fp(preds, labels):
    '''Returns False Positives (FP): count of wrong predictions of actual not true_label'''
    return sum([b_fp_label(preds, labels, label) for label in labels])


def b_tn(preds, labels):
    '''Returns True Negatives (TN): count of correct predictions of actual false_label'''
    return sum([b_tn_label(preds, labels, label) for label in labels])


def b_fn(preds, labels):
    '''Returns False Negatives (FN): count of wrong predictions of actual false_label'''
    return sum([b_fn_label(preds, labels, label) for label in labels])


def b_metrics(preds, labels):
    '''
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    '''
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / (tp + tn + fp + fn)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity
