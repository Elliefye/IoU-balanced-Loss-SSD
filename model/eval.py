import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch

labels = ['Background', 'Knife', 'Horse', 'Human body']


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_model(all_detections, all_annotations, iou_threshold=0.5):
    average_precisions = {}
    p_r = {}

    for label in range(4):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            detections = []
            annotations = []

            for _, xmin, ymin, xmax, ymax, conf, class_id in all_detections[i]:
                if int(class_id) == label:
                    detections.append([xmin, ymin, xmax, ymax])
                    scores = np.append(scores, float(conf))

            if int(all_annotations[i][5]) == label:
                annotations = [[all_annotations[i][1], all_annotations[i][2], all_annotations[i][3], all_annotations[i][4]]]

            detections = np.array(detections, dtype='int')
            annotations = np.array(annotations, dtype='int')

            num_annotations += len(annotations)
            detected_annotations = []

            for d in detections:
                if len(annotations) == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d,  axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            p_r[label] = [], []
            continue

        # sort by score
        # indices = np.argsort(-scores)
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        p_r[label] = precision, recall

    map = 0.0
    for i in range(1, 4):
        map += average_precisions[i][0]
    map /= 3

    print('\nmAP: ', map)
    for label in range(1, 4):
        label_name = labels[label]
        if not label_name.isnumeric():
            print(label_name)
            print("AP: " + str(average_precisions[label][0]))
            precision, recall = p_r[label]
            print("Precision: ", precision[-1] if len(precision) > 0 else 0)
            print("Recall: ", recall[-1] if len(recall) > 0 else 0)

    return average_precisions
