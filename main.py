# This file will use to score your implementations.
# You should not change this file

# mAP score from: https://github.com/bes-dev/mean_average_precision

import os
import pandas as pd
import time
import sys
import cv2
import numpy as np

from character_detector import HanNomOCR
from mean_average_precision import MetricBuilder


def read_label(label_file):
    gt = []
    with open(label_file, "r") as f:
        for line in f.readlines():

            tmp = line.strip().split(' ')

            w, h = img.shape[1], img.shape[0]
            x = [float(w) for w in tmp]

            x1 = int(x[1] * w)
            width = int(x[3] * w)

            y1 = int(x[2] * h)
            height = int(x[4] * h)

            gt += [(x1-width//2, y1-height//2, x1+width//2, y1+height//2, 0, 0, 0)]

    return np.array(gt)

def get_predict(outputs):
    predicts = []
    for o in outputs:
        width = o[3]
        height = o[4]
        x1 = o[1]
        y1 = o[2]

        predicts += [(x1-width//2, y1-height//2, x1+width//2, y1+height//2, 0, o[0])]

    return np.array(predicts)

if __name__ == "__main__":

    input_folder = sys.argv[1]
    label_folder = sys.argv[2]

    start_time = time.time()
    detector = HanNomOCR(20)
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))
    iou = []

    start_time = time.time()
    total = 0
    img_score = []
    for filename in list_files:
        if not ('jpg' in filename or 'jpeg' in filename):
            continue

        total += 1
        img = cv2.imread(os.path.join(input_folder, filename))
        # img_path = os.path.join(input_folder, filename)
        targets = read_label(os.path.join(label_folder, filename[:-4] + ".txt"))
        print(img.shape)

        list_outputs = detector.detect(img, os.path.join(input_folder, filename))
        preds = get_predict(list_outputs)

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
        metric_fn.add(preds, targets)
        score = metric_fn.value(iou_thresholds=np.arange(0.5, 0.75, 0.05))['mAP']
        # for t in range(50, 76, 5):
        #    scores += [map_score(list_outputs, targets, 0.01*t)]

        img_score += [score]

    run_time = time.time() - start_time
    print("Map score: %.6f" % np.mean(img_score))
    print("Run time: ", run_time)

