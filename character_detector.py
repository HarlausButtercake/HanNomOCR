import numpy as np
import subprocess

def read_label(img, str_output):
    gt = []
    for line in str_output.strip().split("\n"):
        tmp = line.strip().split(' ')

        w, h = img.shape[1], img.shape[0]
        # print(w)
        x = [(float)(w.strip()) for w in tmp if w.strip()]

        x1 = int(x[1] * w)
        width = int(x[3] * w)

        y1 = int(x[2] * h)
        height = int(x[4] * h)

        gt += [(x1, y1, width, height, 0, 0, 0)]

    return gt


class HanNomOCR:

    def __init__(self, noise=50):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'HanNomOCR'
        self.noise = noise

        np.random.seed(1)

    def detect(self, img, img_path):
        command = "python yolov5/detect.py --save-txt --conf 0.6 --iou-thres 0.15 --hide-labels --source " + img_path + " --weights yolov5/runs/train/exp/weights/best.pt"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # print(result.stdout)
        base_outputs = read_label(img, result.stdout)

        noise = np.random.randint(0, self.noise, size=(len(base_outputs), 4)) - (self.noise // 2)
        preds = []

        for i in range(len(base_outputs)):
            confidence = np.sum(np.abs(noise[i, :]))
            confidence = 1 - 1.0 * confidence / 200
            preds += [(confidence, base_outputs[i][0],
                       base_outputs[i][1],
                       base_outputs[i][2],
                       base_outputs[i][3])]
        # List of confidence, xcenter, ycenter, width, height
        # print(np.array(preds))
        return np.array(preds)
