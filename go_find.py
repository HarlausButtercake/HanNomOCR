import subprocess

image_path = "examples1/images/nlvnpf-0174-03-013.jpg"
command = "python yolov5/detect.py --save-txt --conf 0.6 --iou-thres 0.15 --hide-labels --source " + image_path + " --weights yolov5/runs/train/exp/weights/best.pt"

result = subprocess.run(command, shell=True, capture_output=True, text=True)
print(result.stdout)
