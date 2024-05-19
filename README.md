IMPLEMENTING YOLO V5 TO RECOGNIZING HAN-NOM CHARACTER

Places relevant images and their respective labels into examples1

Prepare the python environment by 
	pip install -r requirements.txt
		(not the one in the 'yolov5' folder)

Execute the following command in the terminal:
python main.py examples1/images examples1/labels

***The file yolov5/detect.py has been modified to run on the WINDOWS OPERATING SYSTEM, as none of us uses Linux or MacOS,
THUS THE CODE WILL NOT RUN PROPERLY ON UNIX-BASED SYSTEM (i guess...)
	For more detail please check the accompanying report.

The entire project was redesigned to resemble the provided code 'sample_score' as much as possible. 
If the above command throws errors or doesn't show anything after running, try:
	//
python yolov5/detect.py --save-txt --conf 0.6 --iou-thres 0.15 --hide-labels --source examples1/images/nlvnpf-0137-01-045.jpg --weights yolov5/runs/train/exp/weights/best.pt
	//
to directly use the model. The directory to an image proceeding "--source" can be changed. Running this command, of course, will not print out the mAP score and the total run time.

Thank you for your time!
	
