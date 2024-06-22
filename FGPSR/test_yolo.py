from ultralytics import YOLO
from time_cal import TimeRecorder
import torch
from time import time
# Load a pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8n-seg.pt')


# Define path to the image file
source = r"G:\BaoXiu\EX\data\vedio_process\data\dance\LRbicx4\075.png"

# Run inference on the source
# results = model(source)  # list of Results objects
results = model.predict(source, save=False, boxes=False,  imgsz=(480), conf=0.4)

TR = TimeRecorder(benchmark=False)

with torch.no_grad():
    for i in range(20):
        TR.start()
        results1 = model.predict(source, save=True, boxes=True)
        # results = model(source, imgsz=(120, 78))  # list of Results objects
        TR.end()
    # model.predict(source, show=True)

print(1)