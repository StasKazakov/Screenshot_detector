# This module trains the neural net.

from ultralytics import YOLO

model: YOLO = YOLO('yolov8n-cls.pt')

results = model.train(data='DataSet/', epochs=10, imgsz=[360, 640])