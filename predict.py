# This module checks photo

from ultralytics import YOLO
import numpy as np
import timeit

path = 'imgtest.jpg'


def screenshot_detection(path):
    model = YOLO('runs/classify/train7//weights/best.pt') # Use model with best results.
    results = model(path)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    sd = names_dict[np.argmax(probs)]
    return sd



