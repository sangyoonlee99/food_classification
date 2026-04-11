## yolo 욜로 ##

# core/detector.py
from ultralytics import YOLO

class FoodDetector:
    def __init__(self, weight_path, conf=0.4):
        self.model = YOLO(weight_path)
        self.conf = conf

    def detect(self, img):
        result = self.model(img, conf=self.conf)[0]
        return result.boxes
