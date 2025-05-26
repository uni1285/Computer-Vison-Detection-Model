### Install the packgae using below command
# !pip install ultralytics -U

from ultralytics import YOLO
import cv2
import os

class YOLODetector:
    def __init__(self, model_path="yolo11s.pt"):
        self.model = YOLO(model_path)

    def detect_person(self, img_path):
        results = self.model(img_path, verbose=False)
        person_boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # class 0: person
                    person_boxes.append(box)
        return person_boxes

    def draw_boxes(self, img_path, boxes, save_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
