!pip install ultralytics -U

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Dataset을 download 및 unzip
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/Colab Notebooks/컴퓨터비전/CV_Train.zip" /content/
!unzip /content/CV_Train.zip

model = YOLO('yolo11s.pt') # YOLO 11n,11s,11m,11l,11x 순으로 모델 용량이 크다

# 훈련 이미지 디렉토리
img_dir = "CV_Train/Images"

# 저장할 디렉토리 생성 (사람 detection한 이미지 저장)
save_dir = "output_with_boxes"
os.makedirs(save_dir, exist_ok=True)

# CV_Train/Images/000000.png ~ CV_Train/Images/0000299.png 처리
for idx in range(300):
    filename = f"{idx:06d}.png"
    img_path = os.path.join(img_dir, filename)
    print(img_path)
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        continue
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 객체 탐지
    results = model(img_path, verbose=False)

    # 사람 클래스만 Detection (YOLO class ID 0 = person)
    for result in results:
        person_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id == 0:
                person_boxes.append(box)

    # Bounding box 시각화
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "person", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # 결과 저장
    out_path = os.path.join(save_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Label 구조 확인
json_path = "CV_Train/Labels/000000.json"
with open(json_path, 'r') as f:
    data = json.load(f)

import pprint
pprint.pprint(data)


print("Top-level keys:", data.keys())

input_dir = "CV_Train/Labels"
output_dir = "CV_Train/YOLO_Labels" # JSON 파일을 txt 파일로 변환 시켜야함
os.makedirs(output_dir, exist_ok=True)

for json_file in os.listdir(input_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(input_dir, json_file)
    with open(json_path, 'r') as f:
        data = json.load(f)

        img_w = data['imageWidth']
        img_h = data['imageHeight']
        shapes = data['shapes']

        label_lines = []

        for shape in shapes:
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]

            xc = ((x1+x2) / 2) / img_w
            yc = ((y1+y2) / 2) / img_h
            w = abs(x2-x1) / img_w
            h = abs(y2-y1) / img_h

            label_line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            label_lines.append(label_line)

        txt_filename = json_file.replace(".json", ".txt")
        output_path = os.path.join(output_dir, txt_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(label_lines))

import shutil

shutil.copytree("CV_Train/Images", "yolo_data/images", dirs_exist_ok=True)
shutil.copytree("CV_Train/YOLO_Labels", "yolo_data/labels", dirs_exist_ok=True)

# data.yaml 자동 생성 (클래스: person)
with open("data.yaml", "w") as f:
    f.write("""train: yolo_data/images
val: yolo_data/images
nc: 1
names: ['person']
""")

# Fine tuning
model.train(data="data.yaml", epochs=10, imgsz=640)

# Best model load
model_tuning = YOLO("runs/detect/train/weights/best.pt")

# Fine tuning된 model로 다시 detection 수행
# 훈련 이미지 디렉토리
img_dir = "CV_Train/Images"

# 저장할 디렉토리 생성 (사람 detection한 이미지 저장)
save_dir = "output_with_boxes_tuning"
os.makedirs(save_dir, exist_ok=True)

# CV_Train/Images/000000.png ~ CV_Train/Images/0000299.png 처리
for idx in range(300):
    filename = f"{idx:06d}.png"
    img_path = os.path.join(img_dir, filename)
    print(img_path)
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        continue
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 객체 탐지
    results = model_tuning(img_path, verbose=False)

    # 사람 클래스만 Detection (YOLO class ID 0 = person)
    for result in results:
        person_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id == 0:
                person_boxes.append(box)

    # Bounding box 시각화
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "person", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # 결과 저장
    out_path = os.path.join(save_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
