from detect.yolo_detector import YOLODetector
from detect.label_converter import convert_json_to_yolo
from detect.dataset_preparer import prepare_yolo_data

import os
import shutil

def write_yaml(path="data.yaml"):
    with open(path, "w") as f:
        f.write("train: yolo_data/images\n")
        f.write("val: yolo_data/images\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")

def main():
    # Step 1: 초기 모델로 detect 후 이미지 저장
    detector = YOLODetector("yolo11s.pt")
    img_dir = "CV_Train/Images"
    save_dir_init = "output/init"
    os.makedirs(save_dir_init, exist_ok=True)

    for idx in range(300):
        filename = f"{idx:06d}.png"
        img_path = os.path.join(img_dir, filename)
        save_path = os.path.join(save_dir_init, filename)
        if not os.path.exists(img_path): continue

        boxes = detector.detect_person(img_path)
        detector.draw_boxes(img_path, boxes, save_path)

    # Step 2: JSON → YOLO 라벨 변환
    convert_json_to_yolo("CV_Train/Labels", "CV_Train/YOLO_Labels")

    # Step 3: 학습용 디렉토리 구성
    prepare_yolo_data("CV_Train/Images", "CV_Train/YOLO_Labels")

    # Step 4: data.yaml 생성
    write_yaml("data.yaml")

    # Step 5: 모델 학습 (Fine-tuning)
    model = YOLODetector("yolo11s.pt").model
    model.train(data="data.yaml", epochs=10, imgsz=640)

    # Step 6: 학습된 best.pt 모델 저장 (복사)
    best_model_path = "runs/detect/train/weights/best.pt"
    final_model_path = "trained_models/fine_tuned_yolo.pt"
    os.makedirs("trained_models", exist_ok=True)
    shutil.copy(best_model_path, final_model_path)
    print(f"[✔] Fine-tuned model saved to {final_model_path}")

    # Step 7: 학습된 모델로 다시 detect
    detector_tuned = YOLODetector(final_model_path)
    save_dir_final = "output/final"
    os.makedirs(save_dir_final, exist_ok=True)

    for idx in range(300):
        filename = f"{idx:06d}.png"
        img_path = os.path.join(img_dir, filename)
        save_path = os.path.join(save_dir_final, filename)
        if not os.path.exists(img_path): continue

        boxes = detector_tuned.detect_person(img_path)
        detector_tuned.draw_boxes(img_path, boxes, save_path)

    print(f"[✔] Detection results saved to: {save_dir_init} and {save_dir_final}")

if __name__ == "__main__":
    main()
