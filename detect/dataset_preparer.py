import os
import shutil

def prepare_yolo_data(image_dir, label_dir, dest_dir="yolo_data"):
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copytree(image_dir, os.path.join(dest_dir, "images"), dirs_exist_ok=True)
    shutil.copytree(label_dir, os.path.join(dest_dir, "labels"), dirs_exist_ok=True)

