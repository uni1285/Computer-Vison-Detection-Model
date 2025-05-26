import os
import json

def convert_json_to_yolo(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(input_dir):
        if not json_file.endswith(".json"):
            continue
        path = os.path.join(input_dir, json_file)
        with open(path, 'r') as f:
            data = json.load(f)

        img_w = data['imageWidth']
        img_h = data['imageHeight']
        shapes = data['shapes']

        lines = []
        for shape in shapes:
            (x1, y1), (x2, y2) = shape['points']
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(output_dir, json_file.replace(".json", ".txt")), 'w') as f:
            f.write("\n".join(lines))
