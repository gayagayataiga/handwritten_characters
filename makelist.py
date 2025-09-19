import json
import numpy as np
from PIL import Image, ImageDraw

def json_to_image(json_path, size=64):
    with open(json_path, "r", encoding="utf-8") as f:
        strokes = json.load(f)
    img = Image.new("L", (size, size), 255)  # 白背景
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        for i in range(1, len(stroke)):
            x1, y1 = stroke[i-1]
            x2, y2 = stroke[i]
            draw.line([x1, y1, x2, y2], fill=0, width=2)
    return np.array(img)
