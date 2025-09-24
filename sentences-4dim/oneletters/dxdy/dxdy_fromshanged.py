import os
import json
import numpy as np


def dxdy_strokes(strokes):
    """strokes全体をまとめてmin-max正規化 (0〜1)、x,yのみ"""
    all_points = []
    for stroke in strokes:
        for pt in stroke:
            all_points.append(
                [pt["x"]-stroke[0]["x"], pt["y"]-stroke[0]["y"], pt["time"], pt["isTouching"]])

    arr = np.array(all_points, dtype=float)

    # x,y の min-max
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    norm_arr = (arr - min_vals) / (max_vals - min_vals + 1e-9)

    # 元の strokes に戻す
    result = []
    idx = 0
    for stroke in strokes:
        new_stroke = []
        for pt in stroke:
            x, y, time, isTouching = norm_arr[idx]
            if time < 0.001:
                time = 0.001  # time は最小値を0.001にする
            if isTouching < 0.5:
                isTouching = 0
            else:
                isTouching = 1
            new_stroke.append({"x": float(x),
                               "y": float(y),
                               "time": float(time),
                               "isTouching": int(isTouching)})
            idx += 1
        result.append(new_stroke)

    return result


def process_json_folder(input_dir, output_dir, normalize=False):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if normalize and "strokes" in data:
                data["strokes"] = dxdy_strokes(data["strokes"])

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            mode = "正規化" if normalize else "コピー"
            print(f"{filename} を{mode}して {output_path} に保存しました")


# ===== 使用例 =====
input_dir = "sentences-4dim/oneletters/changed_name"             # 元のjsonフォルダ
output_dir_norm = "sentences-4dim/oneletters/dxdy"  # 正規化して保存

process_json_folder(input_dir, output_dir_norm, normalize=True)
