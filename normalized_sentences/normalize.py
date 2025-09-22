import os
import json
import numpy as np

def normalize_strokes(strokes):
    """strokes全体をまとめてmin-max正規化 (0〜1)、x,yのみ"""
    all_points = []
    for stroke in strokes:
        for pt in stroke:
            all_points.append([pt["x"], pt["y"]])
    
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
            x, y = norm_arr[idx]
            new_stroke.append({"x": float(x), "y": float(y)})
            idx += 1
        result.append(new_stroke)
    
    return result

def process_json_folder(input_dir, output_dir, normalize=False):
    """
    input_dir の JSON を output_dir にコピーする。
    normalize=True の場合は strokes を正規化して保存。
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if normalize and "strokes" in data:
                data["strokes"] = normalize_strokes(data["strokes"])
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            mode = "正規化" if normalize else "コピー"
            print(f"{filename} を{mode}して {output_path} に保存しました")

# ===== 使用例 =====
input_dir = "sentences"             # 元のjsonフォルダ
output_dir_raw = "copied_sentences" # コピーのみ
output_dir_norm = "normalized_sentences" # 正規化して保存

# コピーするだけ
process_json_folder(input_dir, output_dir_raw, normalize=False)

# 正規化して保存
process_json_folder(input_dir, output_dir_norm, normalize=True)
