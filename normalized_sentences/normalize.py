import os
import json
import numpy as np

def normalize_strokes(strokes):
    """strokes全体をまとめてmin-max正規化 (0〜1)、x,yのみ"""
    # すべての (x,y) を収集
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
            new_stroke.append({"x": x, "y": y})
            idx += 1
        result.append(new_stroke)
    
    return result

def normalize_json_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "strokes" in data:
                data["strokes"] = normalize_strokes(data["strokes"])
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"{filename} を正規化して {output_path} に保存しました")

# ===== 使用例 =====
input_dir = "sentences"       # 元のjsonフォルダ
output_dir = "normalized_sentences" # 保存先フォルダ
normalize_json_folder(input_dir, output_dir)
