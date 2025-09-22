import os
import json

def strokes_to_sequence(strokes):
    """
    strokes(JSONのリスト)を sequence (N,3) リストに変換
    各点は [x, y, pen_state]
    - ペンを下ろしている間: pen=1
    - ストロークの最後の点: pen=0
    """
    seq = []
    for stroke in strokes:
        for i, pt in enumerate(stroke):
            x, y = pt["x"], pt["y"]
            pen = 0 if i == len(stroke) - 1 else 1
            seq.append([float(x), float(y), pen])
    return seq

def convert_json_folder(input_dir, output_dir):
    """
    input_dir の JSON を読み込み、sequence形式に変換して output_dir に保存
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "strokes" in data:
                strokes = data["strokes"]
                # strokes → sequence に変換
                data["sequence"] = strokes_to_sequence(strokes)
                # strokes は不要なら削除
                del data["strokes"]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"{filename} を変換して {output_path} に保存しました")

# ===== 使用例 =====
input_dir = "normalized_sentences"   # 正規化済みJSONフォルダ
output_dir = "sequence_sentences"    # sequence形式に変換して保存

convert_json_folder(input_dir, output_dir)
