import os
import json

def sequence_to_deltas(sequence):
    """
    sequence (N,3) を dx,dy,pen の形に変換
    """
    deltas = []
    prev_x, prev_y = 0.0, 0.0  # 初期位置を (0,0) とする

    for i, (x, y, pen) in enumerate(sequence):
        if i == 0:
            dx, dy = 0.0, 0.0
        else:
            dx, dy = x - prev_x, y - prev_y
        deltas.append([float(dx), float(dy), int(pen)])
        prev_x, prev_y = x, y

    return deltas

def convert_json_folder_to_deltas(input_dir, output_dir):
    """
    input_dir の JSON (sequence形式) を読み込み、
    dx,dy に変換して output_dir に保存
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "sequence" in data:
                seq = data["sequence"]
                # dx,dy に変換
                data["sequence"] = sequence_to_deltas(seq)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"{filename} を dx,dy に変換して {output_path} に保存しました")

# ===== 使用例 =====
input_dir = "sequence_sentences"    # [x,y,pen] のJSONフォルダ
output_dir = "dxdy_normalized_sentences"      # [dx,dy,pen] に変換したJSONフォルダ

convert_json_folder_to_deltas(input_dir, output_dir)
