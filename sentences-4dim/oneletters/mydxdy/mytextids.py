

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

vocab = {"あ": 1, "い": 2, "う": 3, "え": 4, "お": 5,
             "か": 6, "き": 7, "く": 8, "け": 9, "こ": 10,
             "さ": 11, "し": 12, "す": 13, "せ": 14, "そ": 15,
             "た": 16, "ち": 17, "つ": 18, "て": 19, "と": 20,
             "な": 21, "に": 22, "ぬ": 23, "ね": 24, "の": 25,
             "は": 26, "ひ": 27, "ふ": 28, "へ": 29, "ほ": 30,
             "ま": 31, "み": 32, "む": 33, "め": 34, "も": 35,
             "や": 36, "ゆ": 37, "よ": 38,
             "ら": 39, "り": 40, "る": 41, "れ": 42, "ろ": 43,
             "わ": 44, "を": 45, "ん": 46, "。": 47, "、": 48,}  # 必要に応じて拡張


# 読み込み元フォルダ
input_folder = "sentences-4dim/oneletters/mydxdy/normdxdy"
# 保存先フォルダ
output_folder = "sentences-4dim/oneletters/mydxdy/addtextids"

# 入力フォルダの *.json ファイルを全部取得
json_files = glob.glob(os.path.join(input_folder, "*.json"))

for file_path in json_files:
    # ファイルを読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char = data["text"]
    strokes = data["strokes"]

    # 文字をIDに変換
    text_ids = [vocab.get(c, 0) for c in char]  # 未知の文字は0にマッピング

    # --- データをJSONとして保存 ---
    output_data = {
        "text": char,
        "text_ids": text_ids,
        "strokes": data["strokes"]
    }
    filename = os.path.basename(file_path)
    output_data_path = os.path.join(output_folder, f'{filename}')
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✔️  Standardized '{char}': {filename}.json")
