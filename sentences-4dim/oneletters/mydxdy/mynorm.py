

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


# 読み込み元フォルダ
input_folder = "sentences-4dim/oneletters/mydxdy/dxdy"
# 保存先フォルダ
output_folder = "sentences-4dim/oneletters/mydxdy/normdxdy"

# 入力フォルダの *.json ファイルを全部取得
json_files = glob.glob(os.path.join(input_folder, "*.json"))

for file_path in json_files:
    # ファイルを読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char = data["text"]
    strokes = data["strokes"]

    all_strokes_np = [np.array(stroke) for stroke in strokes]
    # === 標準化処理 ===
    # 1. 文字全体のデータを一つのNumpy配列に結合
    character_data = np.vstack(all_strokes_np)

    # 2. 標準化するデータ(x, y, timeなど)と、しないデータ(touching flag)を分離
    #    入力データの形式に合わせて列数を調整してください (例: [x,y,t,flag]なら4列)
    num_columns = character_data.shape[1]
    features_to_standardize = character_data[:, :num_columns-1]
    touching_flags = character_data[:, num_columns-1]
    
    # 3. 平均(mu)と標準偏差(sigma)を計算
    mu = np.mean(features_to_standardize, axis=0)
    sigma = np.std(features_to_standardize, axis=0)
    sigma[sigma == 0] = 1 # ゼロ除算を防止

    # 4. 標準化を適用
    standardized_features = (features_to_standardize - mu) / sigma

    # 5. 標準化したデータとフラグを再結合
    final_data = np.hstack([standardized_features, touching_flags.reshape(-1, 1)])
        
    # 6. 元のストローク構造に戻す
    strokes_as_list = []
    start_index = 0
    for stroke_np in all_strokes_np:
        num_points_in_stroke = len(stroke_np)
        end_index = start_index + num_points_in_stroke
        strokes_as_list.append(final_data[start_index:end_index].tolist())
        start_index = end_index

    # --- データをJSONとして保存 ---
    output_data = {
        "text": char,
        "strokes": strokes_as_list
    }
    filename = os.path.basename(file_path)
    output_data_path = os.path.join(output_folder, f'{filename}')
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✔️  Standardized '{char}': {filename}.json")
