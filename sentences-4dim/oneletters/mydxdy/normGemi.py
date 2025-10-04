import json
import glob
import os
import numpy as np


# 読み込み元フォルダ
input_folder = "sentences-4dim/oneletters/mydxdy/dxdy"
# 保存先フォルダ
output_folder = "sentences-4dim/oneletters/mydxdy/normdxdy"


def normalize_character(strokes):
    """
    文字のストロークデータを正規化する関数。
    アスペクト比を維持し、[0, 1]の範囲にスケーリングする。
    """
    # 1. 文字全体のデータを一つのNumpy配列に結合
    all_points = np.vstack([np.array(stroke) for stroke in strokes])
    
    # 2. x, y座標のみを抽出 (ここでは [x, y, t, flag] の4次元を想定)
    coords = all_points[:, :2]
    
    # 3. 文字全体のバウンディングボックス（最小・最大のx,y座標）を計算
    min_xy = np.min(coords, axis=0)
    max_xy = np.max(coords, axis=0)
    
    # 4. 文字の幅と高さを計算
    width_height = max_xy - min_xy
    
    # ゼロ除算を防止
    width_height[width_height == 0] = 1
    
    # 5. アスペクト比を維持するため、幅と高さの大きい方をスケールに使う
    scale = np.max(width_height)
    
    # 6. すべてのストロークに対して正規化を適用
    normalized_strokes = []
    for stroke in strokes:
        stroke_np = np.array(stroke)
        
        # x, y 座標を正規化
        normalized_coords = (stroke_np[:, :2] - min_xy) / scale
        
        # 元の時間やフラグ情報と結合
        # (入力データの形式に合わせて列数を調整してください)
        num_columns = stroke_np.shape[1]
        if num_columns > 2:
            other_features = stroke_np[:, 2:]
            normalized_stroke = np.hstack([normalized_coords, other_features])
        else:
            normalized_stroke = normalized_coords
            
        normalized_strokes.append(normalized_stroke.tolist())
        
    return normalized_strokes

# --- メインのループ処理 ---
json_files = glob.glob(os.path.join(input_folder, "*.json"))

for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    char = data["text"]
    strokes = data["strokes"]
    
    # === 正規化処理の呼び出し ===
    normalized_strokes_list = normalize_character(strokes)
    
    # --- データをJSONとして保存 ---
    output_data = {
        "text": char,
        "strokes": normalized_strokes_list
    }
    filename = os.path.basename(file_path)
    output_data_path = os.path.join(output_folder, filename) # f-stringの修正
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✔️  Normalized '{char}': {filename}")