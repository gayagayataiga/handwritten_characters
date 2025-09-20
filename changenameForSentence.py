# 【修正版】データ生成コード
import os
import json
import glob
import numpy as np

# ================================================================
# Part 1: ファイルのリネーム処理 (変更なし)
# ================================================================
target_folder = 'sentences'
prefix_to_replace = 'sentence'
print(f"--- Part 1: ファイルのリネーム処理を開始します ---")
# (処理内容は長いため省略しますが、前回のコードと同じです)
search_pattern = os.path.join(target_folder, '*.json')
json_files = glob.glob(search_pattern)
if not json_files:
    print("処理対象のJSONファイルが見つかりませんでした。")
else:
    processed_count = 0
    for filepath in json_files:
        filename = os.path.basename(filepath)
        if not filename.startswith(prefix_to_replace):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            new_text = data.get('text')
            if new_text and new_text.strip():
                new_filename = filename.replace(prefix_to_replace, new_text, 1)
                new_filepath = os.path.join(target_folder, new_filename)
                os.rename(filepath, new_filepath)
                # print(f"✅ 成功: {filename}  ->  {new_filename}")
                processed_count += 1
        except Exception as e:
            print(f"❌ エラー: {filename} の処理中にエラーが発生しました: {e}")
    print(f"リネーム処理完了: {processed_count}個のファイルをリネームしました。")
    print("-" * 60)


# ================================================================
# Part 2: ストロークデータを変換・正規化 (★★ここを修正★★)
# ================================================================
print(f"\n--- Part 2: ストロークデータの変換と正規化処理を開始します ---")

def convert_file(path):
    """
    JSONファイルを読み込み、ストロークデータを
    正規化済みのΔ座標シーケンスに変換する関数 (ストローク区切り対応版)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strokes = data.get("strokes", [])
    if not strokes:
        return {"text": data.get("text", ""), "sequence": []}

    # --- ステップ1: 全てのΔ座標(dx, dy)を一時的に収集 ---
    all_dx_dy = []
    for stroke in strokes:
        if len(stroke) > 1:
            for i in range(1, len(stroke)):
                dx = stroke[i]["x"] - stroke[i-1]["x"]
                dy = stroke[i]["y"] - stroke[i-1]["y"]
                all_dx_dy.append([dx, dy])

    if not all_dx_dy:
        return {"text": data.get("text", ""), "sequence": []}

    # --- ステップ2: 収集した全座標データで正規化 ---
    coords_np = np.array(all_dx_dy, dtype=float)
    mean = coords_np.mean()
    std = coords_np.std()
    if std < 1e-8:
        std = 1.0 # ゼロ除算を防止
    normalized_coords_np = (coords_np - mean) / std

    # --- ステップ3: 正規化後のデータにストローク終了フラグを付けて最終形式にする ---
    final_sequence = []
    point_idx_counter = 0
    for stroke in strokes:
        if len(stroke) > 1:
            num_points_in_stroke = len(stroke) - 1
            for i in range(num_points_in_stroke):
                norm_dx, norm_dy = normalized_coords_np[point_idx_counter]
                # この点がストロークの最後かどうかを判定
                is_end_of_stroke = (i == num_points_in_stroke - 1)
                flag = 1 if is_end_of_stroke else 0
                final_sequence.append([norm_dx, norm_dy, flag])
                point_idx_counter += 1

    return {
        "text": data.get("text", ""),
        "sequence": final_sequence
    }

# まとめて処理
input_dir = target_folder
output_dir = "processed_json"
os.makedirs(output_dir, exist_ok=True)
print(f"入力フォルダ: '{os.path.abspath(input_dir)}'")
print(f"出力フォルダ: '{os.path.abspath(output_dir)}'\n")

files_to_convert = glob.glob(os.path.join(input_dir, "*.json"))
if not files_to_convert:
    print("変換対象のJSONファイルが見つかりませんでした。")
else:
    conversion_count = 0
    for file in files_to_convert:
        try:
            converted = convert_file(file)
            out_path = os.path.join(output_dir, os.path.basename(file))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(converted, f, ensure_ascii=False, indent=2)
            conversion_count += 1
        except Exception as e:
            print(f"❌ エラー: {os.path.basename(file)} の変換中にエラーが発生しました: {e}")
    print(f"変換・正規化処理完了: {conversion_count}個のファイルを処理しました。")
    print("-" * 60)