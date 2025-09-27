import torch
import json
import os
from tqdm import tqdm
import math # mathライブラリをインポート

def validate_data_final(data_dir):
    """
    データセット内のnan, inf, 非数値を完全にチェックする最終版スクリプト
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
    
    print("ファイルを1つずつ厳密にチェックしています (最終版)...")
    for file_path in tqdm(files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for n_idx, stroke_group in enumerate(data["strokes"]):
            for t_idx, point in enumerate(stroke_group):
                # 4つの特徴量それぞれをチェック
                for f_idx, val in enumerate(point):
                    is_problem = False
                    problem_type = ""
                    
                    # 1. 型チェック (intかfloatか)
                    if not isinstance(val, (int, float)):
                        is_problem = True
                        problem_type = "非数値"
                    # 2. 値チェック (nanか)
                    elif math.isnan(val):
                        is_problem = True
                        problem_type = "NaN"
                    # 3. 値チェック (infか)
                    elif math.isinf(val):
                        is_problem = True
                        problem_type = "Infinity"

                    if is_problem:
                        print(f"\nエラー: {problem_type} の値が見つかりました！")
                        print(f"ファイル名: {file_path}")
                        print(f"ストローク群のインデックス(N): {n_idx}")
                        print(f"点のインデックス(T): {t_idx}")
                        print(f"特徴量のインデックス(F): {f_idx}")
                        print(f"問題のデータ点: {point}")
                        raise ValueError(f"Invalid data ({problem_type}) found in JSON file.")

    print("全ファイルのチェックが完了しました。データに問題は見つかりませんでした。")

if __name__ == '__main__':
    DATA_DIR = "sentences-4dim/oneletters/resampling"
    try:
        validate_data_final(DATA_DIR)
    except ValueError as e:
        print(e)