import os
import json
import glob

def check_strokes_length(data_dir):
    """
    指定フォルダ内の JSON ファイルをチェックし、
    strokes の次元が不揃いなデータを表示する。
    """
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            strokes = data.get("strokes", [])

            # 各 stroke の長さを調べる
            lengths = [len(s) for s in strokes]

            # 不一致がある場合だけ報告
            if len(set(lengths)) > 1:
                print(f"⚠ {file_path}")
                print(f"   strokeごとの長さ: {lengths}")

        except Exception as e:
            print(f"❌ 読み込み失敗: {file_path}, error: {e}")

# 使用例
data_dir = "sentences-4dim/oneletters/mydxdy/dxdy"  # JSONフォルダのパスに変更
check_strokes_length(data_dir)
