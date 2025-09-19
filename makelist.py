import os
import shutil

# ベースのフォルダ（JSONファイルが入っている場所）
base_dir = "hiragana"

# JSONファイルを走査
for filename in os.listdir(base_dir):
    if filename.endswith(".json"):
        # 先頭の1文字を取得
        first_char = filename[0]

        # 移動先のフォルダパスを作成
        target_dir = os.path.join(base_dir, first_char)

        # フォルダが存在しなければ作成
        os.makedirs(target_dir, exist_ok=True)

        # 元ファイルのパスと移動先のパス
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        # ファイルを移動
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename} -> {target_dir}")
