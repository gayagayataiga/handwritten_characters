import json
import glob
import os

# 読み込み元フォルダ
input_folder = "sentences-4dim/oneletters/changed_name"
# 保存先フォルダ
output_folder = "sentences-4dim/oneletters/changed_name_delete1len"

# 画数チェック
stroke_dict = {
    "あ": 3, "い": 2, "う": 2, "え": 2, "お": 3,
    "か": 3, "き": 4, "く": 1, "け": 3, "こ": 2,
    "さ": 3, "し": 1, "す": 2, "せ": 3, "そ": 1,
    "た": 4, "ち": 2, "つ": 1, "て": 1, "と": 2,
    "な": 4, "に": 3, "ぬ": 2, "ね": 2, "の": 1,
    "は": 3, "ひ": 1, "ふ": 4, "へ": 1, "ほ": 4,
    "ま": 3, "み": 2, "む": 3, "め": 2, "も": 3,
    "や": 3, "ゆ": 2, "よ": 2,
    "ら": 2, "り": 2, "る": 1, "れ": 2, "ろ": 1,
    "わ": 2, "を": 3, "ん": 1, "。": 1, "、": 1,
}


# 保存先フォルダが存在しなければ作成
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダの *.json ファイルを全部取得
json_files = glob.glob(os.path.join(input_folder, "*.json"))

for file_path in json_files:
    # ファイルを読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char = data["text"]
    strokes = data["strokes"]

    new_strokes = []
    if char in stroke_dict:
        expected = stroke_dict[char]
        if len(strokes) > expected:
            print(f"⚠ エラー: {file_path} の '{char}' は画数オーバー "
                  f"(期待={expected}, 実際={len(strokes)})")

            for stroke in data["strokes"]:          # stroke = 1本の筆跡 (リスト)
                if len(stroke) < 3:
                    print(
                        f"⚠ エラー: {file_path} の '{char}' は1画が短すぎます (点数={len(stroke)})")
                else:
                    converted = [[p["x"], p["y"], p["time"],
                                  p["isTouching"]] for p in stroke]
                    new_strokes.append(converted)
        else:
            new_strokes = [
                [[p["x"], p["y"], p["time"], p["isTouching"]] for p in stroke]
                for stroke in data["strokes"]
            ]

    data["strokes"] = new_strokes

    # 出力ファイルのパスを作成（同じファイル名で保存）
    file_name = os.path.basename(file_path)  # ファイル名だけ取り出す
    save_path = os.path.join(output_folder, file_name)

    # 保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{save_path} に保存しました")
