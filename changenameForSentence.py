import os
import json
import glob

# ================================================================
# Part 1: JSONファイル内の"text"キーの値に基づいてファイル名をリネーム
# ================================================================

# 処理したいファイルがあるフォルダのパスを指定
target_folder = 'sentences'

# 置き換えたいファイル名の「接頭辞」を指定
prefix_to_replace = 'sentence'

print(f"--- Part 1: ファイルのリネーム処理を開始します ---")
print(f"フォルダ '{os.path.abspath(target_folder)}' を処理します。")
print(f"'{prefix_to_replace}' で始まるJSONファイルをリネームします。\n")

# フォルダ内のすべての.jsonファイルを取得
search_pattern = os.path.join(target_folder, '*.json')
json_files = glob.glob(search_pattern)

if not json_files:
    print("処理対象のJSONファイルが見つかりませんでした。")
else:
    processed_count = 0
    # 取得したファイルリストをループ処理
    for filepath in json_files:
        filename = os.path.basename(filepath)

        # ファイル名が指定の接頭辞で始まるかチェック
        if not filename.startswith(prefix_to_replace):
            # 関係ないファイルはスキップ
            continue

        try:
            # ファイルを開いてJSONデータを読み込む
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # "text"キーの値を取得
            new_text = data.get('text')

            # "text"キーがあり、その値が空でない場合のみ処理
            if new_text and new_text.strip():
                # 新しいファイル名を生成 (prefixを新しいテキストに置き換え)
                # replaceの第3引数に1を指定し、最初の1箇所だけを確実に置き換えます
                new_filename = filename.replace(prefix_to_replace, new_text, 1)
                new_filepath = os.path.join(target_folder, new_filename)

                # ファイル名を変更
                os.rename(filepath, new_filepath)
                print(f"✅ 成功: {filename}  ->  {new_filename}")
                processed_count += 1
            else:
                print(f"⚠️ スキップ: {filename} ('text'キーがないか、値が空です)")

        except json.JSONDecodeError:
            print(f"❌ エラー: {filename} は有効なJSON形式ではありません。")
        except Exception as e:
            print(f"❌ エラー: {filename} の処理中に予期せぬエラーが発生しました: {e}")

    print(f"\nリネーム処理完了: {len(json_files)}個中、{processed_count}個のファイルをリネームしました。")
    print("-" * 60)


# ================================================================
# Part 2: JSONファイル内のストロークデータをΔ座標に変換
# ================================================================

print(f"\n--- Part 2: ストロークデータの変換処理を開始します ---")

def convert_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seq = []
    strokes = data["strokes"]

    # 各ストロークをΔ座標に変換
    for stroke in strokes:
        # ストロークに2点以上ないと差分が計算できない
        if len(stroke) > 1:
            for i in range(1, len(stroke)):
                dx = stroke[i]["x"] - stroke[i-1]["x"]
                dy = stroke[i]["y"] - stroke[i-1]["y"]
                seq.append([dx, dy, 0])  # end=0 で初期化

    # 文の最後だけ end=1
    if seq:
        seq[-1][-1] = 1

    return {
        "text": data["text"],
        "sequence": seq
    }

# まとめて処理
# Part 1で処理したフォルダを入力とする
input_dir = target_folder
output_dir = "processed_json"
os.makedirs(output_dir, exist_ok=True)

print(f"入力フォルダ: '{os.path.abspath(input_dir)}'")
print(f"出力フォルダ: '{os.path.abspath(output_dir)}'\n")

# input_dir 内の全 .json ファイルを対象にする
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
            # print(f"📄 変換完了: {os.path.basename(file)}")
            conversion_count += 1
        except KeyError as e:
            print(f"❌ エラー: {os.path.basename(file)} にキー '{e}' が見つかりません。スキップします。")
        except Exception as e:
            print(f"❌ エラー: {os.path.basename(file)} の変換中にエラーが発生しました: {e}")

    print(f"\n変換処理完了: {len(files_to_convert)}個中、{conversion_count}個のファイルを変換しました。")
    print("-" * 60)