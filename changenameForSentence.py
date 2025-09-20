import os
import json
import glob

# 処理したいファイルがあるフォルダのパスを指定
target_folder = 'sentences'

# 置き換えたいファイル名の「接頭辞」を指定
prefix_to_replace = 'sentence'

print(f"フォルダ '{os.path.abspath(target_folder)}' の処理を開始します。")
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

    print(f"\n処理完了: {len(json_files)}個中、{processed_count}個のファイルをリネームしました。")