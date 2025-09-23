import os
import json
import re

# --- 設定項目 ---
# JSONファイルが保存されているフォルダのパスを指定してください
# 例: 'C:/Users/YourUser/Documents/handwriting_data' や './data' など
TARGET_DIRECTORY = './sentences-4dim'  # フォルダ名を分かりやすく変更
# ----------------


def sanitize_filename(text):
    """ファイル名として安全な文字列に変換する関数"""
    if not isinstance(text, str):
        text = str(text)

    # 1. 前後の空白を削除
    text = text.strip()

    # 2. ファイル名として使えない文字をアンダースコア(_)に置換
    #    Windows/macOSで一般的に使えない文字に対応
    text = re.sub(r'[\\/:*?"<>|]', '_', text)

    # 3. 連続する空白文字(スペースなど)をアンダースコア(_)に置換
    text = re.sub(r'\s+', '_', text)

    # 4. ファイル名の長さを制限 (長すぎると問題になることがあるため)
    #    日本語は1文字3バイト以上になることがあるので、バイト数で制限する方が安全ですが、
    #    ここでは簡潔にするため文字数で制限します。
    return text[:50]  # 先頭から50文字まで


def rename_files_in_directory(directory):
    """指定されたディレクトリ内のJSONファイルをリネームする"""
    print(f"'{directory}' フォルダ内のファイルを処理します...")

    if not os.path.isdir(directory):
        print(f"エラー: 指定されたフォルダ '{directory}' が見つかりません。")
        return

    renamed_count = 0
    for filename in os.listdir(directory):
        save_directory = os.path.join(directory + "/changed_name")
        # 'sentence_'で始まり'.json'で終わるファイルのみを対象とする
        if filename.startswith('sentence_') and filename.endswith('.json'):
            old_filepath = os.path.join(directory, filename)

            try:
                # UTF-8でJSONファイルを開く
                with open(old_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                text_content = data.get('text')

                if not text_content:
                    print(f"スキップ: '{filename}' に 'text' キーが見つかりません。")
                    continue

                # タイムスタンプ部分を元のファイル名から抽出
                # 'sentence_' (9文字) と '.json' (5文字) を除外
                timestamp = filename[9:-5]

                # 新しいファイル名を生成
                sanitized_text = sanitize_filename(text_content)
                new_filename = f"{sanitized_text}_{timestamp}.json"
                new_filepath = os.path.join(save_directory, new_filename)

                # 新旧のパスが同じ場合は何もしない
                if old_filepath == new_filepath:
                    print(f"スキップ: '{filename}' はリネームの必要がありません。")
                    continue

                # ファイルをリネーム
                os.rename(old_filepath, new_filepath)
                print(f"成功: '{filename}' -> '{new_filename}'")
                renamed_count += 1

            except json.JSONDecodeError:
                print(f"エラー: '{filename}' は有効なJSONファイルではありません。")
            except Exception as e:
                print(f"エラー: '{filename}' の処理中に問題が発生しました - {e}")

    print(f"\n処理が完了しました。{renamed_count}個のファイルをリネームしました。")


# --- メイン処理 ---
if __name__ == "__main__":
    rename_files_in_directory(TARGET_DIRECTORY)
