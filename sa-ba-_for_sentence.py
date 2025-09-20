from flask import Flask, request, render_template, jsonify
import os
import json
# datetimeは不要になるので削除してもOK

app = Flask(__name__)

# 保存先ディレクトリ
SAVE_DIR = "sentences"  # ディレクトリ名も分かりやすく変更
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/")
def index():
    # 前回作成した文章用のHTMLを正しく指定
    return render_template("index-sentence.html") 

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    
    # 修正点：フロントエンドから送られてくる'filename'キーを取得
    filename_from_client = data.get("filename")

    # もし何らかの理由でfilenameが送られてこなかった場合の予備処理
    if not filename_from_client:
        return jsonify({"status": "error", "message": "Filename not provided"}), 400

    # 安全対策：意図しないディレクトリに保存されるのを防ぐ (例: ../../passwords.txt)
    # os.path.basename() でファイル名部分だけを安全に抽出します
    safe_filename = os.path.basename(filename_from_client)
    
    # 保存するファイルのフルパスを生成
    save_path = os.path.join(SAVE_DIR, safe_filename)
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 成功したことをクライアントに返す
        return jsonify({"status": "ok", "filename": safe_filename})

    except Exception as e:
        # 保存中にエラーが起きた場合
        print(f"Error saving file: {e}")
        return jsonify({"status": "error", "message": "Failed to save file on server"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)