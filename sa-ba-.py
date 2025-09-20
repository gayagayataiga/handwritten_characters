from flask import Flask, request, render_template, jsonify
import os
import json
import datetime

app = Flask(__name__)

SAVE_DIR = "hiragana"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index-sentence.html")

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    char = data.get("char", "unknown")  # 選んだ文字を取得
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/{char}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok", "filename": filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
