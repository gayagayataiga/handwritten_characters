from flask import Flask, request, render_template, jsonify
import os
import json
import random

RANDOM_SENTENCES = [
    # 既存の文章
    "今日はいい天気です。",
    "明日は雨が降るかも。",
    "プログラミングは楽しい。",
    "テスト用の文章です。",
    "AIと一緒に学習中。",

    # 日常・挨拶
    "おはようございます。",
    "お疲れ様です。",
    "少し休憩しましょう。",
    "今日の予定は何ですか？",
    "よろしくお願いします。",
    "お腹が空きました。",
    "お先に失礼します。",
    "また明日ね。",

    # 大学生活・研究室
    "課題が終わらない…。",
    "実験レポートを提出。",
    "明日は1限からだ。",
    "この講義、楽単？",
    "研究室に配属された。",
    "参考文献を探そう。",
    "学食で昼飯にする。",
    "論文をサーベイする。",
    "ゼミの発表準備をする。",
    "中間試験が近い。",
    "徹夜で間に合わせる。",
    "この単位は落とせない。",

    # プログラミング・Python関連
    "コードを書いてみよう。",
    "デバッグが必要です。",
    "これはPythonです。",
    "変数とは何ですか？",
    "ループ処理を実装する。",
    "関数を定義します。",
    "計算量が多すぎる。",
    "アルゴリズムを改善。",
    "APIを叩いてみる。",
    "バージョン管理は大事。",
    "セグフォで落ちた。",
    "プルリクお願いします。",
    "仮想環境を構築する。",
    "リファクタリングしたい。",

    # 情報科学の基礎
    "データ構造とアルゴリズム。",
    "計算量理論は面白い。",
    "OSの仕組みを学ぶ。",
    "データベースを設計する。",
    "ネットワークは難しい。",
    "再帰呼び出しを実装。",
    "ハッシュテーブルを使う。",
    "探索アルゴリズムを書く。",

    # 機械学習・AI関連
    "モデルを学習させる。",
    "これは深層学習です。",
    "データセットが必要です。",
    "過学習に気をつけよう。",
    "AIは面白いです。",
    "精度を評価します。",
    "ハイパラ調整しよう。",
    "損失関数を定義する。",
    "勾配消失問題だ。",
    "転移学習を使ってみる。",
    "アノテーション作業がつらい。",
    "特徴量エンジニアリング。",
    "GANで画像を生成する。",
    "推論の速度も重要。",

    # ロボット・工学関連
    "モーターが動かない。",
    "回路を設計します。",
    "センサーの値を見る。",
    "ロボットを組み立てる。",
    "ハンダ付けは楽しい。",
    "機構を考えます。",
    "CADで図面を引く。",
    "PID制御を調整する。",
    "マイコンに書き込む。",
    "3Dプリンタで出力。",
    "逆運動学を解く。",
    "オシロで波形を見る。",
    "モータードライバが焼けた。",
    "シミュレーションでは動いた。",

    # テニス関連
    "サーブの練習をしよう。",
    "ナイスショットです！",
    "次の試合は勝ちたい。",
    "ストロークを安定させる。",
    "ダブルスをしましょう。",
    "フットワークが重要。",
    "もっとスピンをかけたい。",
    "ボレーに出よう。",
    "セカンドは確実に入れる。",
    "深く狙っていこう。",
    "ブレークポイントだ！",
    "ガットが切れたかも。",
    "雁行陣でいこう。",
    "センターにサーブを集める。",

    # 日常・その他
    "サークルの飲み会だ。",
    "そろそろバイト行かなきゃ。",
    "マジで？信じられない。",
    "お昼何食べる？",
    "ちょっとコンビニ行ってくる。",
    "今日の夜、暇？",
]

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

@app.route("/random_sentence", methods=["GET"])
def random_sentence():
    sentence = random.choice(RANDOM_SENTENCES)
    return jsonify({"sentence": sentence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)