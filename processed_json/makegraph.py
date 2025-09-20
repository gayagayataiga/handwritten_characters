import json
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def plot_original_char(filepath, ax, char_title=""):
    """
    オリジナル座標 (x, y) で描かれた文字の軌跡を描画する関数。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"オリジナルファイル {os.path.basename(filepath)} の読み込みに失敗: {e}")
        return

    strokes = data.get("strokes", [])
    if not strokes:
        return

    for i, stroke in enumerate(strokes):
        xs = [p["x"] for p in stroke]
        ys = [p["y"] for p in stroke]
        ax.plot(xs, ys, color=plt.cm.jet(i / len(strokes)), alpha=0.5, linewidth=2.5) # ストロークごとに色を変える

    ax.set_title(f"Original: {char_title} ({os.path.basename(filepath)})", fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Y軸を反転
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)


def plot_normalized_char(filepath, ax, char_title=""):
    """
    正規化されたsequenceデータから文字の軌跡を描画する関数。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"正規化済みファイル {os.path.basename(filepath)} の読み込みに失敗: {e}")
        return

    sequence = data.get("sequence", [])
    if not sequence:
        return

    current_pos = np.array([0.0, 0.0]) # 正規化データは原点からスタート
    stroke_points = [list(current_pos)] # 最初の点を追加

    for dx, dy, flag in sequence:
        current_pos += [dx, dy]
        stroke_points.append(list(current_pos))
        
        if flag == 1: # ストロークの終わり
            points = np.array(stroke_points)
            ax.plot(points[:, 0], points[:, 1], color="blue", alpha=0.5, linewidth=2.5)
            stroke_points = [list(current_pos)] # 次のストロークは今の終点から開始

    ax.set_title(f"Normalized: {char_title} ({os.path.basename(filepath)})", fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Y軸を反転
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

# --- メインの描画処理 ---

# 元データが入っているフォルダ
original_data_dir = "sentences" 
# 正規化済みデータが入っているフォルダ (ステップ1で生成)
normalized_data_dir = "processed_json"

if not os.path.exists(original_data_dir):
    print(f"エラー: 元データフォルダ '{original_data_dir}' が見つかりません。")
elif not os.path.exists(normalized_data_dir):
    print(f"エラー: 正規化済みデータフォルダ '{normalized_data_dir}' が見つかりません。")
    print("ステップ1のデータ生成コードを先に実行して、正規化済みデータを作成してください。")
else:
    original_files = [f for f in os.listdir(original_data_dir) if f.endswith(".json")]
    
    if not original_files:
        print(f"フォルダ '{original_data_dir}' に描画対象のJSONファイルが見つかりません。")
    else:
        # ランダムに選択するファイル数
        num_to_plot = min(3, len(original_files)) # 例として3つのファイルを比較
        files_to_compare = random.sample(original_files, num_to_plot)
        
        fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, 5 * num_to_plot))
        if num_to_plot == 1: # ファイルが1つの場合、axesは1次元配列になるため調整
            axes = np.array([axes])

        for i, original_filename in enumerate(files_to_compare):
            char_text = original_filename.split(' ')[0] # ファイル名から文字を取得（例: "ゆゆゆ 1.json" -> "ゆゆゆ"）
            
            # 元データの描画
            original_filepath = os.path.join(original_data_dir, original_filename)
            plot_original_char(original_filepath, axes[i, 0], char_text)
            
            # 正規化データの描画
            normalized_filepath = os.path.join(normalized_data_dir, original_filename)
            if os.path.exists(normalized_filepath):
                plot_normalized_char(normalized_filepath, axes[i, 1], char_text)
            else:
                print(f"警告: 正規化済みファイル '{normalized_filepath}' が見つかりません。")
                axes[i, 1].set_title(f"Normalized: {char_text} (Not Found)", fontsize=10)
                axes[i, 1].set_aspect('equal', adjustable='box')
                axes[i, 1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                axes[i, 1].spines[['top', 'right', 'bottom', 'left']].set_visible(False)


        plt.tight_layout() # レイアウトを自動調整
        plt.show()