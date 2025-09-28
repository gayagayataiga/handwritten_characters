import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のため

# --- 設定 ---
INPUT_DIR = 'sentences-4dim/oneletters/mydxdy/normdxdy' # 標準化済みのデータが入ったフォルダ
OUTPUT_DIR = 'sentences-4dim/oneletters/mydxdy/normdxdy_image'     # 生成した画像を保存するフォルダ

# 出力用フォルダが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_data(json_path):
    """
    標準化済みのJSONファイルを読み込み、プロットを画像として保存する関数
    """
    base_filename = os.path.splitext(os.path.basename(json_path))[0]
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        strokes = data['strokes']
        char_text = data['text']
        
        # --- 描画の準備 ---
        plt.figure(figsize=(6, 6))
        
        # --- 各ストロークを描画 ---
        for i, stroke in enumerate(strokes):
            # strokeは点のリストなので、Numpy配列に変換
            stroke_array = np.array(stroke)
            
            # x座標(0列目)とy座標(1列目)を抽出
            # ペン上げ情報(最後の点)も含めてプロットすることで、ペンの移動軌跡も可視化
            x = stroke_array[:, 0]
            y = stroke_array[:, 1]
            
            # ストロークを線で描画
            plt.plot(x, y, linestyle='-', marker='.', markersize=4, color='blue', 
                     label='Stroke' if i == 0 else "")
            
            # 始点を緑の丸でプロット
            plt.plot(x[0], y[0], 'o', color='green', markersize=8, 
                     label='Start Point' if i == 0 else "")
            
            # 終点を赤の丸でプロット
            plt.plot(x[-1], y[-1], 'o', color='red', markersize=8, fillstyle='none',
                     label='End Point' if i == 0 else "")

        # --- グラフ全体の見た目を設定 ---
        plt.title(f'Standardized Plot for "{char_text}"')
        plt.xlabel('Standardized X')
        plt.ylabel('Standardized Y')
        plt.grid(True)
        plt.legend()
        # 座標系の原点が左上に来るようにy軸を反転
        plt.gca().invert_yaxis()
        plt.axis('equal') # xとyのスケールを合わせ、正しい形で表示

        # --- ファイルに保存 ---
        output_path = os.path.join(OUTPUT_DIR, f'{base_filename}.png')
        plt.savefig(output_path)
        plt.close() # メモリ解放

        print(f"✔️  Visualized and saved: {base_filename}.png")

    except Exception as e:
        print(f"❌ An unexpected error occurred with {base_filename}.json: {e}")

# --- スクリプトの実行 ---
if __name__ == '__main__':
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    for file_path in json_files:
        visualize_data(file_path)
    print("\nAll visualizations created.")