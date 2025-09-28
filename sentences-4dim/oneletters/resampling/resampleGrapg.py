import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np  # カラーマップ用に追加

# --- 修正版: ストロークごとに色分けして描画 ---
def plot_handwriting_from_json(file_path, output_path):
    """
    JSONファイルから手書き文字のストロークデータを読み込み、
    ストロークごとに色分けしてグラフを保存します。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    strokes = data if isinstance(data, list) else data.get('strokes', [])

    if not strokes:
        return

    plt.figure()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    # ストローク数に応じてカラーマップから色を割り当て
    colors = plt.cm.tab20(np.linspace(0, 1, len(strokes)))

    for i, stroke in enumerate(strokes):
        x_coords = [p[0] for p in stroke if len(p) >= 2]
        y_coords = [p[1] for p in stroke if len(p) >= 2]

        if x_coords and y_coords:
            ax.plot(x_coords, y_coords,
                    marker='o', linestyle='-',
                    markersize=2, color=colors[i % len(colors)],
                    label=f"stroke {i+1}")

    # 軸やグリッドを非表示に
    plt.axis('off')
    plt.grid(False)

    # 凡例が必要なければコメントアウト
    # ax.legend(fontsize=6, loc='best')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# --- メイン処理 ---
input_folder = 'sentences-4dim/oneletters/resampling'
output_folder = 'handwriting_images'
os.makedirs(output_folder, exist_ok=True)

json_files = glob.glob(os.path.join(input_folder, '*.json'))

if not json_files:
    print(f"エラー: '{input_folder}' フォルダ内にJSONファイルが見つかりませんでした。")
else:
    print(f"{len(json_files)}個のJSONファイルを処理します...")

    for json_path in json_files:
        try:
            base_filename = os.path.basename(json_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            output_path = os.path.join(output_folder, f"{filename_without_ext}.png")

            plot_handwriting_from_json(json_path, output_path)

            print(f" ✓ '{output_path}' として保存しました。")
        except Exception as e:
            print(f" ❌ '{json_path}' の処理中にエラー: {e}")

    print("\nすべての処理が完了しました。")
