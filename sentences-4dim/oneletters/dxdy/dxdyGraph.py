import json
import os
import matplotlib.pyplot as plt


def plot_handwriting_from_json(file_path, output_dir='.'):
    """
    JSONファイルから手書き文字のストロークデータを読み込み、グラフとして指定されたフォルダに保存します。

    Args:
        file_path (str): データが格納されているJSONファイルのパス。
        output_dir (str): 生成したグラフ画像（PNG）を保存するフォルダのパス。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return
    except json.JSONDecodeError:
        print(f"エラー: ファイル '{file_path}' は有効なJSON形式ではありません。")
        return

    strokes = data.get('strokes', [])
    if not strokes:
        print("ストロークデータが見つかりません。")
        return

    plt.figure()
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')

    # 外側のループで「ストローク」を1つずつ取り出す
    for stroke in strokes:
        x_coords = []
        y_coords = []

        # 内側のループでストローク内の「点」を処理
        # 点が辞書形式 {"x": ..., "y": ...} になっていることに対応
        for point in stroke:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                x_coords.append(point['x'])
                y_coords.append(point['y'])

        # 1ストローク分の座標がたまったら、線としてプロット
        if x_coords and y_coords:
            plt.plot(x_coords, y_coords,
                     marker='o', linestyle='-', markersize=2)

    plt.title('Handwriting Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)

    # --- 保存処理の変更点 ---
    # 1. 保存先のフォルダが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 2. 保存ファイル名を元のJSONファイル名から生成する
    base_filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_filename = f"{filename_without_ext}.png"

    # 3. フォルダパスとファイル名を結合して保存パスを生成
    save_path = os.path.join(output_dir, output_filename)

    plt.savefig(save_path)
    plt.close()
    print(f"グラフを '{save_path}' として保存しました。")


# --- ここから実行部分 ---

# 読み込むJSONファイルのパス
json_file_path = 'sentences-4dim/oneletters/dxdy/あ_20250924141205.json'

# 保存したいフォルダのパスを指定
# 例: 'images' フォルダに保存したい場合
output_folder = 'images/handwriting/dxdy'

# 描画関数を実行
plot_handwriting_from_json(json_file_path, output_folder)
