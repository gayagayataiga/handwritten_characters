import json
import matplotlib.pyplot as plt


def plot_handwriting_from_json(file_path):
    """
    JSONファイルから手書き文字のストロークデータを読み込み、グラフとして表示します。
    (ストロークごとにデータがまとめられた形式に対応)

    Args:
        file_path (str): データが格納されているJSONファイルのパス。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # JSONからデータをロードする部分は同じ
        data = json.load(f)

    # 'strokes' というキーが存在しない場合も考慮
    strokes = data if isinstance(data, list) else data.get('strokes', [])

    if not strokes:
        print("ストロークデータが見つかりません。")
        return

    plt.figure()
    plt.gca().invert_yaxis()  # y軸を反転
    plt.gca().set_aspect('equal', adjustable='box')  # アスペクト比を1:1に

    # --- 変更点: ここからループの構造を大きく変更 ---

    # 外側のループで「ストローク」を1つずつ取り出す
    # strokesは [[点1, 点2,...], [点a, 点b,...]] という構造
    for stroke in strokes:

        # このストロークのx座標とy座標を格納するリストを初期化
        x_coords = []
        y_coords = []

        # 内側のループでストローク内の「点」を処理
        for point in stroke:
            # pointは [x, y, ...] というリスト
            if len(point) >= 2:  # 座標データがある点のみを対象
                x_coords.append(point[0])
                y_coords.append(point[1])

        # 1ストローク分の座標がたまったら、線としてプロット
        # pen_upフラグでの分岐は不要になる
        if x_coords and y_coords:
            plt.plot(x_coords, y_coords,
                     marker='o', linestyle='-', markersize=2)

    # --- 変更ここまで ---

    plt.title('Handwriting Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.savefig('handwriting_plot.png')
    plt.close()


# ファイル名を指定して実行
# こちらのパスはご自身の環境に合わせて修正してください
file_path = 'sentences-4dim/oneletters/normalized_xy/あ_20250924141205.json'
try:
    plot_handwriting_from_json(file_path)
    print(f"グラフを 'handwriting_plot.png' として保存しました。")
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
except Exception as e:
    print(f"エラーが発生しました: {e}")
