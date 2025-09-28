import json
import glob
import os
import matplotlib.pyplot as plt

def plot_handwriting_from_json(file_path):
    """
    JSONファイルから手書き文字のストロークデータを読み込み、グラフとして画面に表示します。

    Args:
        file_path (str): データが格納されているJSONファイルのパス。
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

    # --- 描画処理は変更なし ---
    plt.figure()
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')

    for stroke in strokes:
        x_coords = []
        y_coords = []
        for point in stroke:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                x_coords.append(point['x'])
                y_coords.append(point['y'])

        if x_coords and y_coords:
            plt.plot(x_coords, y_coords,
                     marker='o', linestyle='-', markersize=2)

    plt.title('Handwriting Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)

    # --- 変更点2: ファイル保存に関するコードをすべて削除 ---
    # os.makedirs(output_dir, exist_ok=True)
    # base_filename = os.path.basename(file_path)
    # ... などの行を削除

    # --- 変更点3: plt.savefig() を plt.show() に変更 ---
    print(f"グラフを表示します。ウィンドウを閉じるとプログラムが終了します。")
    plt.show()
    
    # plt.close() は plt.show() の後では通常不要ですが、
    # スクリプトがこの後も続く場合はメモリ解放のために有効です。
    plt.close()

# 読み込み元フォルダ
input_folder = "sentences-4dim/oneletters/mydataCleaning/resampled"
# 保存先フォルダ
output_folder = "sentences-4dim/oneletters/mydxdy/dxdy"

# 画数チェック
stroke_dict = {
    "あ": 3,"い": 2,"う": 2,"え": 2,"お": 3,
    "か": 3,"き": 4,"く": 1,"け": 3,"こ": 2,
    "さ": 3,"し": 1,"す": 2,"せ": 3,"そ": 1,
    "た": 4,"ち": 2,"つ": 1,"て": 1,"と": 2,
    "な": 4,"に": 3,"ぬ": 2,"ね": 2,"の": 1,
    "は": 3,"ひ": 1,"ふ": 4,"へ": 1,"ほ": 4,
    "ま": 3,"み": 2,"む": 3,"め": 2,"も": 3,
    "や": 3,"ゆ": 2,"よ": 2,
    "ら": 2,"り": 2,"る": 1,"れ": 2,"ろ": 1,
    "わ": 2,"を": 3,"ん": 1,"。": 1,"、": 1,
}


# 保存先フォルダが存在しなければ作成
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダの *.json ファイルを全部取得
json_files = glob.glob(os.path.join(input_folder, "*.json"))

for file_path in json_files:
    # ファイルを読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char = data["text"]
    strokes = data["strokes"]
    
    # if char in stroke_dict:
    #     expected = stroke_dict[char]
    #     if len(strokes) > expected:
    #         print(f"⚠ エラー: {file_path} の '{char}' は画数オーバー "
    #               f"(期待={expected}, 実際={len(strokes)})")
    #         plot_handwriting_from_json(file_path)
    #         continue
    
    new_strokes = []
    zeroXpoint = data["strokes"][0][0][0]
    zeroYpoint = data["strokes"][0][0][1]
    divX = 0
    divY = 0

    for stroke in data["strokes"]:          # stroke = 1本の筆跡 (リスト)
        new_stroke = []
        for point in stroke:                
            new_stroke.append([
                round(point[0] - divX,6),
                round(point[1] - divY,6),
                round(point[2],6),
                point[3]
            ])
            divX = point[0]
            divY = point[1]
        new_strokes.append(new_stroke)

    data["strokes"] = new_strokes

    # 出力ファイルのパスを作成（同じファイル名で保存）
    file_name = os.path.basename(file_path)  # ファイル名だけ取り出す
    save_path = os.path.join(output_folder, file_name)


    # 保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{save_path} に保存しました")
