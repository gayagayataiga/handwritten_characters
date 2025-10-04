import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob # ファイルパスのパターンマッチング用

def plot_stroke_data(strokes, output_path, text=""):
    """
    ストロークデータをグラフにプロットし、画像として保存する関数。
    dx, dy, dp, pen_down の形式に対応。
    """
    if not strokes:
        print(f"警告: {output_path} にストロークデータがありません。")
        return

    # 絶対座標に変換するための初期値
    current_x, current_y = 0.0, 0.0
    
    plt.figure(figsize=(6, 6)) # 画像サイズを調整
    ax = plt.gca()
    
    all_x = []
    all_y = []

    for stroke_segment in strokes: # 各ストロークセグメント（例: ひらがなの1画、2画...）
        segment_points_x = []
        segment_points_y = []

        # 最初の点はpen_down=1.0のときに描画を開始
        pen_is_down = True # ストロークセグメントの最初の点では、常にペンが下りていると仮定
                          # もし[dx, dy, dp, 0.0]で始まる場合もあるなら調整が必要

        for i, point in enumerate(stroke_segment):
            # dx, dy, dp, pen_down_flag
            dx, dy, _, pen_down_flag = point

            # ストロークデータのdx, dyは一般的に「次に動く量」なので、現在の位置に加算する
            # ただし、グラフのy軸は上方向が正ですが、手書きデータでは下方向が正の場合も多いので、
            # グラフの見た目が上下反転する場合があるかもしれません。
            # その場合は `current_y -= dy` に変更を検討してください。
            current_x += dx
            current_y += dy # Y軸は上方向が正なので、手書きのY軸方向と合わせるか注意

            all_x.append(current_x)
            all_y.append(current_y)

            # pen_down_flag が 1.0 の場合は線を引く
            if pen_is_down and pen_down_flag == 1.0:
                segment_points_x.append(current_x)
                segment_points_y.append(current_y)
            
            # pen_down_flag が 0.0 の場合、そのストロークは終了
            # 次の点はペンアップ移動と見なす
            if pen_down_flag == 0.0:
                if segment_points_x: # 描画すべき点がある場合のみプロット
                    plt.plot(segment_points_x, segment_points_y, color='black', linewidth=2)
                
                # ペンアップ後の移動で座標だけ更新し、新しいストロークセグメントのためにリセット
                segment_points_x = []
                segment_points_y = []
                pen_is_down = False # 次の点の移動はペンアップ状態
            else:
                pen_is_down = True # ペンダウン継続

        # ストロークセグメントの終わりに残っている点があればプロット
        if segment_points_x:
            plt.plot(segment_points_x, segment_points_y, color='black', linewidth=2)

    # 軸の表示範囲を調整（すべての点が含まれるように）
    if all_x and all_y:
        # グラフの縦横比を固定するために、両軸で同じスケールを使用
        max_range = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y))
        
        center_x = (np.max(all_x) + np.min(all_x)) / 2
        center_y = (np.max(all_y) + np.min(all_y)) / 2

        padding = max_range * 0.1 # 10%のパディングを追加

        ax.set_xlim(center_x - max_range / 2 - padding, center_x + max_range / 2 + padding)
        ax.set_ylim(center_y - max_range / 2 - padding, center_y + max_range / 2 + padding)
        
        # Y軸を反転させることで、手書き文字が上から下へ書かれるイメージに合わせることが多い
        ax.invert_yaxis() 
    else:
        # データがない場合のためにデフォルトの範囲を設定
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.invert_yaxis()

    plt.axis('off') # 軸の目盛りやラベルを非表示にする
    plt.title(f"Text: {text}", fontsize=10) # タイトルとして文字を表示
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close() # メモリリークを防ぐためにプロットを閉じる

def process_all_json_files(input_folder, output_folder):
    """
    指定されたフォルダ内のすべてのJSONファイルを処理し、画像を保存する。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = glob.glob(os.path.join(input_folder, '*.json'))
    
    if not json_files:
        print(f"入力フォルダ '{input_folder}' にJSONファイルが見つかりませんでした。")
        return

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            strokes = data.get('strokes')
            text = data.get('text', 'N/A')
            
            # 出力ファイル名を作成 (元のJSONファイル名から拡張子を除き、.pngを追加)
            base_name = os.path.basename(json_file)
            output_filename = os.path.splitext(base_name)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            
            plot_stroke_data(strokes, output_path, text)
            print(f"'{json_file}' を処理し、'{output_path}' に保存しました。")

        except json.JSONDecodeError:
            print(f"エラー: '{json_file}' は有効なJSONファイルではありません。スキップします。")
        except Exception as e:
            print(f"エラー: '{json_file}' の処理中に問題が発生しました: {e}")

if __name__ == '__main__':
    # 【ここを編集してください】
    input_folder_path = 'sentences-4dim/oneletters/mydxdy/addtextids' # JSONファイルが保存されているフォルダのパス
    output_folder_path = 'sentences-4dim/oneletters/mydxdy/addtextids_image' # 生成された画像を保存するフォルダのパス

    process_all_json_files(input_folder_path, output_folder_path)
    print("すべての処理が完了しました。")