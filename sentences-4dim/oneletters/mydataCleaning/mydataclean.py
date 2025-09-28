# スプライン補間を行う
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.interpolate import CubicSpline

INPUT_DIR = 'sentences-4dim/oneletters/mydxdy/checklength'  # 入力フォルダ   
OUTPUT_DIR = 'sentences-4dim/oneletters/mydataCleaning/images/' 
SAVING_DATA_DIR = 'sentences-4dim/oneletters/mydataCleaning/resampled/'
NUM_POINTS = 10  # 補間後の点の数

def add_noise_to_stroke(stroke, noise_level=1.0, time_noise=0.01, shift_level=2.0):
    """
    ストロークにノイズを付与する関数
    Args:
        stroke: np.array [N, 4] (x, y, time, touching)
        noise_level: 座標ごとのランダム揺らぎの標準偏差
        time_noise: 時間軸の揺らぎの標準偏差
        shift_level: 全体の平行移動の範囲
    """
    noisy_stroke = stroke.copy()

    # --- 座標ごとのガウスノイズ ---
    noisy_stroke[:, 0] += np.random.normal(0, noise_level, size=len(stroke))  # x
    noisy_stroke[:, 1] += np.random.normal(0, noise_level, size=len(stroke))  # y

    # --- 時間に少しノイズ ---
    noisy_stroke[:, 2] += np.random.normal(0, time_noise, size=len(stroke))

    # --- 全体のランダム平行移動 ---
    shift_x = np.random.uniform(-shift_level, shift_level)
    shift_y = np.random.uniform(-shift_level, shift_level)
    noisy_stroke[:, 0] += shift_x
    noisy_stroke[:, 1] += shift_y

    return noisy_stroke



def process_and_save_plot(json_path, output_dir):
    """
    一つのJSONファイルを読み込み、スプライン補間のグラフを生成して保存する関数
    """
    try:
        # --- ファイルの読み込み ---
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        strokes = data.get('strokes', [])
        char_text = data.get('text', 'Unknown')
        
        # --- 描画の準備 ---
        plt.figure(figsize=(8, 8))
        is_first_stroke = True

        # --- 各ストロークに対して補間処理を実行 ---
        for stroke in strokes:
            stroke_array = np.array(stroke)
            if len(stroke_array) < 2:
                continue

            x_points = stroke_array[:, 0]
            y_points = stroke_array[:, 1]
            t = np.arange(len(x_points))

            # 点が4つ未満の場合は補間の次数を調整
            k = min(3, len(t) - 1)
            if k < 1: continue

            # 3次スプライン補間 (または低次の補間)
            cs_x = CubicSpline(t, x_points, bc_type='natural')
            cs_y = CubicSpline(t, y_points, bc_type='natural')

            t_new = np.linspace(t.min(), t.max(), 100)
            x_new = cs_x(t_new)
            y_new = cs_y(t_new)

            # --- 結果を描画 ---
            if is_first_stroke:
                plt.plot(x_points, y_points, 'o', color='red', markersize=2, label='Original Points')
                plt.plot(x_new, y_new, '-', color='blue', label='Spline Curve')
                is_first_stroke = False
            else:
                plt.plot(x_points, y_points, 'o', color='red', markersize=2)
                plt.plot(x_new, y_new, '-', color='blue')

        # --- グラフ全体の見た目を設定 ---
        plt.title(f'Spline Interpolation for "{char_text}"')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.axis('equal')

        # --- ファイルに保存 ---
        base_filename = os.path.basename(json_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        output_path = os.path.join(output_dir, f'{filename_without_ext}.png')
        
        plt.savefig(output_path)
        plt.close() # メモリを解放するためにプロットを閉じる

        print(f"✔️ Successfully processed and saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {json_path}: {e}")

def process_and_resample(json_path):
    """
    JSONを読み込み、スプライン補間とリサンプリングを行い、
    グラフと処理後のデータを保存する関数
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        strokes = data.get('strokes')
        char_text = data.get('text')
        # Unknownを第二引数にすると処理が止まらなくていいね！by Gemini　くそが
                
        # 処理後の全ストロークのデータを格納するリスト
        all_resampled_strokes = []

        for i, stroke in enumerate(strokes):
            stroke_array = np.array(stroke)
            if len(stroke_array) < 2:
                continue

            x_points, y_points = stroke_array[:, 0], stroke_array[:, 1]
            time = stroke_array[:, 2]
            # touching = stroke_array[:, 3]

            t = np.arange(len(x_points))

            if len(t) < 2: continue

            cs_x = CubicSpline(t, x_points, bc_type='natural')
            cs_y = CubicSpline(t, y_points, bc_type='natural')
            cs_t = CubicSpline(t, time, bc_type='natural')

            # # --- ステップ1: データを滑らかに拡張 ---
            # t_expanded = np.linspace(t.min(), t.max(), 200)
            # x_expanded = cs_x(t_expanded)
            # y_expanded = cs_y(t_expanded)
            
            # --- ステップ2: 指定した点数でリサンプリング ---
            t_resampled = np.linspace(t.min(), t.max(), NUM_POINTS)
            x_resampled = cs_x(t_resampled)
            y_resampled = cs_y(t_resampled)
            time_resampled = cs_t(t_resampled)
            touching_resampled = np.ones_like(x_resampled) 

            
            # 処理後のデータを保存用にまとめる
            # 1. NUM_POINTS個の「描画点」データを作成 (タッチフラグは全て1)
            touching_ones = np.ones(NUM_POINTS)
            drawing_points = np.vstack([
                x_resampled, 
                y_resampled, 
                time_resampled, 
                touching_ones
            ]).T
            
            # 2. 最後の描画点と同じ座標・時刻を持つ「ペン上げ点」を1つ作成 (タッチフラグは0)
            pen_up_point = np.array([[
                x_resampled[-1], 
                y_resampled[-1], 
                time_resampled[-1], 
                0
            ]])
            
            # 3. 描画点とペン上げ点を結合して、長さが NUM_POINTS + 1 のデータにする
            resampled_stroke = np.vstack([drawing_points, pen_up_point])
            all_resampled_strokes.append(resampled_stroke)

        
        strokes_as_list = [stroke.tolist() for stroke in all_resampled_strokes]
        base_filename = os.path.splitext(os.path.basename(json_path))[0]

        # --- 新しいJSONデータを作成 ---
        output_data = {
            "text": char_text,
            "strokes": strokes_as_list
        }

        # --- リサンプリングしたデータをファイルとして保存 ---
        output_data_path = os.path.join(SAVING_DATA_DIR, f'{base_filename}.json')
        with open(output_data_path, 'w', encoding='utf-8') as f:
            # indent=2 を指定すると、人間が読みやすいように整形されたJSONファイルが出力される
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✔️ Processed: {base_filename}.json -> Saved json and data.")

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(json_path)}: {e}")


# --- スクリプトの実行 ---
if __name__ == '__main__':
    # 入力フォルダ内のすべての.jsonファイルを取得
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))

    if not json_files:
        print(f"No JSON files found in '{INPUT_DIR}' directory.")
    else:
        if True:
            for file_path in json_files:
                # 順々に画像として保存  
                process_and_save_plot(file_path, OUTPUT_DIR)
        for file_path in json_files:
            process_and_resample(file_path)
        print("\nAll files processed.")
