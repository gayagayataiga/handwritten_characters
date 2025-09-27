import os
import json
import numpy as np
import random
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- 定数設定 ---
NUM_POINTS_PER_STROKE = 64

vocab = {"あ": 1, "い": 2, "う": 3, "え": 4, "お": 5,
             "か": 6, "き": 7, "く": 8, "け": 9, "こ": 10,
             "さ": 11, "し": 12, "す": 13, "せ": 14, "そ": 15,
             "た": 16, "ち": 17, "つ": 18, "て": 19, "と": 20,
             "な": 21, "に": 22, "ぬ": 23, "ね": 24, "の": 25,
             "は": 26, "ひ": 27, "ふ": 28, "へ": 29, "ほ": 30,
             "ま": 31, "み": 32, "む": 33, "め": 34, "も": 35,
             "や": 36, "ゆ": 37, "よ": 38,
             "ら": 39, "り": 40, "る": 41, "れ": 42, "ろ": 43,
             "わ": 44, "を": 45, "ん": 46, "。": 47, "、": 48,}  # 必要に応じて拡張



def convert_dict_strokes_to_lists(strokes_in_dict_format):
    """
    (新規追加) 辞書形式のストロークデータをリスト形式に変換する。
    入力: [ [{"x":..,"y":..}, ...], [{"x":..,"y":..}, ...] ]
    出力: [ [[x,y,t,i], ...], [[x,y,t,i], ...] ]
    """
    strokes_in_list_format = []
    for single_stroke_dicts in strokes_in_dict_format:
        single_stroke_lists = [
            [p['x'], p['y'], p['time'], p['isTouching']]
            for p in single_stroke_dicts
        ]
        strokes_in_list_format.append(single_stroke_lists)
    return strokes_in_list_format


def remove_consecutive_duplicates(points):
    """点のリストから、連続する重複点を除去します。"""
    if not points:
        return []
    cleaned_points = [points[0]]
    for i in range(1, len(points)):
        if points[i] != cleaned_points[-1]:
            cleaned_points.append(points[i])
    return cleaned_points


def resample_stroke(stroke, num_points):
    """
    一つのストロークを等間隔な点にリサンプリングします。
    座標(x, y)と時間(t)を補間します。
    """
    if len(stroke) < 2:
        return []

    # 元データが4次元であることを確認
    if len(stroke[0]) != 4:
        # 予期しない形式の場合は元のストロークを返すかエラー処理
        print("警告: ストロークのデータ形式が4次元ではありません。")
        return stroke

    points = np.array(stroke)

    # 1. 座標(x, y)と時間(t)をそれぞれ抽出
    coords = points[:, :2]  # x, y座標
    times = points[:, 2]   # 3次元目の時間

    # 2. 座標(x,y)に基づいて累積距離を計算（時間は距離計算に含めない）
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    if cumulative_distances[-1] == 0:
        # 最初の点を指定回数分繰り返して返す
        return [stroke[0]] * num_points

    # 3. 軌跡の全長に沿って、リサンプリングしたい位置（等間隔な距離）を生成
    target_distances = np.linspace(0, cumulative_distances[-1], num_points)

    # 4. x, y, time それぞれの補間関数を準備
    interp_x = interp1d(cumulative_distances, coords[:, 0], kind='linear')
    interp_y = interp1d(cumulative_distances, coords[:, 1], kind='linear')
    interp_time = interp1d(cumulative_distances, times, kind='linear')

    # 5. 新しい座標と時間を計算
    new_x = interp_x(target_distances)
    new_y = interp_y(target_distances)
    new_times = interp_time(target_distances)

    # 6. 新しい点のリストを再構成する
    #    [new_x, new_y, new_time, 1.0 (ペンダウン)] の形式にする
    resampled_stroke = [
        [x, y, t, 1.0] for x, y, t in zip(new_x, new_y, new_times)
    ]
    resampled_stroke.append(
        [new_x[-1], new_y[-1], new_times[-1], 0.0])  # 最後はペンアップ

    return resampled_stroke


def convert_dict_strokes_to_lists(strokes_in_dict_format):
    """辞書形式のストロークデータをリスト形式に変換する。"""
    strokes_in_list_format = []
    for single_stroke_dicts in strokes_in_dict_format:
        single_stroke_lists = [
            [p['x'], p['y'], p['time'], p['isTouching']]
            for p in single_stroke_dicts
        ]
        strokes_in_list_format.append(single_stroke_lists)
    return strokes_in_list_format


def remove_consecutive_duplicates(points):
    """点のリストから、連続する重複点を除去します。"""
    if not points:
        return []
    cleaned_points = [points[0]]
    for i in range(1, len(points)):
        if points[i] != cleaned_points[-1]:
            cleaned_points.append(points[i])
    return cleaned_points


def resample_stroke(stroke, num_points):
    """一つのストロークを等間隔な点にリサンプリングします。"""
    if len(stroke) < 2:
        return []

    points = np.array(stroke)
    coords = points[:, :2]
    times = points[:, 2]

    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    if cumulative_distances[-1] == 0:
        resampled = [stroke[0]] * num_points
        resampled[-1][3] = 0.0
        return resampled

    target_distances = np.linspace(0, cumulative_distances[-1], num_points)

    interp_x = interp1d(cumulative_distances,
                        coords[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(cumulative_distances,
                        coords[:, 1], kind='linear', fill_value="extrapolate")
    interp_time = interp1d(cumulative_distances, times,
                           kind='linear', fill_value="extrapolate")

    new_x = interp_x(target_distances)
    new_y = interp_y(target_distances)
    new_times = interp_time(target_distances)

    resampled_stroke = [[x, y, t, 1.0]
                        for x, y, t in zip(new_x, new_y, new_times)]
    if resampled_stroke:
        resampled_stroke[-1][3] = 0.0
    return resampled_stroke


def normalize_character(strokes_list):
    """
    文字全体の位置と大きさを正規化する。
    """
    # データがない場合はそのまま返す
    if not strokes_list or not strokes_list[0]:
        return strokes_list

    # --- 1. 平行移動（位置合わせ） ---
    # 文字全体の最初の点を取得し、その座標をオフセットとする
    first_point = strokes_list[0][0]
    x_offset, y_offset = first_point[0], first_point[1]

    translated_strokes = []
    # このループで平行移動と、次のスケーリングで使うための最大値探索を同時に行う
    max_abs_val = 0.0
    for stroke in strokes_list:
        new_stroke = []
        for point in stroke:
            x, y, t, is_down = point
            # オフセットを引いて原点に移動
            new_x = x - x_offset
            new_y = y - y_offset
            new_stroke.append([new_x, new_y, t, is_down])
            # 移動後の座標の絶対値の最大値を記録
            max_abs_val = max(max_abs_val, abs(new_x), abs(new_y))
        translated_strokes.append(new_stroke)

    # --- 2. 拡大・縮小（サイズ合わせ） ---
    # 最大絶対値が0の場合（点が一つしかないなど）は、ゼロ除算を避ける
    if max_abs_val == 0:
        return translated_strokes

    normalized_strokes = []
    for stroke in translated_strokes:
        new_stroke = []
        for point in stroke:
            x, y, t, is_down = point
            # 全ての座標を最大絶対値で割り、[-1, 1]の範囲に収める
            new_stroke.append([x / max_abs_val, y / max_abs_val, t, is_down])
        normalized_strokes.append(new_stroke)

    return normalized_strokes


def convert_to_offsets(strokes_list):
    """
    (新規追加) 絶対座標を差分座標 (offset) に変換する。
    """
    if not strokes_list:
        return []

    offset_strokes = []
    for stroke in strokes_list:
        if not stroke:
            continue

        # 各ストロークの最初の点は、(0,0,0)からの移動量なので、そのまま絶対座標を使用
        new_stroke = [stroke[0]]

        # 2点目以降は、前の点からの差分を計算
        for i in range(1, len(stroke)):
            prev_point = stroke[i-1]
            curr_point = stroke[i]

            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            dt = curr_point[2] - prev_point[2]
            is_down = curr_point[3]  # ペン状態はそのままコピー

            new_stroke.append([dx, dy, dt, is_down])

        offset_strokes.append(new_stroke)

    return offset_strokes


def reconstruct_from_offsets(offset_strokes):
    """(新規追加) 差分座標から絶対座標を復元してプロットできるようにする"""
    reconstructed_strokes = []
    for stroke in offset_strokes:
        if not stroke:
            continue
        abs_x, abs_y, abs_t = stroke[0][0], stroke[0][1], stroke[0][2]
        new_stroke = [stroke[0]]
        for i in range(1, len(stroke)):
            dx, dy, dt = stroke[i][0], stroke[i][1], stroke[i][2]
            abs_x += dx
            abs_y += dy
            abs_t += dt
            new_stroke.append([abs_x, abs_y, abs_t, stroke[i][3]])
        reconstructed_strokes.append(new_stroke)
    return reconstructed_strokes


def plot_strokes(ax, strokes, title):
    """(新規追加) 指定されたAxesオブジェクトにストロークを描画する"""
    ax.set_title(title)
    if not strokes:
        return
    for stroke in strokes:
        points = np.array(stroke)
        x_coords, y_coords = points[:, 0], points[:, 1]
        ax.plot(x_coords, y_coords, marker='.')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.grid(True)


def augment_strokes(strokes_list):
    """ストロークデータにデータ拡張を適用する。"""
    if not strokes_list:
        return strokes_list
    scale_factor = random.uniform(0.9, 1.1)
    angle_deg = random.uniform(-10, 10)
    shear_factor = random.uniform(-0.1, 0.1)
    angle_rad = np.deg2rad(angle_deg)
    S = np.array([[scale_factor, 0], [0, scale_factor]])
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                 [np.sin(angle_rad), np.cos(angle_rad)]])
    H = np.array([[1, shear_factor], [0, 1]])
    transform_matrix = S @ R @ H
    augmented_strokes = []
    for stroke in strokes_list:
        points = np.array(stroke)
        coords = points[:, :2]
        transformed_coords = coords @ transform_matrix.T
        new_stroke = []
        for i, point in enumerate(stroke):
            new_x, new_y = transformed_coords[i]
            _, _, t, is_down = point
            new_stroke.append([new_x, new_y, t, is_down])
        augmented_strokes.append(new_stroke)
    return augmented_strokes
# ===============================================================
# process_files_in_folder とメイン実行部分は変更ありません
# ===============================================================


def process_files_in_folder(input_dir, output_dir):
    """
    フォルダ内のJSONファイルを前処理し、通常版と拡張版の2種類を同じフォルダに保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"フォルダを作成しました: {output_dir}")

    files = os.listdir(input_dir)
    processed_count = 0

    for filename in files:
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)

            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'strokes' in data and isinstance(data['strokes'], list):
                    # --- 共通の前処理 (正規化まで) ---
                    original_strokes = data['strokes']

                    cleaned_strokes = []
                    for single_stroke in original_strokes:
                        cleaned_stroke = remove_consecutive_duplicates(
                            single_stroke)
                        if cleaned_stroke:
                            cleaned_strokes.append(cleaned_stroke)

                    strokes_as_lists = convert_dict_strokes_to_lists(
                        cleaned_strokes)

                    resampled_strokes = []
                    for single_stroke in strokes_as_lists:
                        resampled = resample_stroke(
                            single_stroke, num_points=NUM_POINTS_PER_STROKE)
                        if resampled:
                            resampled_strokes.append(resampled)

                    normalized_strokes = normalize_character(resampled_strokes)

                    # --- ここから分岐して2つのファイルを保存 ---
                    base_name, ext = os.path.splitext(filename)

                    # 1. データ拡張なし版を保存
                    offset_processed = convert_to_offsets(normalized_strokes)
                    data_processed = data.copy()
                    data_processed['strokes'] = offset_processed

                    text_value = data.get("text", None)
                    text_ids_value = [vocab[text_value]] if text_value in vocab else []
                    data_processed['text_ids'] = text_ids_value
                    output_filename_processed = f"{base_name}_processed.json"
                    output_path_processed = os.path.join(
                        output_dir, output_filename_processed)

                    with open(output_path_processed, 'w', encoding='utf-8') as f:
                        json.dump(data_processed, f,
                                  ensure_ascii=False, indent=2)

                    # 2. データ拡張あり版を保存
                    augmented_strokes = augment_strokes(normalized_strokes)
                    offset_augmented = convert_to_offsets(augmented_strokes)
                    data_augmented = data.copy()
                    data_augmented['strokes'] = offset_augmented
                    data_augmented['text_ids'] = text_ids_value

                    output_filename_augmented = f"{base_name}_augmented.json"
                    output_path_augmented = os.path.join(
                        output_dir, output_filename_augmented)

                    with open(output_path_augmented, 'w', encoding='utf-8') as f:
                        json.dump(data_augmented, f,
                                  ensure_ascii=False, indent=2)

                    print(f"処理完了: {filename} -> 2ファイル保存")
                    processed_count += 1

            except Exception as e:
                print(f"エラー: {filename} の処理中に問題が発生しました - {e}")

    print(f"\nすべての処理が完了しました。{processed_count}個のファイルを処理しました。")


# --- メインの実行部分 ---
if __name__ == '__main__':
    # フォルダのパスを指定
    input_folder = 'sentences-4dim/oneletters/changed_name'
    output_folder = 'sentences-4dim/oneletters/resampling'

    process_files_in_folder(input_folder, output_folder)
