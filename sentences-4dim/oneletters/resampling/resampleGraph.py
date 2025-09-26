import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- 定数設定 ---
NUM_POINTS_PER_STROKE = 64

# --- これまでの前処理関数（変更なし） ---


def convert_dict_strokes_to_lists(strokes_in_dict_format):
    strokes_in_list_format = []
    for single_stroke_dicts in strokes_in_dict_format:
        single_stroke_lists = [[p['x'], p['y'], p['time'],
                                p['isTouching']] for p in single_stroke_dicts]
        strokes_in_list_format.append(single_stroke_lists)
    return strokes_in_list_format


def remove_consecutive_duplicates(points):
    if not points:
        return []
    cleaned_points = [points[0]]
    for i in range(1, len(points)):
        if points[i] != cleaned_points[-1]:
            cleaned_points.append(points[i])
    return cleaned_points


def resample_stroke(stroke, num_points):
    if len(stroke) < 2:
        return []
    points = np.array(stroke)
    coords, times = points[:, :2], points[:, 2]
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    if cumulative_distances[-1] == 0:
        resampled = [stroke[0]] * num_points
        resampled[-1][3] = 0.0
        return resampled
    target_distances = np.linspace(0, cumulative_distances[-1], num_points)
    interp_fn = {
        'x': interp1d(cumulative_distances, coords[:, 0], kind='linear', fill_value="extrapolate"),
        'y': interp1d(cumulative_distances, coords[:, 1], kind='linear', fill_value="extrapolate"),
        't': interp1d(cumulative_distances, times, kind='linear', fill_value="extrapolate")
    }
    new_coords = {k: v(target_distances) for k, v in interp_fn.items()}
    resampled_stroke = [[new_coords['x'][i], new_coords['y']
                         [i], new_coords['t'][i], 1.0] for i in range(num_points)]
    if resampled_stroke:
        resampled_stroke[-1][3] = 0.0
    return resampled_stroke


def normalize_character(strokes_list):
    if not strokes_list or not strokes_list[0]:
        return strokes_list
    first_point = strokes_list[0][0]
    x_offset, y_offset = first_point[0], first_point[1]
    translated_strokes, max_abs_val = [], 0.0
    for stroke in strokes_list:
        new_stroke = []
        for point in stroke:
            x, y, t, is_down = point
            new_x, new_y = x - x_offset, y - y_offset
            new_stroke.append([new_x, new_y, t, is_down])
            max_abs_val = max(max_abs_val, abs(new_x), abs(new_y))
        translated_strokes.append(new_stroke)
    if max_abs_val == 0:
        return translated_strokes
    normalized_strokes = []
    for stroke in translated_strokes:
        new_stroke = []
        for point in stroke:
            x, y, t, is_down = point
            new_stroke.append([x / max_abs_val, y / max_abs_val, t, is_down])
        normalized_strokes.append(new_stroke)
    return normalized_strokes


def convert_to_offsets(strokes_list):
    if not strokes_list:
        return []
    offset_strokes = []
    for stroke in strokes_list:
        if not stroke:
            continue
        new_stroke = [stroke[0]]
        for i in range(1, len(stroke)):
            prev_point, curr_point = stroke[i-1], stroke[i]
            dx, dy, dt = curr_point[0] - prev_point[0], curr_point[1] - \
                prev_point[1], curr_point[2] - prev_point[2]
            new_stroke.append([dx, dy, dt, curr_point[3]])
        offset_strokes.append(new_stroke)
    return offset_strokes

# --- (新規追加) 可視化のための関数 ---


def reconstruct_from_offsets(offset_strokes):
    """差分座標から絶対座標を復元してプロットできるようにする"""
    reconstructed_strokes = []
    for stroke in offset_strokes:
        if not stroke:
            continue
        # 最初の点は絶対座標なのでそのまま
        abs_x, abs_y, abs_t = stroke[0][0], stroke[0][1], stroke[0][2]
        new_stroke = [stroke[0]]
        # 2点目以降は差分を足していく
        for i in range(1, len(stroke)):
            dx, dy, dt = stroke[i][0], stroke[i][1], stroke[i][2]
            abs_x += dx
            abs_y += dy
            abs_t += dt
            new_stroke.append([abs_x, abs_y, abs_t, stroke[i][3]])
        reconstructed_strokes.append(new_stroke)
    return reconstructed_strokes


def plot_strokes(ax, strokes, title):
    """指定されたAxesオブジェクトにストロークを描画する"""
    ax.set_title(title)
    if not strokes:
        return
    for stroke in strokes:
        points = np.array(stroke)
        # x座標とy座標を抽出
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        ax.plot(x_coords, y_coords, marker='.')
    ax.set_aspect('equal', adjustable='box')
    # Y軸の向きを反転させて、描画座標系に合わせる
    ax.invert_yaxis()
    ax.grid(True)


# --- メインの実行部分 ---
if __name__ == '__main__':
    # ★ 可視化したいJSONファイルのパスを指定
    target_file = 'sentences-4dim/oneletters/changed_name/あ_20250924141205.json'

    if not os.path.exists(target_file):
        print(f"エラー: ファイルが見つかりません: {target_file}")
    else:
        # 1. ファイル読み込み
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 5段階のデータを格納するリスト
        pipeline_data = []

        # Step 0: 生データ
        raw_strokes_dict = data['strokes']
        raw_strokes_list = convert_dict_strokes_to_lists(raw_strokes_dict)
        pipeline_data.append(('Raw Data', raw_strokes_list))

        # Step 1: 重複除去
        cleaned_strokes = []
        for s in raw_strokes_list:
            c = remove_consecutive_duplicates(s)
            if c:
                cleaned_strokes.append(c)
        pipeline_data.append(('1. Duplicates Removed', cleaned_strokes))

        # Step 2: リサンプリング
        resampled_strokes = []
        for s in cleaned_strokes:
            r = resample_stroke(s, num_points=NUM_POINTS_PER_STROKE)
            if r:
                resampled_strokes.append(r)
        pipeline_data.append(('2. Resampled', resampled_strokes))

        # Step 3: 正規化
        normalized_strokes = normalize_character(resampled_strokes)
        pipeline_data.append(('3. Normalized', normalized_strokes))

        # Step 4: 差分座標へ変換
        offset_strokes = convert_to_offsets(normalized_strokes)
        # 可視化のために絶対座標に戻す
        reconstructed_for_plot = reconstruct_from_offsets(offset_strokes)
        pipeline_data.append(
            ('4. Offsets (Reconstructed)', reconstructed_for_plot))

        # グラフの描画
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(
            f'Preprocessing Pipeline Visualization: {os.path.basename(target_file)}', fontsize=16)

        for i, (title, strokes) in enumerate(pipeline_data):
            plot_strokes(axes[i], strokes, title)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
