import os
import glob
import json

def find_minmax_xy_in_strokes(strokes):
    """strokes配列からx座標とy座標の最小値・最大値を返す"""
    min_x = max_x = None
    min_y = max_y = None

    for stroke in strokes:
        for point in stroke:  # point = [x, y, time, isTouching]
            if len(point) >= 2:
                x, y = point[0], point[1]

                if isinstance(x, (int, float)):
                    min_x = x if min_x is None else min(min_x, x)
                    max_x = x if max_x is None else max(max_x, x)

                if isinstance(y, (int, float)):
                    min_y = y if min_y is None else min(min_y, y)
                    max_y = y if max_y is None else max(max_y, y)

    return min_x, max_x, min_y, max_y

def find_minmax_in_folder(folder_path):
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    overall_min_x = overall_max_x = None
    overall_min_y = overall_max_y = None

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "strokes" in data:
                min_x, max_x, min_y, max_y = find_minmax_xy_in_strokes(data["strokes"])
                
                if min_x is not None:
                    overall_min_x = min_x if overall_min_x is None else min(overall_min_x, min_x)
                if max_x is not None:
                    overall_max_x = max_x if overall_max_x is None else max(overall_max_x, max_x)

                if min_y is not None:
                    overall_min_y = min_y if overall_min_y is None else min(overall_min_y, min_y)
                if max_y is not None:
                    overall_max_y = max_y if overall_max_y is None else max(overall_max_y, max_y)

        except Exception as e:
            print(f"⚠ エラー {file_path}: {e}")

    return overall_min_x, overall_max_x, overall_min_y, overall_max_y

if __name__ == "__main__":
    folder = "sentences-4dim/oneletters/mydxdy/addtextids"  # ←ここにフォルダパスを入れる
    min_x, max_x, min_y, max_y = find_minmax_in_folder(folder)
    if None not in (min_x, max_x, min_y, max_y):
        print(f"フォルダ内のstrokesの範囲:")
        print(f"x: {min_x} ～ {max_x}")
        print(f"y: {min_y} ～ {max_y}")
    else:
        print("strokes内に数値が見つかりませんでした。")
