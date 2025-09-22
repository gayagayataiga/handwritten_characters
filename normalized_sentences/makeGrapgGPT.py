import json
import matplotlib.pyplot as plt

def plot_strokes(json_path, show=True, save_path=None):
    """
    JSONファイル内のstrokesをmatplotlibで表示
    - show=True なら画面表示
    - save_path を指定すれば画像として保存
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "strokes" not in data:
        print("⚠ strokes データが見つかりません")
        return

    strokes = data["strokes"]

    plt.figure(figsize=(4, 4))
    for stroke in strokes:
        xs = [pt["x"] for pt in stroke]
        ys = [pt["y"] for pt in stroke]
        plt.plot(xs, ys, linewidth=2, color="black")  # ストロークごとに線を引く

    plt.gca().invert_yaxis()   # 手書き座標は下に行くほどyが大きいことが多いので反転
    plt.axis("equal")          # アスペクト比を1:1にする
    plt.axis("off")            # 枠線を非表示

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"画像を {save_path} に保存しました")

    if show:
        plt.show()

# ===== 使用例 =====
# 正規化済みのjsonを表示
plot_strokes("normalized_sentences\AIと一緒に学習しています。_20250920145922.json")

# 元のjsonを表示
# plot_strokes("sentences/sample.json")
