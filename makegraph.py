import json
import matplotlib.pyplot as plt
import os

# フォルダパス
folder = "sentences"

# 色を決める（ストロークごとにループ）
colors = ['red', 'red', 'red', 'orange', 'purple']

plt.figure(figsize=(6, 6))

# フォルダ内のすべてのjsonファイルを処理
for file in sorted(os.listdir(folder)):
    if file.endswith(".json"):
        filepath = os.path.join(folder, file)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        strokes = data["strokes"]
        char = data.get("char", "?")

        # ストロークごとに描画
        for i, stroke in enumerate(strokes):
            xs = [p["x"] for p in stroke]
            ys = [p["y"] for p in stroke]
            plt.plot(xs, ys,
                     color=colors[i % len(colors)],
                     linewidth=2,
                     alpha=0.3,                  # ← 透明度を追加
                     label=f"{file} stroke {i+1}")

plt.gca().invert_yaxis()  # Canvas座標系に合わせる
plt.axis("equal")
plt.title(f"{char} のすべてのデータ")
plt.show()
