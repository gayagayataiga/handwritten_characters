import json
import matplotlib.pyplot as plt

# JSONを読み込む
with open("hiragana/あ.json", "r", encoding="utf-8") as f:
    data = json.load(f)

strokes = data["strokes"]

# 色を決める（1画目: 赤, 2画目: 青）
colors = ['red', 'blue', 'green', 'orange', 'purple']  # 必要に応じて増やせます

plt.figure(figsize=(4, 4))

for i, stroke in enumerate(strokes):
    xs = [p["x"] for p in stroke]
    ys = [p["y"] for p in stroke]
    plt.plot(xs, ys, color=colors[i % len(colors)],
             linewidth=2, label=f"stroke {i+1}")

plt.gca().invert_yaxis()  # Canvas座標系に合わせる
plt.axis("equal")
plt.title(data["char"])
plt.legend()
plt.show()
