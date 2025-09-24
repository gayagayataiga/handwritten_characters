import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence


def load_stroke(path, vocab):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data["text"]
    text_ids = [vocab[ch] for ch in text]

    seq = []
    for stroke in data["strokes"]:
        for (dx, dy, t, p) in stroke:
            # すでに dx, dy に正規化済み前提
            pen = [1, 0] if p == 1 else [0, 1]  # pen down / up
            seq.append([dx, dy] + pen)

    return dict(
        text=text,
        text_ids=text_ids,
        strokes=torch.tensor(seq, dtype=torch.float32)
    )


def save_padded_individual(in_dir, out_dir, vocab):
    os.makedirs(out_dir, exist_ok=True)

    # 入力フォルダから JSON を集める
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".json")]
    samples = [load_stroke(f, vocab) for f in files]

    strokes = [s["strokes"] for s in samples]
    lengths = [len(s) for s in strokes]
    max_len = max(lengths)

    # パディング
    padded = pad_sequence(strokes, batch_first=True, padding_value=0.0)  # [N, T_max, 4]
    masks = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]  # [N, T_max]

    # 1サンプルごとに保存（平らに保存）
    for i, (s, f) in enumerate(zip(samples, files)):
        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{base}_padded.json")

        data_out = dict(
            text=s["text"],
            text_ids=s["text_ids"],
            strokes=padded[i].tolist(),
            stroke_mask=masks[i].int().tolist()
        )

        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(data_out, fout, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(samples)} padded files into {out_dir}/")


# ===== 使い方サンプル =====
if __name__ == "__main__":
    vocab = {"あ": 1, "い": 2, "う": 3, "え": 4, "お": 5,
             "か": 6, "き": 7, "く": 8, "け": 9, "こ": 10,
             "さ": 11, "し": 12, "す": 13, "せ": 14, "そ": 15,
             "た": 16, "ち": 17, "つ": 18, "て": 19, "と": 20,
             "な": 21, "に": 22, "ぬ": 23, "ね": 24, "の": 25,
             "は": 26, "ひ": 27, "ふ": 28, "へ": 29, "ほ": 30,
             "ま": 31, "み": 32, "む": 33, "め": 34, "も": 35,
             "や": 36, "ゆ": 37, "よ": 38,
             "ら": 31, "り": 32, "る": 33, "れ": 34, "ろ": 35,
             "わ": 36, "を": 37, "ん": 38, "。": 39, "、": 40,}  # 必要に応じて拡張

    in_dir = "sentences-4dim/oneletters/normalized_xy"        # 読み込み元フォルダ
    out_dir = "sentences-4dim/oneletters/norm_padding"    # 保存先フォルダ

    save_padded_individual(in_dir, out_dir, vocab)
