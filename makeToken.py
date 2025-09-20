import os
import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. 文字辞書の作成
# ==============================
def build_vocab(json_dir, min_freq=1):
    """
    json_dir: 正規化済みJSONが入ったフォルダ
    min_freq: 出現頻度が低い文字を <unk> にまとめる閾値
    """
    from collections import Counter
    counter = Counter()

    # 全JSONを読み込んで文字を数える
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            counter.update(list(data["text"]))

    # 辞書の初期トークン
    char2id = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    id2char = {0:"<pad>", 1:"<unk>", 2:"<s>", 3:"</s>"}

    # 出現頻度が min_freq 以上の文字を登録
    for ch, freq in counter.items():
        if freq >= min_freq and ch not in char2id:
            idx = len(char2id)
            char2id[ch] = idx
            id2char[idx] = ch

    return char2id, id2char


# ==============================
# 2. Datasetクラス
# ==============================
class HandwritingDataset(Dataset):
    def __init__(self, json_dir, char2id, max_text_len=100, max_seq_len=1000):
        self.files = glob.glob(os.path.join(json_dir, "*.json"))
        self.char2id = char2id
        self.max_text_len = max_text_len
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- テキスト処理 ---
        text = list(data["text"])
        text_ids = [self.char2id.get(ch, self.char2id["<unk>"]) for ch in text]
        text_ids = [self.char2id["<s>"]] + text_ids + [self.char2id["</s>"]]

        # パディング / 切り詰め
        if len(text_ids) < self.max_text_len:
            text_ids += [self.char2id["<pad>"]] * (self.max_text_len - len(text_ids))
        else:
            text_ids = text_ids[:self.max_text_len]

        text_ids = torch.tensor(text_ids, dtype=torch.long)

        # --- ストローク処理 ---
        seq = data["sequence"]  # [[dx, dy, end], ...]
        if len(seq) < self.max_seq_len:
            seq += [[0,0,0]] * (self.max_seq_len - len(seq))  # パディング
        else:
            seq = seq[:self.max_seq_len]

        seq = torch.tensor(seq, dtype=torch.float32)

        return text_ids, seq


# ==============================
# 3. 動作確認
# ==============================
if __name__ == "__main__":
    json_dir = "processed_json"  # 正規化済みJSONが入ったフォルダ

    # 辞書を作る
    char2id, id2char = build_vocab(json_dir)
    print("語彙数:", len(char2id))

    # Datasetを作る
    dataset = HandwritingDataset(json_dir, char2id)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 1バッチ確認
    for text_ids, seq in dataloader:
        print("text_ids.shape:", text_ids.shape)  # (batch, max_text_len)
        print("seq.shape:", seq.shape)            # (batch, max_seq_len, 3)
        print("サンプル text_ids:", text_ids[0][:10])
        print("サンプル seq:", seq[0][:5])
        break
