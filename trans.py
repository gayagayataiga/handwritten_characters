import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
import datetime
import numpy as np

# ------------------------
# 1. データセット作成
# ------------------------
class HandwritingDataset(Dataset):
    def __init__(self, data_dir="hiragana"):
        self.samples = []
        self.chars = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.char2idx = {c:i for i,c in enumerate(self.chars)}
        
        for char in self.chars:
            char_dir = os.path.join(data_dir, char)
            files = glob.glob(os.path.join(char_dir, "*.json"))
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                strokes = data["strokes"]
                seq = []
                for stroke in strokes:
                    for i, p in enumerate(stroke):
                        pen_state = 1 if i < len(stroke)-1 else 0
                        seq.append([p["x"]/300, p["y"]/300, pen_state])
                self.samples.append((char, torch.tensor(seq, dtype=torch.float32)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        char, seq = self.samples[idx]
        char_idx = self.char2idx[char]
        return torch.tensor(char_idx), seq


# ------------------------
# 2. Transformerモデル
# ------------------------
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size=32, nhead=2, num_layers=2, hidden_dim=64):
        super().__init__()
        max_seq_len = 600  # 500くらいにしておくと大抵の文字は入る
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, emb_size))

        self.char_emb = nn.Embedding(vocab_size, emb_size)
        self.input_fc = nn.Linear(3, emb_size)  # (x,y,pen_state) -> emb
        #self.pos_emb = nn.Parameter(torch.randn(100, emb_size))  # max seq len 100
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(emb_size, 3)  # x,y,pen_state
    
    def forward(self, char_idx, seq):
        # seq: (seq_len, 3)
        char_emb = self.char_emb(char_idx).unsqueeze(0)  # (1, emb)
        seq_emb = self.input_fc(seq) + self.pos_emb[:seq.size(0)]
        x = seq_emb.unsqueeze(1)  # (seq_len, batch=1, emb)
        x = self.transformer(x)
        out = self.output_fc(x.squeeze(1))
        return out

# ------------------------
# 3. 学習ループ
# ------------------------
def train_model(data_dir="hiragana", epochs=10):
    dataset = HandwritingDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = HandwritingTransformer(vocab_size=len(dataset.chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for char_idx, seq in loader:
            optimizer.zero_grad()
            # 教師データとして次の座標を予測
            input_seq = seq[:, :-1, :]
            target_seq = seq[:, 1:, :]
            output = model(char_idx, input_seq[0])
            loss = loss_fn(output, target_seq[0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model, dataset

# ------------------------
# 4. サンプル生成
# ------------------------
def generate_sample(model, dataset, char, max_len=50):
    model.eval()
    char_idx = torch.tensor(dataset.char2idx[char])
    seq = torch.zeros((1,3))  # 初期座標は (0,0,1)
    generated = []
    for _ in range(max_len):
        with torch.no_grad():
            out = model(char_idx, seq)
        next_pt = out[-1]
        generated.append(next_pt.numpy())
        seq = torch.cat([seq, next_pt.unsqueeze(0)], dim=0)
        if next_pt[2] < 0.5:  # ペンアップで終了
            break
    return generated

# ------------------------
# 実行例
# ------------------------
if __name__ == "__main__":
    import os
import json
import datetime
import numpy as np  # ← 追加

if __name__ == "__main__":
    # モデル学習
    model, dataset = train_model(epochs=5)

    # サンプル生成
    sample = generate_sample(model, dataset, char="あ")

    # NumPy 配列をリストに変換する関数
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_ndarray_to_list(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
        else:
            return obj

    sample_list = convert_ndarray_to_list(sample)

    # 保存先フォルダ
    SAVE_DIR = "generated_samples"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ファイル名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    char = "あ"
    filename = f"{char}_{timestamp}.json"
    filepath = os.path.join(SAVE_DIR, filename)

    # JSONとして保存
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, ensure_ascii=False, indent=2)

    print(f"Sample saved to {filepath}")