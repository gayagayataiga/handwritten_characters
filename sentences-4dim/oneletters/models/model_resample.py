import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm

# ======================
# Dataset
# ======================
class StrokeDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        with open(self.files[idx], "r", encoding="utf-8") as f:
            data = json.load(f)

        text_ids = torch.tensor(data["text_ids"], dtype=torch.long)        # [L]
        strokes = torch.tensor(data["strokes"], dtype=torch.float32)       # [T, 4]
        return dict(
            text=data["text"],
            text_ids=text_ids,
            strokes=strokes,
        )
# ======================
# Custom Collate Function
# ======================
def custom_collate_fn(batch):
    # batchは、Datasetの__getitem__が返す辞書のリストです
    texts = [item['text'] for item in batch]
    text_ids_list = [item['text_ids'] for item in batch]
    strokes_list = [item['strokes'] for item in batch]

    # text_idsとstrokesを、バッチ内で最も長いデータに合わせてパディングします
    padded_text_ids = pad_sequence(text_ids_list, batch_first=True, padding_value=0)
    padded_strokes = pad_sequence(strokes_list, batch_first=True, padding_value=0.0)

    # stroke_maskを動的に生成します
    # パディングされた部分（全要素が0）はFalse、元のデータ部分はTrueになるマスクを作成
    stroke_mask = (padded_strokes.sum(dim=-1) != 0)

    return {
        'text': texts,
        'text_ids': padded_text_ids,
        'strokes': padded_strokes,
        'stroke_mask': stroke_mask  # ← マスクをここで生成してバッチに含める
    }

# ======================
# Positional Encoding
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ======================
# MDN Loss
# ======================
def mdn_loss_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_x, target_y, eps=1e-8):
    pi = F.log_softmax(pi, dim=-1)  # [B, T, K]
    x = target_x.unsqueeze(-1)
    y = target_y.unsqueeze(-1)
    z_x = (x - mu_x) / (sigma_x + eps)
    z_y = (y - mu_y) / (sigma_y + eps)
    one_minus_rho2 = 1 - rho ** 2 + eps
    norm = (z_x**2 + z_y**2 - 2 * rho * z_x * z_y) / one_minus_rho2
    denom = 2 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho2)
    log_component = -0.5 * norm - torch.log(denom + eps)  # [B, T, K]
    log_mix = torch.logsumexp(pi + log_component, dim=-1)  # [B, T]
    return -log_mix

def compute_loss(model_out, target_dx, target_dy, target_pen, stroke_mask):
    pi = model_out["pi"]
    mu_x = model_out["mu_x"]
    mu_y = model_out["mu_y"]
    sigma_x = model_out["sigma_x"]
    sigma_y = model_out["sigma_y"]
    rho = model_out["rho"]
    pen_logits = model_out["pen"]

    # mdn_nll = mdn_loss_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_dx, target_dy)

    log_mix = mdn_loss_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho, target_dx, target_dy)
    mdn_nll = (log_mix * stroke_mask.float()).sum() / (stroke_mask.sum() + 1e-8)


    pen_ce = F.cross_entropy(
        pen_logits.view(-1, pen_logits.size(-1)),
        target_pen.view(-1),
        reduction="none"
    )
    mask = stroke_mask.view(-1).float()
    pen_loss = (pen_ce * mask).sum() / (mask.sum() + 1e-8)

    return mdn_nll + pen_loss, dict(mdn=mdn_nll.item(), pen=pen_loss.item())

# ======================
# Transformer Model
# ======================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1,
                 mdn_components=20, pen_states=2, stroke_dim=4): # stroke_dimを使うように変更
        super().__init__()
        self.d_model = d_model
        self.text_tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.stroke_in = nn.Linear(stroke_dim, d_model) # stroke_dim を使う
        self.stroke_pos = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        K = mdn_components
        self.mdn_pi = nn.Linear(d_model, K)
        self.mdn_mu_x = nn.Linear(d_model, K)
        self.mdn_mu_y = nn.Linear(d_model, K)
        self.mdn_sigma_x = nn.Linear(d_model, K)
        self.mdn_sigma_y = nn.Linear(d_model, K)
        self.mdn_rho = nn.Linear(d_model, K)
        self.pen_logits = nn.Linear(d_model, pen_states)

    # 1. エンコーダー部分を独立させる
    def encode(self, text_ids, text_mask):
        te = self.text_tok(text_ids) * math.sqrt(self.d_model)
        te = self.text_pos(te)
        memory = self.encoder(te, src_key_padding_mask=~text_mask)
        return memory

    # 2. デコーダー部分を独立させる
    def decode(self, strokes, stroke_mask, memory, memory_mask):
        B, T, _ = strokes.shape
        di = self.stroke_in(strokes)
        di = self.stroke_pos(di)
        
        # デコーダー用の未来を見ないためのマスク
        tgt_mask = torch.triu(torch.ones((T, T), device=strokes.device), diagonal=1).bool()
        
        dec = self.decoder(tgt=di, memory=memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=~memory_mask,
                           tgt_key_padding_mask=~stroke_mask)

        # MDNパラメータの計算
        pi = self.mdn_pi(dec)
        mu_x = self.mdn_mu_x(dec)
        mu_y = self.mdn_mu_y(dec)
        sigma_x = torch.exp(self.mdn_sigma_x(dec)).clamp(min=1e-4)
        sigma_y = torch.exp(self.mdn_sigma_y(dec)).clamp(min=1e-4)

        # rho (相関係数) は-1と1に近すぎると不安定になるため、少しだけ内側に制限する
        rho = torch.tanh(self.mdn_rho(dec)).clamp(-0.99999, 0.99999)
        pen = self.pen_logits(dec)
        return dict(pi=pi, mu_x=mu_x, mu_y=mu_y,
                    sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
                    pen=pen)

    # 3. forwardはこれらを呼び出すだけにする（今回は使わない）
    def forward(self, text_ids, text_mask, strokes, stroke_mask):
        memory = self.encode(text_ids, text_mask)
        return self.decode(strokes, stroke_mask, memory, text_mask)

# ======================
# Training Loop
# ======================
def train_loop(data_dir, vocab, epochs=1000, batch_size=32, lr=1e-4, device="cuda",checkpoint_path=None):
    dataset = StrokeDataset(data_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,num_workers=4)

    model = HandwritingTransformer(
        vocab_size=len(vocab)+1,
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, mdn_components=20, pen_states=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"チェックポイントを読み込みます: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        # ファイル名からエポック番号を復元して、続きから開始する（おまけ機能）
        try:
            # "handwriting_epoch100.pt" -> "100" を抽出
            epoch_num_str = os.path.basename(checkpoint_path).split('epoch')[1].split('.pt')[0]
            start_epoch = int(epoch_num_str) + 1
            print(f"エポック {start_epoch} から学習を再開します。")
        except:
            print("エポック番号の復元に失敗しました。指定されたエポックから開始します。")
            
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        # train_loop内のforループ
        for batch in pbar:
            text_ids = batch["text_ids"].to(device)
            strokes = batch["strokes"].to(device)
            stroke_mask = batch["stroke_mask"].to(device)
            
            text_mask = text_ids != 0

            # 1. エンコード
            memory = model.encode(text_ids, text_mask)

            # 2. コンテキストをリピート
            B, N, T, F = strokes.shape
            expanded_memory = memory.repeat(1, N, 1) 
            expanded_text_mask = text_mask.repeat(1, N)

            # 3. テンソルを変形
            strokes = strokes.view(B * N, T, F)
            stroke_mask = stroke_mask.view(B * N, T)
            memory = expanded_memory.view(B * N, 1, -1)
            # print(strokes, stroke_mask, memory)
            
            memory_mask = expanded_text_mask.view(B * N, 1) 
            # 4. デコード
            out = model.decode(strokes, stroke_mask, memory, memory_mask)
            
            # print(out)

            # ターゲット（正解データ）を作成
            target_dx = strokes[:,:,0]
            target_dy = strokes[:,:,1]
            target_pen = torch.argmax(strokes[:,:,2:], dim=-1)

            loss, loss_dict = compute_loss(out, target_dx, target_dy, target_pen, stroke_mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), mdn=loss_dict["mdn"], pen=loss_dict["pen"])
        print(f"Epoch {epoch} | Avg Loss: {total_loss/len(loader):.4f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/myresampling/handwriting_epoch{epoch}.pt")

    print("✅ Training finished!")

# ======================
# 実行
# ======================
if __name__ == "__main__":
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

    train_loop("sentences-4dim/oneletters/mydxdy/addtextids", vocab, epochs=400, batch_size=512*2, lr=1e-5,checkpoint_path="checkpoints/myresampling/handwriting_epoch250.pt")
    