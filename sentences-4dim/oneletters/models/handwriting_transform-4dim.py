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
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_padded.json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "r", encoding="utf-8") as f:
            data = json.load(f)

        text_ids = torch.tensor(data["text_ids"], dtype=torch.long)        # [L]
        strokes = torch.tensor(data["strokes"], dtype=torch.float32)       # [T, 4]
        stroke_mask = torch.tensor(data["stroke_mask"], dtype=torch.bool)  # [T]

        return dict(
            text=data["text"],
            text_ids=text_ids,
            strokes=strokes,
            stroke_mask=stroke_mask
        )

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
                 mdn_components=20, pen_states=2):
        super().__init__()
        self.d_model = d_model
        self.text_tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.stroke_in = nn.Linear( pen_states, d_model)
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

    def forward(self, text_ids, text_mask, strokes, stroke_mask):
        B, T, _ = strokes.shape
        te = self.text_tok(text_ids) * math.sqrt(self.d_model)
        te = self.text_pos(te)
        enc = self.encoder(te, src_key_padding_mask=~text_mask)

        di = self.stroke_in(strokes)
        di = self.stroke_pos(di)
        tgt_mask = torch.triu(torch.ones((T, T), device=strokes.device), diagonal=1).bool()
        dec = self.decoder(tgt=di, memory=enc,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=~text_mask,
                           tgt_key_padding_mask=~stroke_mask)

        pi = self.mdn_pi(dec)
        mu_x = self.mdn_mu_x(dec)
        mu_y = self.mdn_mu_y(dec)
        sigma_x = torch.exp(self.mdn_sigma_x(dec))
        sigma_y = torch.exp(self.mdn_sigma_y(dec))
        rho = torch.tanh(self.mdn_rho(dec))
        pen = self.pen_logits(dec)
        return dict(pi=pi, mu_x=mu_x, mu_y=mu_y,
                    sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
                    pen=pen)

# ======================
# Training Loop
# ======================
def train_loop(data_dir, vocab, epochs=1000, batch_size=32, lr=1e-4, device="cuda"):
    dataset = StrokeDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HandwritingTransformer(
        vocab_size=len(vocab)+1,
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, mdn_components=20, pen_states=4
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            text_ids = batch["text_ids"].to(device)
            text_mask = text_ids != 0
            strokes = batch["strokes"].to(device)
            stroke_mask = batch["stroke_mask"].to(device)

            out = model(text_ids, text_mask, strokes, stroke_mask)

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
        torch.save(model.state_dict(), f"checkpoints/handwriting_epoch{epoch}.pt")

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
             "ら": 31, "り": 32, "る": 33, "れ": 34, "ろ": 35,
             "わ": 36, "を": 37, "ん": 38, "。": 39, "、": 40,}  # 必要に応じて拡張

    train_loop("./sentences-4dim/oneletters/norm_padding", vocab, epochs=1000, batch_size=32, lr=1e-4)
    