import torch
import torch.nn as nn

class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # --- Encoder（テキスト側） ---
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(500, d_model)  # 最大500文字対応

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder（ストローク側） ---
        self.stroke_embedding = nn.Linear(3, d_model)  # (dx, dy, end) → d_model
        self.pos_decoder = nn.Embedding(1000, d_model) # 最大1000ステップ対応

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.stroke_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 出力ヘッド ---
        self.fc_xy = nn.Linear(d_model, 2)   # Δx, Δy
        self.fc_end = nn.Linear(d_model, 2)  # end (0=続く, 1=終了)

    def forward(self, text_ids, stroke_seq):
        """
        text_ids: (batch, text_len)
        stroke_seq: (batch, seq_len, 3)  # 教師強制 (dx, dy, end)
        """
        B, T_text = text_ids.shape
        _, T_stroke, _ = stroke_seq.shape

        # --- テキスト埋め込み ---
        pos_text = torch.arange(T_text, device=text_ids.device).unsqueeze(0).expand(B, -1)
        text_emb = self.text_embedding(text_ids) + self.pos_encoder(pos_text)
        text_emb = text_emb.transpose(0, 1)  # (T_text, B, d_model)
        memory = self.text_encoder(text_emb)

        # --- ストローク埋め込み ---
        pos_stroke = torch.arange(T_stroke, device=text_ids.device).unsqueeze(0).expand(B, -1)
        stroke_emb = self.stroke_embedding(stroke_seq) + self.pos_decoder(pos_stroke)
        stroke_emb = stroke_emb.transpose(0, 1)  # (T_stroke, B, d_model)

        # --- デコーダ ---
        out = self.stroke_decoder(stroke_emb, memory)  # (T_stroke, B, d_model)

        # --- 出力 ---
        out = out.transpose(0, 1)  # (B, T_stroke, d_model)
        dxdy = self.fc_xy(out)     # (B, T_stroke, 2)
        end = self.fc_end(out)     # (B, T_stroke, 2)

        return dxdy, end


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# モデルをインポート（前回の HandwritingTransformer）
# from model import HandwritingTransformer

# ==============================
# 損失関数
# ==============================
def compute_loss(dxdy_pred, end_pred, seq_gt):
    """
    dxdy_pred: (B, T, 2)
    end_pred:  (B, T, 2)
    seq_gt:    (B, T, 3)  # [dx, dy, end]
    """
    # 教師データ分割
    dxdy_gt = seq_gt[:, :, :2]      # (B, T, 2)
    end_gt = seq_gt[:, :, 2].long() # (B, T)

    # Δx, Δy: MSE Loss
    mse_loss = nn.MSELoss()(dxdy_pred, dxdy_gt)

    # end: CrossEntropy
    ce_loss = nn.CrossEntropyLoss()(end_pred.reshape(-1, 2), end_gt.reshape(-1))

    # 総合Loss
    loss = mse_loss + ce_loss
    return loss, mse_loss.item(), ce_loss.item()


# ==============================
# 学習ループ
# ==============================
def train_model(model, dataloader, epochs=10, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, total_mse, total_ce = 0, 0, 0

        for text_ids, seq in dataloader:
            text_ids = text_ids.to(device)   # (B, T_text)
            seq = seq.to(device)             # (B, T_seq, 3)

            # 教師強制（decoderに入れる）
            dxdy_pred, end_pred = model(text_ids, seq)

            # 損失計算
            loss, mse, ce = compute_loss(dxdy_pred, end_pred, seq)

            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse
            total_ce += ce

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_ce = total_ce / len(dataloader)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, CE: {avg_ce:.4f})")

    return model


import torch
import matplotlib.pyplot as plt

def generate_strokes(model, text, char2id, max_len=500, device="cuda"):
    model.eval()
    with torch.no_grad():
        # --- テキストをトークン化 ---
        text_ids = [char2id.get(ch, char2id["<unk>"]) for ch in text]
        text_ids = [char2id["<s>"]] + text_ids + [char2id["</s>"]]
        text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)

        # --- 開始ストローク ---
        seq = torch.zeros((1, 1, 3), device=device)  # (B, T=1, 3)

        strokes = []
        x, y = 0.0, 0.0

        for t in range(max_len):
            dxdy_pred, end_pred = model(text_ids, seq)

            # 最新の予測を取得
            dxdy = dxdy_pred[:, -1, :]        # (B, 2)
            end_logits = end_pred[:, -1, :]   # (B, 2)

            # サンプリング
            dx, dy = dxdy[0].cpu().numpy()
            end_prob = torch.softmax(end_logits, dim=-1)[0].cpu().numpy()
            end = 1 if end_prob[1] > 0.5 else 0

            # 座標を更新
            x += dx
            y += dy
            strokes.append((x, y, end))

            # 次の入力に追加
            new_point = torch.tensor([[[dx, dy, end]]], dtype=torch.float32, device=device)
            seq = torch.cat([seq, new_point], dim=1)

            if end == 1:
                break

    return strokes


# ==============================
# 描画用関数
# ==============================
def plot_strokes(strokes, title="Generated Handwriting"):
    xs, ys = [], []
    for (x, y, end) in strokes:
        xs.append(x)
        ys.append(-y)  # y軸を反転すると自然に見える
        if end == 1:
            plt.plot(xs, ys, "k-")
            xs, ys = [], []
    if xs:  # 最後のストローク
        plt.plot(xs, ys, "k-")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    text = "私は天才です"
    strokes = generate_strokes(trained_model, text, char2id, max_len=300, device="cuda")
    plot_strokes(strokes, title=text)
