import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# Positional Encoding
# ======================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ======================
# Handwriting Transformer
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.stroke_in = nn.Linear(2 + pen_states, d_model)
        self.stroke_pos = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
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
        tgt_mask = torch.triu(torch.ones(
            (T, T), device=strokes.device), diagonal=1).bool()
        dec = self.decoder(tgt=di, memory=enc,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=~text_mask,
                           tgt_key_padding_mask=~stroke_mask)

        return dict(
            pi=self.mdn_pi(dec),
            mu_x=self.mdn_mu_x(dec),
            mu_y=self.mdn_mu_y(dec),
            sigma_x=torch.exp(self.mdn_sigma_x(dec)),
            sigma_y=torch.exp(self.mdn_sigma_y(dec)),
            rho=torch.tanh(self.mdn_rho(dec)),
            pen=self.pen_logits(dec)
        )

# ======================
# サンプリング関数
# ======================


def sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho):
    pi = torch.softmax(pi, dim=-1).detach().cpu().numpy()
    comp = np.random.choice(len(pi), p=pi)

    mean = [mu_x[comp].item(), mu_y[comp].item()]
    cov = [
        [sigma_x[comp].item()**2, rho[comp].item() *
         sigma_x[comp].item()*sigma_y[comp].item()],
        [rho[comp].item()*sigma_x[comp].item()*sigma_y[comp].item(),
         sigma_y[comp].item()**2]
    ]
    dx, dy = np.random.multivariate_normal(mean, cov)
    return dx, dy


def generate_strokes(model, vocab, text="あ", max_steps=200, device="cuda"):
    model.eval()
    text_ids = torch.tensor([[vocab[ch] for ch in text]],
                            dtype=torch.long, device=device)
    text_mask = text_ids != 0
    enc = model.encoder(model.text_pos(model.text_tok(text_ids) * np.sqrt(model.d_model)),
                        src_key_padding_mask=~text_mask)

    strokes = [torch.tensor([[0.0, 0.0, 1.0, 0.0]],
                            dtype=torch.float32, device=device)]

    for _ in range(max_steps):
        di = model.stroke_in(torch.cat(strokes, dim=0).unsqueeze(0))
        di = model.stroke_pos(di)
        T = di.size(1)
        tgt_mask = torch.triu(torch.ones(
            (T, T), device=device), diagonal=1).bool()
        dec = model.decoder(tgt=di, memory=enc,
                            tgt_mask=tgt_mask,
                            memory_key_padding_mask=~text_mask)
        last = dec[0, -1]

        pi = model.mdn_pi(last)
        mu_x = model.mdn_mu_x(last)
        mu_y = model.mdn_mu_y(last)
        sigma_x = torch.exp(model.mdn_sigma_x(last))
        sigma_y = torch.exp(model.mdn_sigma_y(last))
        rho = torch.tanh(model.mdn_rho(last))
        pen_logits = model.pen_logits(last)

        dx, dy = sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho)
        pen_state = torch.multinomial(
            torch.softmax(pen_logits, dim=-1), 1).item()
        pen_onehot = [1, 0] if pen_state == 0 else [0, 1]
        new_step = torch.tensor([[dx, dy] + pen_onehot],
                                dtype=torch.float32, device=device)
        strokes.append(new_step)

    return torch.cat(strokes, dim=0).cpu().numpy()


def strokes_to_xy(strokes):
    x, y = 0, 0
    xy = []
    for dx, dy, p_down, p_up in strokes:
        x += dx
        y += dy
        xy.append((x, y, p_down))
    return np.array(xy)


def plot_strokes(xy, save_path):
    plt.figure(figsize=(4, 4))
    xs, ys, ps = xy[:, 0], xy[:, 1], xy[:, 2]
    start = 0
    for i in range(1, len(xs)):
        if ps[i] > 0.5:  # pen down
            plt.plot(xs[start:i+1], ys[start:i+1], "k-")
        else:
            start = i
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


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
             "わ": 36, "を": 37, "ん": 38, "。": 39, "、": 40, }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HandwritingTransformer(
        vocab_size=len(vocab)+1,
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, mdn_components=20, pen_states=2
    ).to(device)

    model.load_state_dict(torch.load(
        "checkpoints/handwriting_epoch100.pt", map_location=device))

    # strokes = generate_strokes(model, vocab, text="り", max_steps=100, device=device)
    # xy = strokes_to_xy(strokes)
    # plot_strokes(xy)
    output_folder = "sentences-4dim/oneletters/generated_strokes"
    os.makedirs(output_folder, exist_ok=True)  # フォルダが存在しない場合は自動で作成

    # 3. リストの文字を1つずつループで処理
    for char in vocab.keys():
        try:
            print(f"Processing: '{char}'")

            # 4. 各文字に対してストロークを生成
            # text="..." の部分をループ変数 `char` に変更
            strokes = generate_strokes(
                model, vocab, text=char, max_steps=2000, device=device)
            xy = strokes_to_xy(strokes)

            # 5. ストロークをプロット
            # plot_strokes関数は内部でmatplotlibを呼び出していると想定
            save_path = os.path.join(output_folder, f"{char}.png")
            plot_strokes(xy, save_path)

            print(f"-> Saved to {save_path}")

            # 7. 現在のプロットを閉じて、次のループに備える
            # これがないと、前の文字のプロットの上に新しいプロットが重ねて描画されてしまいます
            plt.close()

        except Exception as e:
            print(f"Error processing '{char}': {e}")

        print("\nAll characters have been processed.")
