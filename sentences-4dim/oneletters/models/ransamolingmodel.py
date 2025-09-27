import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # tqdmを追加すると進捗が分かりやすいです

# ======================
# Positional Encoding (変更なし)
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
# Handwriting Transformer (encodeとdecodeに分割し、よりクリーンに)
# ======================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1,
                 mdn_components=20, pen_states=2, stroke_dim=4): # stroke_dim引数を追加
        super().__init__()
        self.d_model = d_model
        self.text_tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.stroke_in = nn.Linear(stroke_dim, d_model)
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

    def encode(self, text_ids, text_mask):
        te = self.text_tok(text_ids) * math.sqrt(self.d_model)
        te = self.text_pos(te)
        memory = self.encoder(te, src_key_padding_mask=~text_mask)
        return memory

    def decode(self, strokes, stroke_mask, memory, memory_mask):
        B, T, _ = strokes.shape
        di = self.stroke_in(strokes)
        di = self.stroke_pos(di)
        tgt_mask = torch.triu(torch.ones(
            (T, T), device=strokes.device), diagonal=1).bool()
        
        dec = self.decoder(tgt=di, memory=memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=~memory_mask,
                           tgt_key_padding_mask=~stroke_mask)
        return dec
        
    def forward(self, text_ids, text_mask, strokes, stroke_mask):
        """学習時のフォワードパス（今回は使わない）"""
        memory = self.encode(text_ids, text_mask)
        dec = self.decode(strokes, stroke_mask, memory, text_mask)
        # ... (損失計算用のヘッドをここに書く) ...

# ======================
# サンプリング関数 (モデルのencode/decodeを使うように修正)
# ======================
def sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho):
    # softmaxで確率に変換し、numpy配列に
    pi_probs = torch.softmax(pi, dim=-1).detach().cpu().numpy()
    # 確率pi_probsに基づいて、どの混合要素（ガウス分布）を使うか選択
    comp_idx = np.random.choice(len(pi_probs), p=pi_probs)

    # 選択されたガウス分布のパラメータを取得
    mean = [mu_x[comp_idx].item(), mu_y[comp_idx].item()]
    sx = sigma_x[comp_idx].item()
    sy = sigma_y[comp_idx].item()
    r = rho[comp_idx].item()
    
    # 共分散行列を作成
    cov = [[sx**2, r * sx * sy], [r * sx * sy, sy**2]]
    
    # 2変量正規分布からサンプリング
    dx, dy = np.random.multivariate_normal(mean, cov)
    return dx, dy


def generate_strokes(model, vocab, text="あ", max_steps=200, device="cuda"):
    model.eval()
    
    # 1. テキストIDを準備し、エンコーダーで文脈(memory)を一度だけ計算
    text_ids = torch.tensor([[vocab[ch] for ch in text]], dtype=torch.long, device=device)
    text_mask = (text_ids != 0)
    memory = model.encode(text_ids, text_mask)

    # 2. 生成ループの初期化
    # 最初のストロークは (0,0) でペンを下ろした状態
    stroke_sequence = [torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device)]

    for _ in range(max_steps):
        # 現在のストローク系列全体をデコーダーの入力とする
        strokes_tensor = torch.cat(stroke_sequence, dim=0).unsqueeze(0) # [1, 現在の長さ, 4]
        
        # デコーダーを実行
        # stroke_maskは生成時には使わないのでダミー（すべてTrue）を渡す
        dummy_stroke_mask = torch.ones(strokes_tensor.shape[:2], dtype=torch.bool, device=device)
        dec_out = model.decode(strokes_tensor, dummy_stroke_mask, memory, text_mask)
        
        # 最後のステップの出力だけを取り出す
        last_step_out = dec_out[0, -1]

        # 3. 最後のステップの出力からMDNパラメータとペンの状態を取得
        pi = model.mdn_pi(last_step_out)
        mu_x, mu_y = model.mdn_mu_x(last_step_out), model.mdn_mu_y(last_step_out)
        sigma_x, sigma_y = torch.exp(model.mdn_sigma_x(last_step_out)), torch.exp(model.mdn_sigma_y(last_step_out))
        rho = torch.tanh(model.mdn_rho(last_step_out))
        pen_logits = model.pen_logits(last_step_out)

        # 4. 次のストロークをサンプリング
        dx, dy = sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho)
        pen_probs = torch.softmax(pen_logits, dim=-1)
        pen_state = torch.multinomial(pen_probs, 1).item()
        
        # ペンの状態をone-hotベクトルに変換
        pen_onehot = [1.0, 0.0] if pen_state == 0 else [0.0, 1.0]
        
        # 新しいストロークを追加
        new_stroke = torch.tensor([[dx, dy] + pen_onehot], dtype=torch.float32, device=device)
        stroke_sequence.append(new_stroke)
        
        # もしペンが紙から離れたら（end of stroke）、生成を終了
        if pen_state == 1:
            break

    return torch.cat(stroke_sequence, dim=0).cpu().numpy()

# ======================
# プロット関数 (変更なし)
# ======================
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
    # ペンが down (ps > 0.5) の間だけ線を引く
    for i in range(1, len(xs)):
        if ps[i-1] > 0.5: # 1つ前の点がペンを下ろした状態なら線を引く
            plt.plot(xs[i-1:i+1], ys[i-1:i+1], "k-", linewidth=2.0)
    
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis('off') # 軸を非表示に
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close() # プロットを閉じる

# ======================
# 実行ブロック
# ======================
if __name__ == "__main__":
    # vocab辞書の重複と番号の誤りを修正
    vocab = {
        "あ": 1, "い": 2, "う": 3, "え": 4, "お": 5,
        "か": 6, "き": 7, "く": 8, "け": 9, "こ": 10,
        "さ": 11, "し": 12, "す": 13, "せ": 14, "そ": 15,
        "た": 16, "ち": 17, "つ": 18, "て": 19, "と": 20,
        "な": 21, "に": 22, "ぬ": 23, "ね": 24, "の": 25,
        "は": 26, "ひ": 27, "ふ": 28, "へ": 29, "ほ": 30,
        "ま": 31, "み": 32, "む": 33, "め": 34, "も": 35,
        "や": 36, "ゆ": 37, "よ": 38,
        "ら": 39, "り": 40, "る": 41, "れ": 42, "ろ": 43,
        "わ": 44, "を": 45, "ん": 46, "。": 47, "、": 48,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HandwritingTransformer(
        vocab_size=len(vocab)+1,
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, mdn_components=20, pen_states=2, stroke_dim=4
    ).to(device)

    # チェックポイントのパスを正しく指定してください
    checkpoint_path = "checkpoints/handwriting_epoch10.pt"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"🚨 Error: Checkpoint file not found at {checkpoint_path}")
        exit() # ファイルがなければ終了

    output_folder = "generated_strokes"
    os.makedirs(output_folder, exist_ok=True)

    # vocabの文字をループで処理
    # tqdmでラップするとプログレスバーが表示されます
    for char in tqdm(vocab.keys(), desc="Generating Characters"):
        try:
            strokes = generate_strokes(model, vocab, text=char, max_steps=1000, device=device)
            xy = strokes_to_xy(strokes)
            
            save_path = os.path.join(output_folder, f"{char}.png")
            plot_strokes(xy, save_path)

        except Exception as e:
            print(f"Error processing '{char}': {e}")
            # エラーの詳細を知りたい場合は、以下の行のコメントを解除
            # import traceback
            # traceback.print_exc()

    print(f"\n✅ All characters have been processed and saved to '{output_folder}' folder.")