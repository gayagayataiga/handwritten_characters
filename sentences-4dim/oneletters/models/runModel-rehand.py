import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ======================
# Handwriting Transformer (学習時と完全に同じ構造)
# ======================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=512, dropout=0.1,
                 mdn_components=20, pen_states=2, stroke_dim=4):
        super().__init__()
        self.d_model = d_model
        self.text_tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 学習時と同じ層定義
        self.stroke_embed = nn.Linear(stroke_dim, d_model) 
        self.decoder_input_proj = nn.Linear(d_model + d_model, d_model)

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

    def encode(self, text_ids, text_mask):
        te = self.text_tok(text_ids) * math.sqrt(self.d_model)
        te = self.text_pos(te)
        memory = self.encoder(te, src_key_padding_mask=~text_mask)
        return memory

    def decode(self, strokes, stroke_mask, memory, memory_mask):
        B, T, _ = strokes.shape
        context_vec = memory.squeeze(1)
        context_vec_expanded = context_vec.unsqueeze(1).expand(-1, T, -1)
        stroke_embedding = self.stroke_embed(strokes)
        decoder_input = torch.cat([stroke_embedding, context_vec_expanded], dim=-1)
        di = self.decoder_input_proj(decoder_input)
        di = self.stroke_pos(di)
        
        tgt_mask = torch.triu(torch.ones((T, T), device=strokes.device), diagonal=1).bool()
        
        dec = self.decoder(tgt=di, memory=memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=~memory_mask,
                           tgt_key_padding_mask=~stroke_mask)

        pi = self.mdn_pi(dec)
        mu_x = self.mdn_mu_x(dec)
        mu_y = self.mdn_mu_y(dec)
        sigma_x = torch.exp(self.mdn_sigma_x(dec)).clamp(min=1e-4)
        sigma_y = torch.exp(self.mdn_sigma_y(dec)).clamp(min=1e-4)
        rho = torch.tanh(self.mdn_rho(dec)).clamp(-0.99999, 0.99999)
        pen = self.pen_logits(dec)
        
        return dict(pi=pi, mu_x=mu_x, mu_y=mu_y,
                    sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
                    pen=pen)

# ======================
# サンプリング関数 (修正後)
# ======================
def sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho):
    # piは [1, 1, 20] のような形状なので、 squeeze() で [20] のベクトルにする
    pi_norm = torch.softmax(pi.squeeze(), dim=-1).cpu().numpy()
    
    # 浮動小数点数の誤差で合計が1にならない場合があるので、再正規化して確実にする
    pi_norm /= pi_norm.sum()
    
    # どのガウス分布を使うか、確率piに従って選択
    comp_idx = np.random.choice(len(pi_norm), p=pi_norm)

    # 各パラメータも [1, 1, 20] の形状なので、squeeze() で [20] にしてから値を取得
    mean = [mu_x.squeeze()[comp_idx].item(), mu_y.squeeze()[comp_idx].item()]
    sx = sigma_x.squeeze()[comp_idx].item()
    sy = sigma_y.squeeze()[comp_idx].item()
    r = rho.squeeze()[comp_idx].item()
    
    # 共分散行列を作成
    cov = [[sx*sx, r*sx*sy], [r*sx*sy, sy*sy]]
    
    # 選ばれたガウス分布からdx, dyをサンプリング
    dx, dy = np.random.multivariate_normal(mean, cov)
    return dx, dy

# ======================
# 生成関数 (修正後)
# ======================
@torch.no_grad()
def generate_strokes(model, vocab, text="あ", max_steps=200, device="cuda"):
    model.eval()
    
    # 1. テキストをエンコードして文脈ベクトル memory を作成
    text_ids = torch.tensor([[vocab[ch] for ch in text]], dtype=torch.long, device=device)
    text_mask = (text_ids != 0)
    memory = model.encode(text_ids, text_mask)

    # 2. 生成ループの初期化
    # 最初のストローク点は [dx, dy, pen_down, pen_up] ではなく [dx, dy, dt, pen_state] に合わせる
    # pen_state は 0 or 1. ここではペンが下りている状態(1)から始める
    current_stroke = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    generated_sequence = []

    for _ in range(max_steps):
        # 3. 現在のストローク系列を使って、次の点のパラメータを予測
        stroke_mask = torch.ones(current_stroke.shape[:-1], dtype=torch.bool, device=device)
        out = model.decode(current_stroke, stroke_mask, memory, text_mask)
        
        # 最後の点のパラメータを取得
        pi = out['pi'][:, -1:, :]
        mu_x = out['mu_x'][:, -1:, :]
        mu_y = out['mu_y'][:, -1:, :]
        sigma_x = out['sigma_x'][:, -1:, :]
        sigma_y = out['sigma_y'][:, -1:, :]
        rho = out['rho'][:, -1:, :]
        pen_logits = out['pen'][:, -1:, :]
        
        # 4. パラメータから次の点をサンプリング
        dx, dy = sample_bivariate(pi, mu_x, mu_y, sigma_x, sigma_y, rho)
        pen_state = torch.multinomial(torch.softmax(pen_logits.squeeze(), dim=-1), 1).item()

        # 5. サンプリングした点を結果に追加し、次の入力とする
        # pen_state 1がペンダウン, 0がペンアップと仮定
        # dtは今回は使わないので0で固定
        new_point = torch.tensor([[[dx, dy, 0.0, float(pen_state)]]], dtype=torch.float32, device=device)
        generated_sequence.append(new_point.squeeze().cpu().numpy())
        current_stroke = torch.cat([current_stroke, new_point], dim=1)
        
        # ペンが離れたら（かつ、ある程度の長さがあったら）生成を終了する
        if pen_state == 0 and len(generated_sequence) > 10:
            break
            
    return np.array(generated_sequence)

# ======================
# プロット関数など (変更なし)
# ======================
def strokes_to_xy(strokes):
    """差分座標から絶対座標に変換"""
    abs_coords = np.cumsum(strokes[:, :2], axis=0)
    # pen_state (4列目) を結合
    return np.hstack([abs_coords, strokes[:, 3:]])

def plot_strokes(xy, save_path):
    plt.figure(figsize=(4, 4))
    
    # pen_stateが1 (ペンダウン) の部分だけを繋げてプロット
    pen_down_segments = np.where(xy[:, 2] == 1)[0]
    
    # 連続したセグメントを見つける
    if len(pen_down_segments) > 0:
        segments = np.split(pen_down_segments, np.where(np.diff(pen_down_segments) != 1)[0] + 1)
        for segment in segments:
            if len(segment) > 1:
                plt.plot(xy[segment, 0], xy[segment, 1], "k-")
    
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# ======================
# 実行ブロック
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
             "わ": 44, "を": 45, "ん": 46, "。": 47, "、": 48}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # モデルのインスタンス化 (学習時と同じパラメータ)
    model = HandwritingTransformer(
        vocab_size=len(vocab)+1,
        stroke_dim=4 
    ).to(device)

    # 読み込むチェックポイントのパス
    checkpoint_path = "checkpoints/handwriting_epoch5.pt" #  <-- ここを目的のファイルに変更
    
    if not os.path.exists(checkpoint_path):
        print(f"エラー: チェックポイントファイルが見つかりません: {checkpoint_path}")
    else:
        print(f"チェックポイントを読み込み中: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        output_folder = "generated_images"
        os.makedirs(output_folder, exist_ok=True)

        for char in tqdm(vocab.keys(), desc="文字を生成中"):
            try:
                strokes = generate_strokes(model, vocab, text=char, max_steps=150, device=device)
                
                if len(strokes) > 0:
                    xy = strokes_to_xy(strokes)
                    save_path = os.path.join(output_folder, f"{char}.png")
                    plot_strokes(xy, save_path)
                else:
                    print(f"'{char}'のストローク生成に失敗しました（空の系列）。")

            except Exception as e:
                print(f"'{char}'の処理中にエラーが発生しました: {e}")

        print(f"\n生成完了。画像は'{output_folder}'フォルダに保存されました。")