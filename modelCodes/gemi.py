import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import MultivariateNormal, Categorical
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import datetime
import json
import os
from functools import partial
import math
import numpy as np

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 0. ハイパーパラメータ
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
N_MIXTURES = 20 # 混合ガウス分布の数
WARMUP_STEPS = 4000 # 学習率スケジューラ用

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 1. モデル定義 (HandwritingTransformer)
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, n_mixtures, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.d_model = d_model
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(5000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True, activation='gelu')
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.stroke_embedding = nn.Linear(3, d_model)
        self.pos_decoder = nn.Embedding(10000, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True, activation='gelu')
        self.stroke_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.mdn_head = nn.Linear(d_model, self.n_mixtures * 6)
        self.fc_end = nn.Linear(d_model, 2)

    def forward(self, text_ids, stroke_seq, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, T_text = text_ids.shape
        _, T_stroke, _ = stroke_seq.shape
        device = text_ids.device
        pos_text = torch.arange(0, T_text, device=device).unsqueeze(0).repeat(B, 1)
        text_emb = self.text_embedding(text_ids) * math.sqrt(self.d_model) + self.pos_encoder(pos_text)
        memory = self.text_encoder(text_emb, src_key_padding_mask=src_key_padding_mask)
        pos_stroke = torch.arange(0, T_stroke, device=device).unsqueeze(0).repeat(B, 1)
        stroke_emb = self.stroke_embedding(stroke_seq) * math.sqrt(self.d_model) + self.pos_decoder(pos_stroke)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_stroke, device=device)
        out = self.stroke_decoder(stroke_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        mdn_params = self.mdn_head(out)
        pi_logits, mu_x, mu_y, sigma_x_log, sigma_y_log, rho_xy_tanh = mdn_params.split(self.n_mixtures, dim=-1)
        pi = torch.softmax(pi_logits, dim=-1)
        sigma_x = torch.exp(sigma_x_log)
        sigma_y = torch.exp(sigma_y_log)
        rho_xy = torch.tanh(rho_xy_tanh)
        end = self.fc_end(out)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 2. 損失関数と学習ループ
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
def mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, target_dxdy):
    target_dx = target_dxdy[:, 0].unsqueeze(-1)
    target_dy = target_dxdy[:, 1].unsqueeze(-1)
    norm_x = (target_dx - mu_x) / sigma_x
    norm_y = (target_dy - mu_y) / sigma_y
    z = norm_x**2 + norm_y**2 - 2 * rho_xy * norm_x * norm_y
    one_minus_rho_sq = 1 - rho_xy**2
    one_minus_rho_sq = torch.clamp(one_minus_rho_sq, min=1e-5)
    log_pdf = -z / (2 * one_minus_rho_sq) - torch.log(2 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho_sq))
    log_pi = torch.log(pi + 1e-10)
    log_likelihood = torch.logsumexp(log_pi + log_pdf, dim=-1)
    return -log_likelihood.mean()

def compute_loss(mdn_params, end_pred, seq_gt, mask):
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = mdn_params
    dxdy_gt = seq_gt[:, :, :2]
    end_gt = seq_gt[:, :, 2].long()
    pi_masked, mu_x_masked, mu_y_masked = pi[mask], mu_x[mask], mu_y[mask]
    sigma_x_masked, sigma_y_masked, rho_xy_masked = sigma_x[mask], sigma_y[mask], rho_xy[mask]
    dxdy_gt_masked = dxdy_gt[mask]
    loss_mdn = mdn_loss(pi_masked, mu_x_masked, mu_y_masked, sigma_x_masked, sigma_y_masked, rho_xy_masked, dxdy_gt_masked) if pi_masked.nelement() > 0 else torch.tensor(0.0, device=pi.device)
    end_pred_masked, end_gt_masked = end_pred[mask], end_gt[mask]
    loss_ce = nn.CrossEntropyLoss()(end_pred_masked, end_gt_masked) if end_pred_masked.nelement() > 0 else torch.tensor(0.0, device=end_pred.device)
    return loss_mdn + loss_ce, loss_mdn.item(), loss_ce.item()

def train_model(model, dataloader, epochs=50, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # 変更点: 学習率スケジューラを追加
    lr_scheduler = LambdaLR(optimizer, lambda step: min((step + 1)**-0.5, (step + 1) * WARMUP_STEPS**-1.5))
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mdn, total_ce = 0, 0, 0
        for text_ids, seq, text_padding_mask, seq_padding_mask in dataloader:
            text_ids, seq, text_padding_mask, seq_padding_mask = text_ids.to(device), seq.to(device), text_padding_mask.to(device), seq_padding_mask.to(device)
            decoder_input_seq, target_seq = seq[:, :-1], seq[:, 1:]
            decoder_input_padding_mask, target_mask = seq_padding_mask[:, :-1], ~seq_padding_mask[:, 1:]
            if target_seq.shape[1] == 0: continue
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(text_ids, decoder_input_seq, text_padding_mask, decoder_input_padding_mask)
            mdn_params = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
            loss, mdn, ce = compute_loss(mdn_params, end_pred, target_seq, target_mask)
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step() # 変更点: スケジューラを更新
                total_loss += loss.item(); total_mdn += mdn; total_ce += ce
        avg_loss = total_loss / len(dataloader) if dataloader else 0
        avg_mdn = total_mdn / len(dataloader) if dataloader else 0
        avg_ce = total_ce / len(dataloader) if dataloader else 0
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} (MDN: {avg_mdn:.4f}, CE: {avg_ce:.4f}) LR: {lr_scheduler.get_last_lr()[0]:.6f}")
    return model

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 3. 生成と描画
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 変更点: temperature を追加
def sample_from_mdn(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature=0.8):
    # Temperatureを適用してpiの分布を調整
    pi = torch.pow(pi, 1/temperature)
    pi /= pi.sum()

    cat_dist = Categorical(pi)
    mixture_idx = cat_dist.sample()
    mu_x_s, mu_y_s = mu_x[mixture_idx], mu_y[mixture_idx]
    # Temperatureを適用して分散を調整
    sx = sigma_x[mixture_idx] * math.sqrt(temperature)
    sy = sigma_y[mixture_idx] * math.sqrt(temperature)
    rho = rho_xy[mixture_idx]
    mean = torch.tensor([mu_x_s, mu_y_s], device=pi.device)
    cov = torch.tensor([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]], device=pi.device)
    mvn = MultivariateNormal(mean, cov)
    dxdy = mvn.sample()
    return dxdy[0].item(), dxdy[1].item()

def generate_strokes(model, text, char2id, id2char, stats, max_len=700, device="cuda", temperature=0.8):
    model.eval().to(device)
    mean_dx, std_dx, mean_dy, std_dy = stats['mean_dx'], stats['std_dx'], stats['mean_dy'], stats['std_dy']
    with torch.no_grad():
        text_ids = [char2id.get(c, char2id["<unk>"]) for c in text]
        text_ids = torch.tensor([char2id["<s>"]] + text_ids + [char2id["</s>"]], dtype=torch.long, device=device).unsqueeze(0)
        seq = torch.zeros((1, 1, 3), device=device)
        strokes, x, y = [], 0.0, 0.0
        for _ in range(max_len):
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(text_ids, seq)
            pi_last, mu_x_last, mu_y_last = pi[0, -1, :], mu_x[0, -1, :], mu_y[0, -1, :]
            sigma_x_last, sigma_y_last, rho_xy_last = sigma_x[0, -1, :], sigma_y[0, -1, :], rho_xy[0, -1, :]
            end_logits_last = end_pred[0, -1, :]
            # 変更点: temperature を渡す
            dx_norm, dy_norm = sample_from_mdn(pi_last, mu_x_last, mu_y_last, sigma_x_last, sigma_y_last, rho_xy_last, temperature)
            end = torch.argmax(end_logits_last).item()
            # 変更点: 正規化を解除して元のスケールに戻す
            dx = dx_norm * std_dx + mean_dx
            dy = dy_norm * std_dy + mean_dy
            x += dx; y += dy
            strokes.append((x, y, end))
            if end == 1 and len(strokes) > 5: break
            # 変更点: 次の入力も正規化された値を使う
            next_point = torch.tensor([[[dx_norm, dy_norm, float(end)]]], device=device, dtype=torch.float32)
            seq = torch.cat([seq, next_point], dim=1)
    return strokes

def plot_strokes(strokes, title="Generated Handwriting"):
    plt.figure(figsize=(6, 6)); ax = plt.gca(); ax.set_aspect('equal', adjustable='box'); plt.title(title, fontsize=16)
    current_stroke_x, current_stroke_y = [], []
    for (x, y, end) in strokes:
        current_stroke_x.append(x); current_stroke_y.append(-y)
        if end == 1:
            if current_stroke_x: ax.plot(current_stroke_x, current_stroke_y, "k-", linewidth=2.0)
            current_stroke_x, current_stroke_y = [], []
    if current_stroke_x: ax.plot(current_stroke_x, current_stroke_y, "k-", linewidth=2.0)
    plt.show()

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 4. データセット関連
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 変更点: データ正規化関数を追加
def calculate_stats_and_normalize(dataset):
    all_dx, all_dy = [], []
    for _, strokes in dataset:
        for dx, dy, _ in strokes:
            all_dx.append(dx)
            all_dy.append(dy)
    
    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)
    
    stats = {
        'mean_dx': np.mean(all_dx), 'std_dx': np.std(all_dx),
        'mean_dy': np.mean(all_dy), 'std_dy': np.std(all_dy)
    }
    
    normalized_dataset = []
    for text, strokes in dataset:
        norm_strokes = [
            ((dx - stats['mean_dx']) / stats['std_dx'],
             (dy - stats['mean_dy']) / stats['std_dy'],
             end) for dx, dy, end in strokes
        ]
        normalized_dataset.append((text, norm_strokes))
        
    print(f"Data normalized. Stats: {stats}")
    return normalized_dataset, stats

class HandwritingDataset(Dataset):
    def __init__(self, data, char2id): self.data, self.char2id = data, char2id
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text, strokes = self.data[idx]
        text_ids = [self.char2id["<s>"]] + [self.char2id.get(c, self.char2id["<unk>"]) for c in text] + [self.char2id["</s>"]]
        stroke_seq = [(0.0, 0.0, 0.0)] + strokes
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(stroke_seq, dtype=torch.float32)

def load_dataset_from_directory(directory_path):
    # ... (変更なし) ...
    dataset = []
    if not os.path.exists(directory_path): print(f"Warning: Directory not found: {directory_path}"); return dataset
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                text, strokes = data.get('text'), data.get('sequence')
                if text is not None and strokes is not None: dataset.append((text, [tuple(p) for p in strokes]))
            except Exception as e: print(f"Could not load/parse {filename}: {e}")
    return dataset

def collate_fn(batch, text_pad_id):
    # ... (変更なし) ...
    texts, strokes = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=text_pad_id)
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    text_padding_mask = (padded_texts == text_pad_id)
    stroke_lengths = torch.tensor([len(s) for s in strokes])
    stroke_padding_mask = torch.arange(padded_strokes.size(1))[None, :] >= stroke_lengths[:, None]
    return padded_texts, padded_strokes, text_padding_mask, stroke_padding_mask


# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# 5. メイン実行ブロック
# _=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
if __name__ == "__main__":
    D_MODEL, NHEAD, NUM_LAYERS = 256, 4, 6
    DROPOUT, EPOCHS, LR, BATCH_SIZE = 0.1, 100, 1.0, 8 # LRはスケジューラが管理するので1.0でOK
    
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    
    DATA_DIRECTORY = "processed_json"
    my_data = load_dataset_from_directory(DATA_DIRECTORY)
    if not my_data: print("No data found. Exiting."); exit()

    # 変更点: データセットを正規化
    normalized_data, stats = calculate_stats_and_normalize(my_data)

    all_texts = "".join([text for text, _ in normalized_data])
    unique_chars = sorted(list(set(all_texts)))
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    VOCAB = special_tokens + unique_chars
    char2id = {token: i for i, token in enumerate(VOCAB)}
    id2char = {i: token for i, token in enumerate(VOCAB)}
    PAD_ID = char2id['<pad>']
    print(f"Vocabulary size: {len(VOCAB)}")

    dataset = HandwritingDataset(normalized_data, char2id) # 変更点: 正規化済みデータを使用
    collate_with_pad = partial(collate_fn, text_pad_id=PAD_ID)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_pad, num_workers=4)

    model = HandwritingTransformer(vocab_size=len(VOCAB), n_mixtures=N_MIXTURES, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT)
    
    print("--- Start Training ---")
    trained_model = train_model(model, dataloader, epochs=EPOCHS, lr=LR, device=device)
    print("--- Finish Training ---")
    
    torch.save(trained_model.state_dict(), "handwriting_transformer.pth")

    test_text = "こんにちは"
    print(f"\nGenerating for text: '{test_text}'")
    
    # 変更点: statsを渡す
    generated_strokes = generate_strokes(trained_model, test_text, char2id, id2char, stats, device=device)
    
    if generated_strokes:
        plot_strokes(generated_strokes, title=f"Generated for '{test_text}'")
