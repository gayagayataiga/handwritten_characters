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

# ===================================================================
# 1. モデル定義
# ===================================================================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, n_mixtures, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.n_mixtures, self.d_model = n_mixtures, d_model
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(5000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.stroke_embedding = nn.Linear(3, d_model)
        self.pos_decoder = nn.Embedding(10000, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu')
        self.stroke_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.mdn_head = nn.Linear(d_model, self.n_mixtures * 6)
        self.fc_end = nn.Linear(d_model, 2)

    def forward(self, text_ids, stroke_seq, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, T_text = text_ids.shape; _, T_stroke, _ = stroke_seq.shape; device = text_ids.device
        
        # ===============================================================
        # 修正点: torch.arange の device 引数をキーワード引数に変更
        # ===============================================================
        pos_text = torch.arange(0, T_text, device=device).unsqueeze(0).repeat(B, 1)
        # ===============================================================
        
        text_emb = self.text_embedding(text_ids) * math.sqrt(self.d_model) + self.pos_encoder(pos_text)
        memory = self.text_encoder(text_emb, src_key_padding_mask=src_key_padding_mask)
        
        # ===============================================================
        # 修正点: torch.arange の device 引数をキーワード引数に変更
        # ===============================================================
        pos_stroke = torch.arange(0, T_stroke, device=device).unsqueeze(0).repeat(B, 1)
        # ===============================================================
        
        stroke_emb = self.stroke_embedding(stroke_seq) * math.sqrt(self.d_model) + self.pos_decoder(pos_stroke)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_stroke, device=device)
        out = self.stroke_decoder(stroke_emb, memory, tgt_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        mdn_params_raw = self.mdn_head(out)
        pi_logits, mu_x, mu_y, sigma_x_log, sigma_y_log, rho_xy_tanh = mdn_params_raw.split(self.n_mixtures, dim=-1)
        sigma_x, sigma_y, rho_xy = torch.exp(sigma_x_log), torch.exp(sigma_y_log), torch.tanh(rho_xy_tanh)
        end = self.fc_end(out)
        return pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end

# ===================================================================
# 2. 損失関数と学習ループ
# ===================================================================
def mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, target_dxdy):
    target_dx, target_dy = target_dxdy[:, 0].unsqueeze(-1), target_dxdy[:, 1].unsqueeze(-1)
    norm_x, norm_y = (target_dx - mu_x) / sigma_x, (target_dy - mu_y) / sigma_y
    z = norm_x**2 + norm_y**2 - 2 * rho_xy * norm_x * norm_y
    one_minus_rho_sq = torch.clamp(1 - rho_xy**2, min=1e-5)
    log_pdf = -z / (2 * one_minus_rho_sq) - torch.log(2 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho_sq))
    log_pi = torch.log(pi + 1e-10)
    log_likelihood = torch.logsumexp(log_pi + log_pdf, dim=-1)
    return -log_likelihood.mean()

def compute_loss(mdn_params, end_pred, seq_gt, mask):
    pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = mdn_params
    pi = torch.softmax(pi_logits, dim=-1)
    dxdy_gt, end_gt = seq_gt[:, :, :2], seq_gt[:, :, 2].long()
    pi_masked, mu_x_masked, mu_y_masked = pi[mask], mu_x[mask], mu_y[mask]
    sigma_x_masked, sigma_y_masked, rho_xy_masked = sigma_x[mask], sigma_y[mask], rho_xy[mask]
    dxdy_gt_masked = dxdy_gt[mask]
    loss_mdn = mdn_loss(pi_masked, mu_x_masked, mu_y_masked, sigma_x_masked, sigma_y_masked, rho_xy_masked, dxdy_gt_masked) if pi_masked.nelement() > 0 else torch.tensor(0.0, device=pi.device)
    end_pred_masked, end_gt_masked = end_pred[mask], end_gt[mask]
    loss_ce = nn.CrossEntropyLoss()(end_pred_masked, end_gt_masked) if end_pred_masked.nelement() > 0 else torch.tensor(0.0, device=end_pred.device)
    return loss_mdn + loss_ce, loss_mdn.item(), loss_ce.item()

def train_model(model, dataloader, epochs, lr, device, warmup_steps):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer, lambda step: min((step + 1)**-0.5, (step + 1) * warmup_steps**-1.5))
    for epoch in range(1, epochs + 1):
        model.train(); total_loss, total_mdn, total_ce = 0, 0, 0
        for text_ids, seq, text_padding_mask, seq_padding_mask in dataloader:
            text_ids, seq, text_padding_mask, seq_padding_mask = text_ids.to(device), seq.to(device), text_padding_mask.to(device), seq_padding_mask.to(device)
            decoder_input_seq, target_seq = seq[:, :-1], seq[:, 1:]
            decoder_input_padding_mask, target_mask = seq_padding_mask[:, :-1], ~seq_padding_mask[:, 1:]
            if target_seq.shape[1] == 0: continue
            pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(text_ids, decoder_input_seq, text_padding_mask, decoder_input_padding_mask)
            mdn_params = (pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
            loss, mdn, ce = compute_loss(mdn_params, end_pred, target_seq, target_mask)
            if torch.isfinite(loss):
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); lr_scheduler.step()
                total_loss += loss.item(); total_mdn += mdn; total_ce += ce
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_mdn = total_mdn / len(dataloader) if len(dataloader) > 0 else 0
        avg_ce = total_ce / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} (MDN: {avg_mdn:.4f}, CE: {avg_ce:.4f}) LR: {lr_scheduler.get_last_lr()[0]:.6f}")
    return model

# ===================================================================
# 3. 生成と描画
# ===================================================================
def sample_from_mdn(pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature=0.8):
    pi = torch.softmax(pi_logits / temperature, dim=-1)
    if not torch.all(torch.isfinite(pi)):
        pi = torch.ones_like(pi) * (1.0 / len(pi))
    cat_dist = Categorical(pi)
    mixture_idx = cat_dist.sample()
    mu_x_s, mu_y_s = mu_x[mixture_idx], mu_y[mixture_idx]
    sx, sy = sigma_x[mixture_idx] * math.sqrt(temperature), sigma_y[mixture_idx] * math.sqrt(temperature)
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
            pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(text_ids, seq)
            pi_logits_last = pi_logits[0, -1, :]
            mu_x_last, mu_y_last = mu_x[0, -1, :], mu_y[0, -1, :]
            sigma_x_last, sigma_y_last, rho_xy_last = sigma_x[0, -1, :], sigma_y[0, -1, :], rho_xy[0, -1, :]
            end_logits_last = end_pred[0, -1, :]
            dx_norm, dy_norm = sample_from_mdn(pi_logits_last, mu_x_last, mu_y_last, sigma_x_last, sigma_y_last, rho_xy_last, temperature)
            end = torch.argmax(end_logits_last).item()
            dx, dy = dx_norm * std_dx + mean_dx, dy_norm * std_dy + mean_dy
            x += dx; y += dy
            strokes.append((x, y, end))
            if end == 1 and len(strokes) > 5: break
            next_point = torch.tensor([[[dx_norm, dy_norm, float(end)]]], device=device, dtype=torch.float32)
            seq = torch.cat([seq, next_point], dim=1)
    return strokes

# ===================================================================
# 4. データセット関連
# ===================================================================
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

def calculate_stats_and_normalize(dataset):
    all_dx, all_dy = [], []
    for _, strokes in dataset:
        for dx, dy, _ in strokes:
            all_dx.append(dx); all_dy.append(dy)
    all_dx, all_dy = np.array(all_dx), np.array(all_dy)
    stats = {'mean_dx': np.mean(all_dx), 'std_dx': np.std(all_dx), 'mean_dy': np.mean(all_dy), 'std_dy': np.std(all_dy)}
    normalized_dataset = [ (text, [((dx - stats['mean_dx']) / stats['std_dx'], (dy - stats['mean_dy']) / stats['std_dy'], end) for dx, dy, end in strokes]) for text, strokes in dataset ]
    print(f"Data normalized. Stats: {stats}"); return normalized_dataset, stats

class HandwritingDataset(Dataset):
    def __init__(self, data, char2id): self.data, self.char2id = data, char2id
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text, strokes = self.data[idx]
        text_ids = [self.char2id["<s>"]] + [self.char2id.get(c, self.char2id["<unk>"]) for c in text] + [self.char2id["</s>"]]
        stroke_seq = [(0.0, 0.0, 0.0)] + strokes
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(stroke_seq, dtype=torch.float32)

def load_dataset_from_directory(directory_path):
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
    texts, strokes = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=text_pad_id)
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    text_padding_mask = (padded_texts == text_pad_id)
    stroke_lengths = torch.tensor([len(s) for s in strokes])
    stroke_padding_mask = torch.arange(padded_strokes.size(1))[None, :] >= stroke_lengths[:, None]
    return padded_texts, padded_strokes, text_padding_mask, stroke_padding_mask

# ===================================================================
# 5. メイン実行ブロック
# ===================================================================
if __name__ == "__main__":
    DEBUG_MODE = False

    N_MIXTURES = 20
    D_MODEL, NHEAD = 256, 4
    NUM_LAYERS, DIM_FEEDFORWARD = 4, 1024
    DROPOUT, EPOCHS, LR = 0.1, 300, 1.0
    BATCH_SIZE = 16 if not DEBUG_MODE else 8
    WARMUP_STEPS = 4000 if not DEBUG_MODE else 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    DATA_DIRECTORY = "processed_json"
    my_data = load_dataset_from_directory(DATA_DIRECTORY)

    if DEBUG_MODE:
        print("="*40 + f"\nRUNNING IN DEBUG MODE (WARMUP_STEPS={WARMUP_STEPS})\n" + "="*40)
        my_data = my_data[:20]

    if not my_data: print("No data found. Exiting."); exit()
    normalized_data, stats = calculate_stats_and_normalize(my_data)
    all_texts = "".join([text for text, _ in normalized_data])
    unique_chars = sorted(list(set(all_texts)))
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    VOCAB = special_tokens + unique_chars
    char2id, id2char = {t: i for i, t in enumerate(VOCAB)}, {i: t for i, t in enumerate(VOCAB)}
    PAD_ID = char2id['<pad>']
    print(f"Vocabulary size: {len(VOCAB)}")

    dataset = HandwritingDataset(normalized_data, char2id)
    collate_with_pad = partial(collate_fn, text_pad_id=PAD_ID)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_pad, num_workers=0)

    model = HandwritingTransformer(vocab_size=len(VOCAB), n_mixtures=N_MIXTURES, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT)
    
    model_path = "gemi.pth"
    if os.path.exists(model_path) and not DEBUG_MODE:
        print(f"Loading pre-trained model from {model_path}")
        # モデル構造が変更された可能性があるため、古いモデルはロードしない方が安全
        # 代わりに、学習を続ける場合は手動で古いモデルを削除する運用を推奨
        # model.load_state_dict(torch.load(model_path, map_location=device))

    print("--- Start Training ---")
    trained_model = train_model(model, dataloader, EPOCHS, LR, device, WARMUP_STEPS)
    print("--- Finish Training ---")
    
    if not DEBUG_MODE:
        torch.save(trained_model.state_dict(), model_path)

    test_text = "こんにちは"
    print(f"\nGenerating for text: '{test_text}'")
    
    generated_strokes = generate_strokes(trained_model, test_text, char2id, id2char, stats, device=device)
    
    if generated_strokes:
        plot_strokes(generated_strokes, title=f"Generated for '{test_text}'")

