import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import datetime
import json
import os
from functools import partial
import math

# ===================================================================
# 0. ハイパーパラメータ
# ===================================================================
N_MIXTURES = 20 # 混合ガウス分布の数

# ===================================================================
# 1. モデル定義 (HandwritingTransformer)
# ===================================================================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, n_mixtures, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.n_mixtures = n_mixtures

        # --- Encoder（テキスト側） ---
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(5000, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder（ストローク側） ---
        self.stroke_embedding = nn.Linear(3, d_model)
        self.pos_decoder = nn.Embedding(10000, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.stroke_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 出力ヘッド ---
        self.mdn_head = nn.Linear(d_model, self.n_mixtures * 6)
        self.fc_end = nn.Linear(d_model, 2)

    def forward(self, text_ids, stroke_seq, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, T_text = text_ids.shape
        _, T_stroke, _ = stroke_seq.shape
        device = text_ids.device

        # --- Encoder ---
        pos_text = torch.arange(0, T_text, device=device).unsqueeze(0).repeat(B, 1)
        text_emb = self.text_embedding(text_ids) + self.pos_encoder(pos_text)
        memory = self.text_encoder(text_emb, src_key_padding_mask=src_key_padding_mask)

        # --- Decoder ---
        pos_stroke = torch.arange(0, T_stroke, device=device).unsqueeze(0).repeat(B, 1)
        stroke_emb = self.stroke_embedding(stroke_seq) + self.pos_decoder(pos_stroke)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_stroke, device=device) # UserWarningの原因だが動作は正常
        
        out = self.stroke_decoder(
            stroke_emb, memory, tgt_mask=tgt_mask, 
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # --- 出力ヘッドの処理 ---
        mdn_params = self.mdn_head(out)
        pi_logits, mu_x, mu_y, sigma_x_log, sigma_y_log, rho_xy_tanh = mdn_params.split(self.n_mixtures, dim=-1)

        pi = torch.softmax(pi_logits, dim=-1)
        sigma_x = torch.exp(sigma_x_log)
        sigma_y = torch.exp(sigma_y_log)
        rho_xy = torch.tanh(rho_xy_tanh)

        end = self.fc_end(out)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end

# ===================================================================
# 2. 損失関数と学習ループ
# ===================================================================
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

# ===================================================================
# エラー修正点: compute_loss 関数のマスキング方法を修正
# ===================================================================
def compute_loss(mdn_params, end_pred, seq_gt, mask):
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = mdn_params
    dxdy_gt = seq_gt[:, :, :2]
    end_gt = seq_gt[:, :, 2].long()
    
    # --- MDN Loss ---
    # 修正前: mask_expanded = mask.unsqueeze(-1) を使っていたためIndexErrorが発生
    # 修正後: 2次元のboolマスク(B, T)を直接3次元のテンソル(B, T, D)に適用する。
    # これにより、有効なタイムステップのデータのみが正しく抽出される。
    pi_masked = pi[mask]          # (num_valid_tokens, N_MIXTURES)
    mu_x_masked = mu_x[mask]        # (num_valid_tokens, N_MIXTURES)
    mu_y_masked = mu_y[mask]        # (num_valid_tokens, N_MIXTURES)
    sigma_x_masked = sigma_x[mask]  # (num_valid_tokens, N_MIXTURES)
    sigma_y_masked = sigma_y[mask]  # (num_valid_tokens, N_MIXTURES)
    rho_xy_masked = rho_xy[mask]    # (num_valid_tokens, N_MIXTURES)
    dxdy_gt_masked = dxdy_gt[mask]  # (num_valid_tokens, 2)

    if pi_masked.nelement() > 0:
        loss_mdn = mdn_loss(
            pi_masked, mu_x_masked, mu_y_masked,
            sigma_x_masked, sigma_y_masked, rho_xy_masked,
            dxdy_gt_masked
        )
    else:
        loss_mdn = torch.tensor(0.0, device=pi.device)

    # --- End flag Loss (CrossEntropy) ---
    end_pred_masked = end_pred[mask]
    end_gt_masked = end_gt[mask]
    
    if end_pred_masked.nelement() > 0:
        loss_ce = nn.CrossEntropyLoss()(end_pred_masked, end_gt_masked)
    else:
        loss_ce = torch.tensor(0.0, device=end_pred.device)
    
    loss = loss_mdn + loss_ce
    return loss, loss_mdn.item(), loss_ce.item()


def train_model(model, dataloader, epochs=50, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mdn, total_ce = 0, 0, 0

        for text_ids, seq, text_padding_mask, seq_padding_mask in dataloader:
            text_ids, seq, text_padding_mask, seq_padding_mask = \
                text_ids.to(device), seq.to(device), text_padding_mask.to(device), seq_padding_mask.to(device)
            
            decoder_input_seq = seq[:, :-1]
            target_seq = seq[:, 1:]
            decoder_input_padding_mask = seq_padding_mask[:, :-1]
            target_mask = ~seq_padding_mask[:, 1:]

            if target_seq.shape[1] == 0: continue

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(
                text_ids, decoder_input_seq, 
                src_key_padding_mask=text_padding_mask, 
                tgt_key_padding_mask=decoder_input_padding_mask
            )
            mdn_params = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
            loss, mdn, ce = compute_loss(mdn_params, end_pred, target_seq, target_mask)

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item(); total_mdn += mdn; total_ce += ce

        avg_loss = total_loss / len(dataloader) if dataloader else 0
        avg_mdn = total_mdn / len(dataloader) if dataloader else 0
        avg_ce = total_ce / len(dataloader) if dataloader else 0
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} (MDN: {avg_mdn:.4f}, CE: {avg_ce:.4f})")
    return model

# ===================================================================
# 3. 生成と描画
# ===================================================================
def sample_from_mdn(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    cat_dist = Categorical(pi)
    mixture_idx = cat_dist.sample()

    mu_x_s, mu_y_s = mu_x[mixture_idx], mu_y[mixture_idx]
    sx, sy = sigma_x[mixture_idx], sigma_y[mixture_idx]
    rho = rho_xy[mixture_idx]
    
    mean = torch.tensor([mu_x_s, mu_y_s], device=pi.device)
    cov = torch.tensor([[sx*sx, rho*sx*sy],
                      [rho*sx*sy, sy*sy]], device=pi.device)
    
    mvn = MultivariateNormal(mean, cov)
    dxdy = mvn.sample()
    return dxdy[0].item(), dxdy[1].item()

def generate_strokes(model, text, char2id, id2char, max_len=700, device="cuda"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        text_ids = [char2id["<s>"]] + [char2id.get(c, char2id["<unk>"]) for c in text] + [char2id["</s>"]]
        text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        seq = torch.zeros((1, 1, 3), device=device)
        strokes, x, y = [], 0.0, 0.0

        for _ in range(max_len):
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, end_pred = model(text_ids, seq)
            
            pi_last = pi[0, -1, :]
            mu_x_last, mu_y_last = mu_x[0, -1, :], mu_y[0, -1, :]
            sigma_x_last, sigma_y_last = sigma_x[0, -1, :], sigma_y[0, -1, :]
            rho_xy_last = rho_xy[0, -1, :]
            end_logits_last = end_pred[0, -1, :]

            dx, dy = sample_from_mdn(pi_last, mu_x_last, mu_y_last, sigma_x_last, sigma_y_last, rho_xy_last)
            end = torch.argmax(end_logits_last).item()
            
            x += dx; y += dy
            strokes.append((x, y, end))
            
            if end == 1 and len(strokes) > 1: break # 1点だけでの終了を防ぐ
            
            new_point = torch.tensor([[[dx, dy, float(end)]]], device=device, dtype=torch.float32)
            seq = torch.cat([seq, new_point], dim=1)
    return strokes

# ... plot_strokes, データセット関連の関数は変更なし ...
def plot_strokes(strokes, title="Generated Handwriting"):
    plt.figure(figsize=(8, 8)); ax = plt.gca(); ax.set_facecolor('#f0f0f0')
    current_stroke_x, current_stroke_y = [], []
    for (x, y, end) in strokes:
        current_stroke_x.append(x); current_stroke_y.append(-y)
        if end == 1:
            if current_stroke_x: ax.plot(current_stroke_x, current_stroke_y, "k-", linewidth=2.5)
            current_stroke_x, current_stroke_y = [], []
    if current_stroke_x: ax.plot(current_stroke_x, current_stroke_y, "k-", linewidth=2.5)
    ax.set_aspect('equal', adjustable='box'); plt.title(title, fontsize=16)
    plt.xticks([]); plt.yticks([]); plt.grid(True, linestyle='--', alpha=0.6); plt.show()

def load_dataset_from_directory(directory_path):
    dataset = []
    if not os.path.exists(directory_path):
        print(f"Warning: Directory not found: {directory_path}"); return dataset
    print(f"Loading data from: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                text, strokes = data.get('text'), data.get('sequence')
                # ここでsequenceが(x,y,end)のリストであることを確認
                # もしくはstrokesが(x,y)のリストでendは0で補完されている場合も考慮
                if text is not None and strokes is not None:
                    dataset.append((text, [tuple(p) for p in strokes]))
            except Exception as e: print(f"Could not load/parse {filename}: {e}")
    print(f"Successfully loaded {len(dataset)} samples."); return dataset

class HandwritingDataset(Dataset):
    def __init__(self, data, char2id): self.data, self.char2id = data, char2id
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text, strokes = self.data[idx]
        text_ids = [self.char2id["<s>"]] + [self.char2id.get(c, self.char2id["<unk>"]) for c in text] + [self.char2id["</s>"]]
        stroke_seq = [(0.0, 0.0, 0.0)] + strokes
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(stroke_seq, dtype=torch.float32)

def collate_fn(batch, text_pad_id):
    texts, strokes = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=text_pad_id)
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    text_padding_mask = (padded_texts == text_pad_id)
    stroke_lengths = torch.tensor([len(s) for s in strokes])
    stroke_padding_mask = torch.arange(padded_strokes.size(1))[None, :] >= stroke_lengths[:, None]
    return padded_texts, padded_strokes, text_padding_mask, stroke_padding_mask

def save_strokes_to_json(strokes, text, directory="outputs"):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join([c for c in text if c.isalnum()]).rstrip() or "output"
    filepath = os.path.join(directory, f"{safe_text}_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"text": text, "strokes": strokes}, f, ensure_ascii=False, indent=2)
    print(f"Strokes saved to: {filepath}"); return filepath

def plot_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        plot_strokes([tuple(p) for p in data["strokes"]], title=f"Generated for '{data['text']}' (from file)")
    except Exception as e: print(f"An error occurred while plotting from file: {e}")

# ===================================================================
# 5. メイン実行ブロック
# ===================================================================
if __name__ == "__main__":
    D_MODEL, NHEAD, NUM_LAYERS = 256, 4, 6
    DROPOUT, EPOCHS, LR, BATCH_SIZE = 0.1, 300, 0.0005, 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")

    DATA_DIRECTORY = "dxdy_normalized_sentences" # 正規化済みデータフォルダ
    my_data = load_dataset_from_directory(DATA_DIRECTORY)
    if not my_data: print("No data found. Exiting."); exit()

    all_texts = "".join([text for text, _ in my_data])
    unique_chars = sorted(list(set(all_texts)))
    
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    VOCAB = special_tokens + unique_chars
    char2id = {token: i for i, token in enumerate(VOCAB)}
    id2char = {i: token for i, token in enumerate(VOCAB)}
    PAD_ID = char2id['<pad>']
    print(f"Vocabulary size: {len(VOCAB)}")

    dataset = HandwritingDataset(my_data, char2id)
    collate_with_pad = partial(collate_fn, text_pad_id=PAD_ID)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_pad)

    model = HandwritingTransformer(
        vocab_size=len(VOCAB), n_mixtures=N_MIXTURES, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT
    )
    
    print("--- Start Training ---")
    trained_model = train_model(model, dataloader, epochs=EPOCHS, lr=LR, device=device)
    print("--- Finish Training ---")

    test_text = "うい" if 'う' in char2id and 'い' in char2id else (unique_chars[0] if unique_chars else "a")
    print(f"\nGenerating for text: '{test_text}'")
    
    generated_strokes = generate_strokes(trained_model, test_text, char2id, id2char, device=device)
    
    if generated_strokes:
        saved_filepath = save_strokes_to_json(generated_strokes, test_text)
        plot_from_file(saved_filepath)
    else:
        print("No strokes were generated.")
