import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import random
import datetime 

# ===================================================================
# 1. モデル定義 (HandwritingTransformer)
# ===================================================================
class HandwritingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # --- Encoder（テキスト側） ---
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        # 位置エンコーディングは学習可能なパラメータとして定義
        self.pos_encoder = nn.Embedding(500, d_model)  # 最大テキスト長50

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder（ストローク側） ---
        self.stroke_embedding = nn.Linear(3, d_model)  # (dx, dy, end) → d_model
        # 位置エンコーディングは学習可能なパラメータとして定義
        self.pos_decoder = nn.Embedding(1000, d_model) # 最大ストローク長100

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.stroke_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 出力ヘッド ---
        self.fc_xy = nn.Linear(d_model, 2)   # Δx, Δy
        self.fc_end = nn.Linear(d_model, 2)  # end (0=続く, 1=終了)

    def forward(self, text_ids, stroke_seq, text_mask=None, stroke_mask=None, memory_mask=None):
        """
        text_ids: (B, T_text)
        stroke_seq: (B, T_stroke, 3) # 教師強制 (dx, dy, end)
        text_mask: (B, T_text) - パディング部分を無視するためのマスク
        stroke_mask: (T_stroke, T_stroke) - 未来の情報を隠すためのマスク
        memory_mask: (B, T_stroke, T_text) - パディング部分を無視するためのマスク
        """
        B, T_text = text_ids.shape
        _, T_stroke, _ = stroke_seq.shape
        device = text_ids.device

        # --- テキスト埋め込み & エンコード ---
        pos_text = torch.arange(0, T_text, device=device).unsqueeze(0).repeat(B, 1)
        text_emb = self.text_embedding(text_ids) + self.pos_encoder(pos_text)
        memory = self.text_encoder(text_emb, src_key_padding_mask=text_mask)

        # --- ストローク埋め込み ---
        pos_stroke = torch.arange(0, T_stroke, device=device).unsqueeze(0).repeat(B, 1)
        stroke_emb = self.stroke_embedding(stroke_seq) + self.pos_decoder(pos_stroke)

        # --- デコーダ ---
        # Decoder用の未来を見ないためのマスクを作成
        if stroke_mask is None:
             stroke_mask = nn.Transformer.generate_square_subsequent_mask(T_stroke).to(device)
        
        # Decoder用のパディングマスク
        # PyTorchのTransformerDecoderは (B, T_stroke, T_text) ではなく (B, T_text) を期待
        # メモリのパディングマスク
        memory_key_padding_mask = text_mask

        out = self.stroke_decoder(stroke_emb, memory, tgt_mask=stroke_mask, memory_key_padding_mask=memory_key_padding_mask)

        # --- 出力 ---
        dxdy = self.fc_xy(out)
        end = self.fc_end(out)

        return dxdy, end

# ===================================================================
# 2. 損失関数と学習ループ
# ===================================================================
def compute_loss(dxdy_pred, end_pred, seq_gt, mask):
    """
    dxdy_pred: (B, T, 2)
    end_pred:  (B, T, 2)
    seq_gt:    (B, T, 3) # [dx, dy, end]
    mask:      (B, T)    # パディング部分の損失を計算しないためのマスク
    """
    dxdy_gt = seq_gt[:, :, :2]
    end_gt = seq_gt[:, :, 2].long()

    # マスクを適用して有効な要素のみを選択
    mask_expanded = mask.unsqueeze(-1).expand_as(dxdy_pred)
    
    # Δx, Δy: MSE Loss
    mse_loss = nn.MSELoss(reduction='none')(dxdy_pred, dxdy_gt)
    mse_loss = (mse_loss * mask_expanded).sum() / mask_expanded.sum()

    # end: CrossEntropy
    # reshapeの前にマスクを適用
    end_pred_masked = end_pred[mask] # (num_valid_tokens, 2)
    end_gt_masked = end_gt[mask]     # (num_valid_tokens)
    
    # データがないエッジケースを回避
    if end_pred_masked.nelement() == 0:
        return torch.tensor(0.0, device=dxdy_pred.device), 0.0, 0.0

    ce_loss = nn.CrossEntropyLoss()(end_pred_masked, end_gt_masked)
    
    loss = mse_loss + ce_loss * 3.0
    return loss, mse_loss.item(), ce_loss.item()

def train_model(model, dataloader, epochs=50, lr=1e-4, device="cuda"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mse, total_ce = 0, 0, 0

        for text_ids, seq, text_mask, seq_mask in dataloader:
            text_ids = text_ids.to(device)
            seq = seq.to(device)
            text_mask = text_mask.to(device)
            seq_mask = seq_mask.to(device)

            # Decoderへの入力は最後の点を抜く
            decoder_input_seq = seq[:, :-1]
            # 教師データは最初の点を抜く
            target_seq = seq[:, 1:]
            target_mask = seq_mask[:, 1:]

            dxdy_pred, end_pred = model(text_ids, decoder_input_seq, text_mask=text_mask)

            loss, mse, ce = compute_loss(dxdy_pred, end_pred, target_seq, target_mask)

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

# ===================================================================
# 3. 生成と描画
# ===================================================================
def generate_strokes(model, text, char2id, id2char, max_len=100, device="cuda"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        text_ids = [char2id["<s>"]] + [char2id.get(c, char2id["<unk>"]) for c in text] + [char2id["</s>"]]
        text_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # <sos>トークンから生成を開始
        # [dx, dy, end_flag]
        seq = torch.zeros((1, 1, 3), device=device) 
        
        strokes = []
        x, y = 0.0, 0.0

        for _ in range(max_len):
            dxdy_pred, end_pred = model(text_ids, seq)
            
            # 最新のタイムステップの予測値を取得
            dxdy_last = dxdy_pred[:, -1, :]
            end_logits_last = end_pred[:, -1, :]

            # 予測から次の点をサンプリング
            dx, dy = dxdy_last[0].cpu().numpy()
            
            # endフラグは確率の高い方を採用
            end = torch.argmax(end_logits_last, dim=-1).item()

            # 座標を更新
            x += dx
            y += dy
            strokes.append((float(x), float(y), int(end)))
            
            if end == 1:
                break
            
            # 次の入力として現在の予測を追加
            new_point = torch.tensor([[[dx, dy, float(end)]]], dtype=torch.float32, device=device)
            seq = torch.cat([seq, new_point], dim=1)

    return strokes

def plot_strokes(strokes, title="Generated Handwriting"):
    plt.figure(figsize=(6, 6))
    xs, ys = [], []
    for (x, y, end) in strokes:
        xs.append(x)
        ys.append(-y) # y軸を反転すると自然に見える
        if end == 1:
            if xs: # 点が1つ以上あれば描画
                plt.plot(xs, ys, "k-")
            xs, ys = [], []
    if xs: # ループ終了後、最後のストロークが残っていれば描画
        plt.plot(xs, ys, "k-")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

import json
import os # osモジュールをインポート

def load_dataset_from_directory(directory_path):
    """
    指定されたディレクトリ内のすべてのJSONファイルを読み込み、
    (text, stroke_data)のリストを返す関数。
    """
    dataset = []
    
    print(f"Loading data from: {directory_path}")
    
    # ディレクトリ内の全ファイル名を取得
    for filename in os.listdir(directory_path):
        # .jsonで終わるファイルのみを対象とする
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            
            # UTF-8エンコーディングを指定してファイルを開く
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    text = data['text']
                    strokes = data['sequence']
                    
                    # (推奨) list of listsをlist of tuplesに変換
                    stroke_data_tuples = [tuple(point) for point in strokes]
                    
                    dataset.append((text, stroke_data_tuples))
                except Exception as e:
                    print(f"Could not load or parse {filename}: {e}")

    print(f"Successfully loaded {len(dataset)} samples.")
    return dataset

class HandwritingDataset(Dataset):
    def __init__(self, data, char2id):
        self.data = data
        self.char2id = char2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, strokes = self.data[idx]
        
        # テキストをIDに変換
        text_ids = [self.char2id["<s>"]] + [self.char2id.get(c, self.char2id["<unk>"]) for c in text] + [self.char2id["</s>"]]
        
        # ストロークの先頭に開始点(0,0,0)を追加
        stroke_seq = [(0.0, 0.0, 0.0)] + strokes

        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(stroke_seq, dtype=torch.float32)

def collate_fn(batch):
    """可変長のシーケンスをパディングしてバッチを作成"""
    texts, strokes = zip(*batch)
    
    # パディング
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_strokes = pad_sequence(strokes, batch_first=True, padding_value=0.0)
    
    # パディングマスクを作成 (Trueがパディング部分)
    text_mask = (padded_texts == 0)
    stroke_mask = (padded_strokes.sum(dim=-1) == 0)

    return padded_texts, padded_strokes, text_mask, stroke_mask


def save_strokes_to_json(strokes, text, directory="outputs"):
    """生成されたストロークとテキストをJSONファイルに保存する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # ファイル名にテキストとタイムスタンプを入れる
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{text}_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    
    # 保存するデータ構造
    data_to_save = {
        "text": text,
        "strokes": strokes # strokesは既に(x, y, end)のタプルのリスト
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"Strokes saved to: {filepath}")
    return filepath

def plot_from_file(filepath):
    """JSONファイルからストロークデータを読み込んで描画する"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data["text"]
            strokes = data["strokes"]
            
            # JSONではタプルがリストとして保存されるので、タプルに戻す
            strokes_as_tuples = [tuple(p) for p in strokes]
            
            print(f"Plotting from {filepath}...")
            plot_strokes(strokes_as_tuples, title=f"Generated for '{text}' (from file)")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")


# ===================================================================
# 5. メイン実行ブロック (修正版)
# ===================================================================
if __name__ == "__main__":
    # --- ハイパーパラメータ ---
    
    # 修正点: VOCABをトークンの「リスト」として定義
    special_tokens = ['<s>', '</s>', '<pad>', '<unk>']
    # 'a'から'z'までのアルファベットリストを生成
    import string
    alphabet = list(string.ascii_lowercase)
    
    VOCAB = special_tokens + alphabet

    char2id = {token: i for i, token in enumerate(VOCAB)}
    id2char = {i: token for i, token in enumerate(VOCAB)}
    VOCAB_SIZE = len(VOCAB)
    
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    EPOCHS = 1000
    LR = 0.001
    BATCH_SIZE = 8
    # --- データ準備 ---

    # --- デバイス設定 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. あなたのデータセットを読み込む
    # JSONファイルが格納されているフォルダのパスを指定してください
    # 例: "C:/Users/ryots/Desktop/my_handwriting_data"
    DATA_DIRECTORY = "processed_json"  # ここをあなたのデータセットのパスに変更してください
    my_data = load_dataset_from_directory(DATA_DIRECTORY)

    # 2. 語彙(VOCAB)をデータセットから自動生成する
    all_texts = "".join([text for text, strokes in my_data])
    unique_chars = sorted(list(set(all_texts)))
    
    special_tokens = ['<s>', '</s>', '<pad>', '<unk>']
    VOCAB = special_tokens + unique_chars
    
    char2id = {token: i for i, token in enumerate(VOCAB)}
    id2char = {i: token for i, token in enumerate(VOCAB)}
    VOCAB_SIZE = len(VOCAB)

    # 3. DataLoaderにデータを渡す
    dataset = HandwritingDataset(my_data, char2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # --- モデルの初期化と学習 ---
    model = HandwritingTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    print("--- Start Training ---")
    trained_model = train_model(model, dataloader, epochs=EPOCHS, lr=LR, device=device)
    print("--- Finish Training ---")

    # --- 生成と描画 ---
    test_text = "あ"
    print(f"\nGenerating for text: '{test_text}'")
    generated_strokes = generate_strokes(trained_model, test_text, char2id, id2char, device=device)
    
    # plot_strokes(generated_strokes, title=f"Generated for '{test_text}'")

    print("--- Finish Training ---")
    
    saved_filepath = save_strokes_to_json(generated_strokes, test_text,"outputs")

    if saved_filepath:
        plot_from_file(saved_filepath)