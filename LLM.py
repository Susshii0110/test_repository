import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =====================
# 設定
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 5000
seq_len = 32
embed_dim = 256
num_heads = 4
num_layers = 4
batch_size = 16
lr = 3e-4
epochs = 5

# =====================
# ダミーデータセット
# =====================
class RandomTextDataset(Dataset):
    def __init__(self, size=10000):
        self.data = torch.randint(0, vocab_size, (size, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #[1,2,3,4]->[2,3,4,5]を予測するため
        x = self.data[idx, :-1] #入力
        y = self.data[idx, 1:]  #正解
        return x, y

# =====================
# Transformer LLM
# =====================
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        #B: batch size, T: シーケンス長
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.encoder(x)
        logits = self.lm_head(x)
        return logits

# =====================
# 学習準備
# =====================
dataset = RandomTextDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TransformerLM().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# =====================
# 学習ループ
# =====================
model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

print("Training finished.")