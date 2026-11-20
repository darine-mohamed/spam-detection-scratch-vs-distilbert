"""Model A - Transformer encoder implemented from scratch (no nn.MultiheadAttention,
no nn.TransformerEncoder, no pretrained weights). Identical to the Member 1 notebook."""
import json, math, re
import torch
import torch.nn as nn

MAX_LEN = 100


class ManualSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        b, n, _ = x.shape
        split = lambda t: t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, V).transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.output(context)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attention = ManualSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, padding_mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class ScratchSpamTransformer(nn.Module):
    def __init__(self, vocab_size, pad_id=0, max_len=100, d_model=128, num_heads=4,
                 num_layers=2, ff_dim=256, dropout=0.15, num_classes=2):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        b, n = input_ids.shape
        positions = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(b, -1)
        padding_mask = input_ids != self.pad_id
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        mask = padding_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


# ---- preprocessing: exactly what was used for training (clean_text -> tokenize -> encode) ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", str(text).lower())


def encode_text(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokenize_text(text)][:max_len]
    if not ids:
        ids = [vocab["[UNK]"]]
    return ids + [vocab["[PAD]"]] * (max_len - len(ids))


def load_scratch(weights_path, vocab_path, device="cpu"):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    model = ScratchSpamTransformer(len(vocab), pad_id=vocab["[PAD]"], max_len=MAX_LEN,
                                   d_model=128, num_heads=4, num_layers=2, ff_dim=256,
                                   dropout=0.15, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval(), vocab
