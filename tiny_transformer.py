# mini_char_gpt.py
# A tiny character-level decoder-only Transformer (mini GPT) in a single file.

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# device
# -----------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()

# -----------------------------
# hyperparameters
# -----------------------------
batch_size   = 64
block_size   = 64     # context length (how many previous chars the model can look at)
max_steps    = 3000
eval_interval= 300
eval_iters   = 200
learning_rate= 3e-4

n_embed = 128         # embedding dimension
n_head  = 4           # number of attention heads
n_layer = 4           # number of Transformer blocks
dropout = 0.1

torch.manual_seed(1337)

# -----------------------------
# data
# -----------------------------
if os.path.exists("input.txt"):
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
else:
    text = ("To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n") * 50

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])         # (B, T)
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])     # (B, T)  next-token targets
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------
# model: decoder-only Transformer
# -----------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no looking into the future)."""
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask (T,T): allow attending only to current/past positions
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)  # not a parameter

    def forward(self, x):
        B, T, C = x.shape  # (batch, time, channels)

        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # attention scores: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # apply causal mask: mask out future positions with -inf
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # weighted sum -> (B, nh, T, hs)
        y = att @ v

        # re-assemble: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Pre-LN Transformer block."""
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniCharGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # weight tying (optional but common)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = idx.shape[1]

        pos = torch.arange(0, T, device=idx.device)  # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)    # (B,T,C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                     # (B,T,vocab)

        loss = None
        if targets is not None:
            # flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(B*T, vocab_size),
                targets.view(B*T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # last step

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B,1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# -----------------------------
# train
# -----------------------------
model = MiniCharGPT(vocab_size, block_size, n_embed, n_head, n_layer, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"device={device}, vocab_size={vocab_size}, params={sum(p.numel() for p in model.parameters())}")

for step in range(1, max_steps + 1):
    if step % eval_interval == 0 or step == 1:
        losses = estimate_loss(model)
        print(f"step {step:4d}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# -----------------------------
# sample
# -----------------------------
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token = id 0 (arbitrary)
out = model.generate(start, max_new_tokens=400, temperature=0.9, top_k=50)[0].tolist()
print("\n--- generated ---")
print(decode(out))
