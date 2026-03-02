import os, math, torch
import torch.nn as nn
import torch.nn.functional as F

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# data
if os.path.exists("input.txt"):
    text = open("input.txt","r",encoding="utf-8").read() 
else: 
    text = "hello world\n"*500

chars = sorted(set(text)) 
V = len(chars)
stoi = {c:i for i,c in enumerate(chars)} 
itos = {i:c for c,i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9*len(data)); tr, va = data[:n], data[n:]

B,T = 64,64          # batch, context
E = 64               # embed dim (small)
steps, lr = 1500, 3e-4

def batch(split):
    src = tr if split=="tr" else va
    ix = torch.randint(0, len(src)-T-1, (B,))
    x = torch.stack([src[i:i+T] for i in ix])
    y = torch.stack([src[i+1:i+T+1] for i in ix])
    return x.to(device), y.to(device)

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = T  # ★ 追加: 位置埋め込み長を固定
        self.tok = nn.Embedding(V, E)
        self.pos = nn.Embedding(self.block_size, E)
        self.ln1 = nn.LayerNorm(E)
        self.qkv = nn.Linear(E, 3*E, bias=False)
        self.proj= nn.Linear(E, E, bias=False)
        self.ln2 = nn.LayerNorm(E)
        self.ff  = nn.Sequential(nn.Linear(E,4*E), nn.GELU(), nn.Linear(4*E,E))
        self.lnf = nn.LayerNorm(E)
        self.head= nn.Linear(E, V, bias=False)
        self.head.weight = self.tok.weight
        self.vocab_size = self.tok.num_embeddings  # = V
        self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))

    def attn(self, x):
        B_, T_, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                 # (B,T,C)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(C)          # (B,T,T)
        w = w.masked_fill(self.mask[:T_, :T_] == 0, float("-inf"))  # (B,T,T) のまま
        w = F.softmax(w, dim=-1)
        return self.proj(w @ v)                               # (B,T,C)

    def forward(self, idx, tgt=None):
        B_, T_ = idx.shape
        if T_ > self.block_size:          # ★ 追加: 必ず crop
            idx = idx[:, -self.block_size:]
            T_ = self.block_size

        pos = torch.arange(T_, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        logits = self.head(self.lnf(x))
        loss = None if tgt is None else F.cross_entropy(logits.view(-1,V), tgt.view(-1))
        return logits, loss

    @torch.no_grad()
    def gen(self, idx, n, temp=1.0, topk=30):
        for _ in range(n):
            idxc = idx[:, -self.block_size:]
            logits, _ = self(idxc)              # (B,T,V)
            logits = logits[:, -1, :] / max(temp, 1e-8)   # (B,V)

            if topk:
                k = min(topk, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)   # (B,V)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

m = TinyGPT().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=lr)

for s in range(1, steps+1):
    x,y = batch("tr")
    _,loss = m(x,y)
    opt.zero_grad(set_to_none=True); 
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if s % 200 == 0:
        with torch.no_grad():
            vx,vy = batch("va")
            _,vl = m(vx,vy)
        print(f"step {s:4d} train {loss.item():.3f} val {vl.item():.3f}")

start = torch.zeros((1,1), dtype=torch.long, device=device)
out = m.gen(start, 300, temp=0.9, topk=30)[0].tolist()
print("".join(itos[i] for i in out))