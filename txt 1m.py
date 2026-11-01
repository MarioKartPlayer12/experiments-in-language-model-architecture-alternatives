 
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

 
WIKI_TITLES = [
    "Wolfgang Amadeus Mozart", "Joseph Haydn", "Ludwig van Beethoven",
    "Johann Sebastian Bach", "Antonio Salieri", "Franz Joseph Haydn",
    "Classical period (music)", "Symphony", "Opera", "Requiem (Mozart)",
    "The Magic Flute", "Don Giovanni", "Eine kleine Nachtmusik",
    "Piano sonata", "String quartet",
]


def fetch_wikitext(title: str) -> str:
    url = (f"https://en.wikipedia.org/w/api.php?action=parse"
           f"&page={title.replace(' ', '_')}&format=json&prop=wikitext")
    req = urllib.request.Request(url, headers={"User-Agent": "KRE-Explorer/1.0"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["parse"]["wikitext"]["*"]


def clean_wikitext(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\|[^}]*\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\[\[(File|Image|Category):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"^=+\s*[^=]+\s*=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def build_corpus(path: str = "wikids.txt") -> str:
    if Path(path).exists():
        return Path(path).read_text()
    parts = []
    for title in WIKI_TITLES:
        try:
            raw = fetch_wikitext(title)
            parts.append(clean_wikitext(raw))
        except Exception as e:
            print(f"[data] skipped {title}: {e}")
    text = "\n\n".join(parts)
    Path(path).write_text(text)
    return text


def load_tokenizer(path: str = "wikiDS_bpe.json") -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_corpus(tokenizer: Tokenizer, text: str) -> np.ndarray:
    return np.array(tokenizer.encode(text).ids, dtype=np.int64)

 
class SoftSegCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(hidden,    hidden, kernel_size=3, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(hidden,    hidden, kernel_size=3, padding=8, dilation=4)
        self.conv4 = nn.Conv1d(hidden,    hidden, kernel_size=3, padding=16, dilation=8)
        self.boundary_head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.feature_head  = nn.Conv1d(hidden, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = self.emb(x).transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        b = torch.sigmoid(self.boundary_head(h).squeeze(1))
        f = self.feature_head(h).transpose(1, 2)
        return b, f

 
def soft_unit_formation(b, f, max_unit_len=8, min_unit_len=1, max_starts_per_pos=None):
    B, T, D = f.shape
    if max_starts_per_pos is None:
        max_starts_per_pos = T // 4

    stay = (1.0 - b).clamp(min=1e-3)
    log_stay = torch.log(stay)
    cum_log_stay = torch.cumsum(log_stay, dim=1)
    pad = torch.zeros(B, 1, device=b.device, dtype=cum_log_stay.dtype)
    cum_padded = torch.cat([pad, cum_log_stay], dim=1)

    _, start_indices = torch.topk(-b, k=max_starts_per_pos, dim=1)
    start_indices, _ = torch.sort(start_indices, dim=1)
    N_starts = start_indices.size(1)

    starts_cum = cum_padded.gather(1, start_indices).unsqueeze(2)
    cum_exp = cum_log_stay.unsqueeze(1)
    log_alpha = cum_exp - starts_cum

    end_idx = torch.arange(T, device=b.device).unsqueeze(0)
    start_idx = start_indices.unsqueeze(2)
    length = end_idx.unsqueeze(0) - start_idx
    mask = ((length >= min_unit_len) & (length <= max_unit_len)).float()
    log_alpha = log_alpha + torch.log(mask + 1e-12)
    log_alpha = log_alpha - log_alpha.max(dim=2, keepdim=True).values.detach()
    alpha = torch.exp(log_alpha) * mask
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)

    
    end_positions = torch.arange(T, device=b.device).view(1, 1, T).float()   
    expected_end = (alpha * end_positions).sum(dim=2)  

    units = torch.einsum("bnt,btd->bnd", alpha, f)
    boundary_rate = b.mean()
    return units, alpha, expected_end, boundary_rate

 
class RelativePositionWeightMLP(nn.Module):
    def __init__(self, embed_dim: int = 128, pos_dim: int = 32, hidden: int = 64):
        super().__init__()        
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
        )
        
        self.rel_encoder = nn.Sequential(
            nn.Linear(1, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
        )
       
        self.weight_mlp = nn.Sequential(
            nn.Linear(embed_dim + 2 * pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, units, unit_positions, current_positions):
        
        B, N, D = units.shape
        T = current_positions.size(1)

        
        unit_pos_enc = self.pos_encoder(unit_positions.unsqueeze(-1))  
        
        rel_dist = current_positions.unsqueeze(2) - unit_positions.unsqueeze(1)  

        
        rel_dist_enc = self.rel_encoder(rel_dist.unsqueeze(-1))  

        
        units_exp = units.unsqueeze(1).expand(B, T, N, D)  
        unit_pos_exp = unit_pos_enc.unsqueeze(1).expand(B, T, N, -1)  

       
        mlp_input = torch.cat([units_exp, unit_pos_exp, rel_dist_enc], dim=-1)  
        w = self.weight_mlp(mlp_input).squeeze(-1)  

        return w

 
class KREForm1BPERelPos(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 128,
                 pos_dim: int = 32, max_unit_len: int = 8, min_unit_len: int = 1,
                 boundary_target: float = 0.2):
        super().__init__()
        self.seg = SoftSegCNN(vocab_size, embed_dim=embed_dim, hidden=hidden)
        self.weight_mlp = RelativePositionWeightMLP(
            embed_dim=embed_dim, pos_dim=pos_dim, hidden=64,
        )
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_unit_len = max_unit_len
        self.min_unit_len = min_unit_len
        self.boundary_target = boundary_target

    def forward(self, x):
        b, f = self.seg(x)
        units, alpha, unit_positions, br = soft_unit_formation(
            b, f, self.max_unit_len, self.min_unit_len,
            max_starts_per_pos=x.size(1) // 4,
        )
        B, T = x.size()
        current_positions = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, T)

       
        w = self.weight_mlp(units, unit_positions, current_positions)  

       
        c = torch.einsum("btn,bnd->btd", w, units)  

       
        logits = self.head(c) 
        return logits, w, b, br
 
def get_batch(ids, seq_len, batch_size):
    n = len(ids) - seq_len - 1
    starts = np.random.randint(0, n, size=batch_size)
    x = np.stack([ids[s:s + seq_len] for s in starts])
    y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def train(steps=12000, seq_len=256, batch_size=8, lr=2e-4,
          boundary_reg=0.5, ckpt_path="kre_form1_bpe_relpos.pt",
          tokenizer_path="wikiDS_bpe.json", eval_every=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = build_corpus()
    tokenizer = load_tokenizer(tokenizer_path)
    ids = encode_corpus(tokenizer, text)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] corpus={len(text):,} chars  tokens={len(ids):,}  vocab={vocab_size}")

    model = KREForm1BPERelPos(vocab_size=vocab_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] device={device}  params={n_params:,}")

    losses, val_ppls = [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        x_np, y_np = get_batch(ids, seq_len, batch_size)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        logits, w, b, br = model(x)
        if torch.isnan(logits).any():
            print(f"[warn] NaN at step {step}, skipping")
            optim.zero_grad()
            continue

        ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        b_reg = (br - model.boundary_target) ** 2
        loss = ce + boundary_reg * b_reg

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        losses.append((ce.item(), b_reg.item(), br.item()))

        if step % 200 == 0 or step == 1:
            ce_avg = sum(L[0] for L in losses[-200:]) / min(200, len(losses))
            br_avg = sum(L[2] for L in losses[-200:]) / min(200, len(losses))
            print(f"  step {step:5d}/{steps}  ce={ce_avg:.4f}  br={br_avg:.3f}  ({time.time()-t0:.1f}s)")

        if step % eval_every == 0 or step == steps:
            ppl = evaluate(model, ids, vocab_size, device, seq_len, n_batches=20)
            val_ppls.append((step, ppl))
            print(f"  [eval] step {step}  val_perplexity={ppl:.2f}")
            model.train()

    save_model(model, tokenizer, ckpt_path, losses, val_ppls)
    plot_training(losses, val_ppls)
    return model, tokenizer


@torch.no_grad()
def evaluate(model, ids, vocab_size, device, seq_len, n_batches=20, batch_size=8):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x_np, y_np = get_batch(ids, seq_len, batch_size)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        logits, _, _, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(ce.item())
    return math.exp(sum(losses) / len(losses))


 
def save_model(model, tokenizer, path, losses, val_ppls):
    tok_path = Path(path).with_suffix(".tokenizer.json")
    tokenizer.save(str(tok_path))
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.get_vocab_size(),
            "embed_dim": model.seg.emb.embedding_dim,
            "hidden": model.seg.conv1.out_channels,
            "pos_dim": model.weight_mlp.pos_encoder[0].out_features,
            "max_unit_len": model.max_unit_len,
            "min_unit_len": model.min_unit_len,
            "boundary_target": model.boundary_target,
        },
        "losses": losses,
        "val_ppls": val_ppls,
        "tokenizer_path": str(tok_path),
    }
    torch.save(ckpt, path)
    print(f"[save] wrote {path} and {tok_path}")


def load_model(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tokenizer = Tokenizer.from_file(ckpt["tokenizer_path"])
    cfg = ckpt["config"]
    model = KREForm1BPERelPos(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[load] {path} on {device}  vocab={cfg['vocab_size']}  params={sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer
 
@torch.no_grad()
def sample(model, tokenizer, seed: str, length: int = 200, temperature: float = 1.0,
           device: str | None = None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = tokenizer.encode(seed).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(length):
        ctx = x[:, -256:]
        logits, _, _, _ = model(ctx)
        last = logits[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(last, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tokenizer.decode(x[0].tolist())

 

def plot_training(losses, val_ppls):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot([L[0] for L in losses])
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("step")
    axes[1].plot([L[2] for L in losses], color="orange")
    axes[1].axhline(0.2, color="k", ls=":", lw=0.7)
    axes[1].set_title("Boundary rate")
    axes[1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig("kre_bpe_relpos_training.png", dpi=120)
    plt.close()
    if val_ppls:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([s for s, _ in val_ppls], [p for _, p in val_ppls], marker="o")
        ax.set_title("Validation perplexity")
        ax.set_xlabel("step")
        plt.tight_layout()
        plt.savefig("kre_bpe_relpos_perplexity.png", dpi=120)
        plt.close()
    print("[plot] saved kre_bpe_relpos_training.png and kre_bpe_relpos_perplexity.png")


@torch.no_grad()
def plot_weight_heatmap(model, tokenizer, text: str, out_path: str = "kre_bpe_relpos_weights.png"):
    """Visualise the per-(unit, position) weight matrix for a given text."""
    model.eval()
    ids = tokenizer.encode(text).ids[:256]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    b, f = model.seg(x)
    units, alpha, unit_positions, br = soft_unit_formation(
        b, f, model.max_unit_len, model.min_unit_len,
        max_starts_per_pos=x.size(1) // 4,
    )
    B, T = x.size()
    current_positions = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    w = model.weight_mlp(units, unit_positions, current_positions)  
    w = w.squeeze(0).cpu().numpy()  

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    vmax = float(np.abs(w).max()) or 1.0
    axes[0].imshow(w.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel("current position (where we're predicting)")
    axes[0].set_ylabel("unit index")
    axes[0].set_title("Per-(unit, position) weights — each column is what the model\n"
                       "attends to when predicting at that position")

    up = unit_positions.squeeze(0).cpu().numpy()
    axes[1].scatter(up, np.arange(len(up)), s=4, color="orange")
    axes[1].set_xlabel("unit position in sequence")
    axes[1].set_ylabel("unit index")
    axes[1].set_title("Where each unit is located in the sequence")
    axes[1].set_xlim(0, T)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[plot] saved {out_path}")


if __name__ == "__main__":
    model, tokenizer = train(steps=50000, seq_len=256, batch_size=8, lr=3e-4, boundary_reg=0.5)
    for seed in ["Mozart was born in ", "The symphony was composed in ", "Beethoven met Mozart in "]:
        print(f"\n[seed] {seed!r}")
        print(sample(model, tokenizer, seed, length=200, temperature=0.8))
    text = build_corpus()[:2000]
    plot_weight_heatmap(model, tokenizer, text)