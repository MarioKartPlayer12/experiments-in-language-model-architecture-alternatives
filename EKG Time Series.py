 
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

 
_WAVE_CENTERS = np.array([15, 30, 40, 50, 70], dtype=np.float32)
_WAVE_AMPS    = np.array([0.15, -0.20, 1.00, -0.30, 0.35], dtype=np.float32)
_WAVE_SIGMAS  = np.array([4.0, 2.0, 3.0, 2.0, 5.0], dtype=np.float32)


def _single_beat(rng: np.random.RandomState) -> np.ndarray:
    """One heartbeat as a 100-sample 1D array, with mild per-beat jitter."""
    t = np.arange(100, dtype=np.float32)
  
    centers = _WAVE_CENTERS + rng.normal(0, 1.0, 5).astype(np.float32)
    amps    = _WAVE_AMPS    * (1.0 + rng.normal(0, 0.08, 5).astype(np.float32))
    sigmas  = _WAVE_SIGMAS  * (1.0 + rng.normal(0, 0.08, 5).astype(np.float32))
    wave = np.zeros(100, dtype=np.float32)
    for c, a, s in zip(centers, amps, sigmas):
        wave += a * np.exp(-0.5 * ((t - c) / s) ** 2)
    return wave


def _apply_anomaly(beat: np.ndarray, kind: str, rng: np.random.RandomState) -> np.ndarray:
    """Mutate a beat to simulate an arrhythmia."""
    if kind == "noisy":
        return beat * (1.5 + rng.random() * 0.5) + rng.normal(0, 0.05, 100).astype(np.float32)
    if kind == "inverted_t":
        t = np.arange(100, dtype=np.float32)
        t_mask = np.exp(-0.5 * ((t - 70.0) / 5.0) ** 2)
        return beat - 2.0 * _WAVE_AMPS[4] * t_mask
    if kind == "low_amplitude":
        return beat * 0.4
    return beat 

def make_window(n_beats: int = 16, anomaly_rate: float = 0.05, seed: int | None = None):
    """Return (window[1600], target[100]) for a synthetic EKG segment."""
    rng = np.random.RandomState(seed)
    beats = []
    for i in range(n_beats + 1):  
        beat = _single_beat(rng)
        if rng.random() < anomaly_rate:
            kind = rng.choice(["noisy", "inverted_t", "low_amplitude"])
            beat = _apply_anomaly(beat, kind, rng)
        beats.append(beat)
    window = np.concatenate(beats[:n_beats]).astype(np.float32)   
    target = beats[n_beats].astype(np.float32)                   
    return window, target

def visualize_ekg_data(n_windows: int = 3, n_beats: int = 16, seed: int = 42):
    """
    Plot a handful of synthetic EKG windows, with the target beat highlighted
    and a low-resolution 'unit boundary' overlay based on the nominal beat length.
    """
    rng = np.random.RandomState(seed)
    nominal_beat_len = 100   

    fig, axes = plt.subplots(n_windows, 1, figsize=(12, 2.4 * n_windows), sharex=False)
    if n_windows == 1:
        axes = [axes]

    for ax_i in range(n_windows):
        win, target = make_window(n_beats=n_beats, anomaly_rate=0.10, seed=seed + ax_i)
        t = np.arange(len(win) + len(target))
        full = np.concatenate([win, target])

        
        ax = axes[ax_i]
        ax.plot(t, full, color="C0", lw=0.9, label="input + target")
        ax.axvline(n_beats * nominal_beat_len, color="k", ls=":", alpha=0.6,
                   label="input → target boundary")

 
        ax.plot(t[n_beats * nominal_beat_len:], full[n_beats * nominal_beat_len:],
                color="green", lw=2.2, label="target beat")

        for k in range(1, n_beats + 1):
            ax.axvline(k * nominal_beat_len, color="red", ls="--", lw=0.6, alpha=0.35)
        ax.axvline(0, color="red", ls="--", lw=0.6, alpha=0.35, label="nominal beat boundaries")

        ax.set_title(f"Window {ax_i + 1}  (anomaly_rate=0.10)")
        ax.set_xlabel("sample index")
        ax.set_ylabel("amplitude")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("ekg_data_preview.png", dpi=120)
    plt.close()
    print("[ekg] saved ekg_data_preview.png")

 
 
class UnitDiscoveryCNN(nn.Module):
    def __init__(self, n_units: int = 16, embed_dim: int = 64):
        super().__init__()
        self.n_units = n_units
        self.embed_dim = embed_dim
         
        self.conv1 = nn.Conv1d(1,  32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=25)
        self.proj  = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = torch.relu(self.conv4(h))           
        h = h.transpose(1, 2)                    
        h = self.proj(h)                        
        return h
 
class UnitWeightingHead(nn.Module):
    def __init__(self, embed_dim: int = 64, n_units: int = 16, horizon: int = 100):
        super().__init__()
       
        self.weight_mlp = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
         
        self.head = nn.Linear(embed_dim, horizon)

    def forward(self, units: torch.Tensor) -> torch.Tensor:
 
        w = self.weight_mlp(units).squeeze(-1)    
        c = torch.einsum("bn,bne->be", w, units)  
        y = self.head(c)                           
        return y, w


 
class KREForm1(nn.Module):
    def __init__(self, n_units: int = 16, embed_dim: int = 64, horizon: int = 100):
        super().__init__()
        self.cnn = UnitDiscoveryCNN(n_units=n_units, embed_dim=embed_dim)
        self.head = UnitWeightingHead(embed_dim=embed_dim, n_units=n_units, horizon=horizon)

    def forward(self, x: torch.Tensor):
        units = self.cnn(x)         
        y, w  = self.head(units)   
        return y, w




 
def save_model(model: "KREForm1", path: str = "kre_form1.pt"):
    """Save the model state, config, and a quick loss log for reuse."""
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "n_units": model.cnn.n_units,
            "embed_dim": model.cnn.embed_dim,
            "horizon": model.head.head.out_features,
        },
    }
    torch.save(ckpt, path)
    print(f"[KRE-Form1] saved checkpoint to {path}")


def load_model(path: str = "kre_form1.pt", device: str | None = None) -> "KREForm1":
    """Reload a saved KREForm1. Defaults to CPU if no device given."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = KREForm1(n_units=cfg["n_units"], embed_dim=cfg["embed_dim"], horizon=cfg["horizon"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[KRE-Form1] loaded {path} on {device}")
    return model


def train(steps: int = 3000, batch_size: int = 32, log_every: int = 200, plot: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KREForm1().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[KRE-Form1] device={device}  params={n_params:,}")

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
       
        x = np.stack([make_window(seed=None)[0] for _ in range(batch_size)])
        y = np.stack([make_window(seed=None)[1] for _ in range(batch_size)])
        x = torch.from_numpy(x).unsqueeze(1).to(device)  
        y = torch.from_numpy(y).to(device)            

        yhat, w = model(x)
        loss = ((yhat - y) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            avg = sum(losses[-log_every:]) / min(log_every, len(losses))
            print(f"  step {step:5d}/{steps}  loss={avg:.5f}  ({elapsed:.1f}s)")

    if plot:
        
        _plot_results(model, device, losses)
    save_model(model)   
    return model, losses


 

def _plot_results(model, device, losses, out_dir: str = "."):
    out = Path(out_dir)
  
    plt.figure(figsize=(8, 3))
    plt.plot(losses)
    plt.title("Training loss (MSE on next-heartbeat regression)")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(out / "kre_loss.png", dpi=120)
    plt.close()

     
    model.eval()
    with torch.no_grad():
        x_win, y_true = make_window(seed=42)
        x = torch.from_numpy(x_win).unsqueeze(0).unsqueeze(0).to(device)
        yhat, w = model(x)
        yhat = yhat.squeeze(0).cpu().numpy()
        w = w.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    axes[0].plot(x_win, label="input window (16 heartbeats)", color="C0")
    axes[0].plot(np.arange(1600, 1700), y_true, label="true next beat", color="green", lw=2)
    axes[0].plot(np.arange(1600, 1700), yhat, "--", label="predicted next beat", color="red", lw=2)
    axes[0].axvline(1600, color="k", ls=":", alpha=0.5)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Next-heartbeat prediction")

    axes[1].stem(np.arange(1, len(w) + 1), w, basefmt=" ")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Per-unit weights learned by the MLP (no softmax)")
    axes[1].set_xlabel("unit index")
    axes[1].set_ylabel("weight")
    plt.tight_layout()
    plt.savefig(out / "kre_pred.png", dpi=120)
    plt.close()
    print(f"[KRE-Form1] saved {out/'kre_loss.png'} and {out/'kre_pred.png'}")

 
def _baseline_wander(n: int, rng: np.random.RandomState, amp: float = 0.3) -> np.ndarray:
    """Slow low-frequency drift (respiration, electrode drift)."""
    t = np.arange(n, dtype=np.float32) / 100.0
    f1 = rng.uniform(0.1, 0.3)
    f2 = rng.uniform(0.05, 0.15)
    return (amp * np.sin(2 * np.pi * f1 * t) + 0.5 * amp * np.sin(2 * np.pi * f2 * t)).astype(np.float32)


def _muscle_noise(n: int, rng: np.random.RandomState, amp: float = 0.05) -> np.ndarray:
    """High-frequency EMG / muscle artefact."""
    return rng.normal(0, amp, n).astype(np.float32)


def _powerline(n: int, rng: np.random.RandomState, freq: float = 50.0, amp: float = 0.05) -> np.ndarray:
    """50 Hz (or 60 Hz) mains interference."""
    t = np.arange(n, dtype=np.float32) / 125.0
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _pvc_beat(rng: np.random.RandomState) -> np.ndarray:
    """Premature ventricular contraction: wide QRS, no P, big amplitude."""
    t = np.arange(100, dtype=np.float32)
    return (1.4 * np.exp(-0.5 * ((t - 55) / 10) ** 2)
          - 0.25 * np.exp(-0.5 * ((t - 70) / 4) ** 2)
          - 0.25 * np.exp(-0.5 * ((t - 80) / 4) ** 2)).astype(np.float32)


def _af_beat(rng: np.random.RandomState) -> np.ndarray:
    """Atrial fibrillation: irregular amplitude, no clear P, noisy baseline."""
    t = np.arange(100, dtype=np.float32)
    amp = rng.uniform(0.6, 1.2)
    return (amp * np.exp(-0.5 * ((t - 40) / 8) ** 2)
          - 0.2 * np.exp(-0.5 * ((t - 50) / 3) ** 2)
          + 0.3 * np.exp(-0.5 * ((t - 75) / 8) ** 2)
          + rng.normal(0, 0.06, 100).astype(np.float32)).astype(np.float32)


def _artefact_beat(rng: np.random.RandomState) -> np.ndarray:
    """Motion artefact: a high-frequency burst mid-beat over a normal beat."""
    beat = _single_beat(rng)
    start = rng.randint(20, 70)
    burst = rng.normal(0, 0.7, 30).astype(np.float32)
    beat[start:start + 30] += burst
    return beat


def _compensatory_beat(rng: np.random.RandomState) -> np.ndarray:
    """Post-PVC compensatory beat: low-amplitude, slightly prolonged."""
    t = np.arange(100, dtype=np.float32)
    return (0.35 * np.exp(-0.5 * ((t - 50) / 10) ** 2)).astype(np.float32)

 

def make_window_v3(
    mode: str = "baseline",
    n_beats: int = 16,
    seed: int | None = None,
):
    """
    Modes (each returns a 1600-sample window + 100-sample target + info dict):
      'baseline'   : control — original simple generator
      'wander'     : + baseline wander
      'noise'      : + muscle noise + powerline interference
      'vrr'        : variable RR interval with slow drift in heart rate
      'morphology' : mix of normal / PVC / AF / artefact beats
      'pvc_chain'  : a PVC at beat k forces a compensatory beat at k+1
      'af_run'     : a run of AF beats in the middle of the window
      'unaligned'  : window starts mid-beat, ends mid-beat
      'all'        : everything together (the kitchen-sink stress test)
    """
    rng = np.random.RandomState(seed)
    SAMPLES_PER_BEAT = 100  

    
    if mode == "baseline":
        win, tgt = make_window(n_beats=n_beats, seed=seed)
        return win, tgt, {}
 
    if mode == "wander":
        win, tgt = make_window(n_beats=n_beats, seed=seed)
        win = (win + _baseline_wander(len(win), rng)).astype(np.float32)
        tgt = (tgt + _baseline_wander(len(tgt), rng, amp=0.3)).astype(np.float32)
        return win, tgt, {}
 
    if mode == "noise":
        win, tgt = make_window(n_beats=n_beats, seed=seed)
        win = (win + _muscle_noise(len(win), rng, amp=0.08) + _powerline(len(win), rng)).astype(np.float32)
        tgt = (tgt + _muscle_noise(len(tgt), rng, amp=0.08) + _powerline(len(tgt), rng)).astype(np.float32)
        return win, tgt, {}
 
    if mode == "vrr":
        
        drift = np.linspace(rng.uniform(-15, 15), rng.uniform(-15, 15), n_beats + 1)
        rr = np.maximum(70, (SAMPLES_PER_BEAT + drift + rng.normal(0, 3, n_beats + 1)).astype(int))
        beats = []
        for length in rr:
            b = _single_beat(rng)
            if len(b) >= length:
                beats.append(b[:length])
            else:
                beats.append(np.pad(b, (0, length - len(b))))
      
        raw = np.concatenate(beats).astype(np.float32)
        x_old = np.linspace(0, 1, len(raw))
        x_new = np.linspace(0, 1, n_beats * SAMPLES_PER_BEAT)
        win = np.interp(x_new, x_old, raw).astype(np.float32)
       
        tgt = _single_beat(rng).astype(np.float32)
        return win, tgt, {"rr": rr.tolist()}
 
    if mode == "morphology":
        kinds = rng.choice(
            ["normal", "pvc", "af", "artefact"],
            size=n_beats + 1,
            p=[0.80, 0.07, 0.07, 0.06],
        )
        beat_fns = {"normal": lambda: _single_beat(rng),
                    "pvc":    lambda: _pvc_beat(rng),
                    "af":     lambda: _af_beat(rng),
                    "artefact": lambda: _artefact_beat(rng)}
        beats = []
        for k in kinds:
            b = beat_fns[k]()
            if len(b) >= SAMPLES_PER_BEAT:
                beats.append(b[:SAMPLES_PER_BEAT])
            else:
                beats.append(np.pad(b, (0, SAMPLES_PER_BEAT - len(b))))
        win = np.concatenate(beats[:n_beats]).astype(np.float32)
        tgt = beats[n_beats].astype(np.float32)
        return win, tgt, {"kinds": kinds.tolist()}
 
    if mode == "pvc_chain":
    
        k = rng.randint(2, n_beats - 2)
        beats = [_single_beat(rng) for _ in range(n_beats + 1)]
        beats[k] = _pvc_beat(rng)
        beats[n_beats] = _compensatory_beat(rng)
        beats = [b[:SAMPLES_PER_BEAT] if len(b) >= SAMPLES_PER_BEAT
                 else np.pad(b, (0, SAMPLES_PER_BEAT - len(b))) for b in beats]
        win = np.concatenate(beats[:n_beats]).astype(np.float32)
        tgt = beats[n_beats].astype(np.float32)
        return win, tgt, {"pvc_position": k}

 
    if mode == "af_run":
        beats = [_single_beat(rng) for _ in range(n_beats + 1)]
        start = rng.randint(4, n_beats - 6)
        length = rng.randint(4, 7)
        for i in range(start, min(start + length, n_beats + 1)):
            beats[i] = _af_beat(rng)
        beats = [b[:SAMPLES_PER_BEAT] if len(b) >= SAMPLES_PER_BEAT
                 else np.pad(b, (0, SAMPLES_PER_BEAT - len(b))) for b in beats]
        win = np.concatenate(beats[:n_beats]).astype(np.float32)
        tgt = beats[n_beats].astype(np.float32)
        return win, tgt, {"af_start": start, "af_length": length}

 
    if mode == "unaligned":
  
        full_win, full_tgt = make_window(n_beats=n_beats + 2, seed=seed)
        full = np.concatenate([full_win, full_tgt])
        start = rng.randint(20, 80)
        end = start + n_beats * SAMPLES_PER_BEAT
        if end + SAMPLES_PER_BEAT > len(full):
            end = len(full) - SAMPLES_PER_BEAT
        win = full[start:end].astype(np.float32)
        tgt = full[end:end + SAMPLES_PER_BEAT].astype(np.float32)
     
        if len(win) < n_beats * SAMPLES_PER_BEAT:
            win = np.pad(win, (0, n_beats * SAMPLES_PER_BEAT - len(win))).astype(np.float32)
        if len(tgt) < SAMPLES_PER_BEAT:
            tgt = np.pad(tgt, (0, SAMPLES_PER_BEAT - len(tgt))).astype(np.float32)
        return win, tgt, {"start_offset": start}

  
    if mode == "all":
      
        win, tgt, info = make_window_v3("morphology", n_beats=n_beats, seed=seed)
        win = (win + _baseline_wander(len(win), rng) + _muscle_noise(len(win), rng, 0.06)).astype(np.float32)
        tgt = (tgt + _baseline_wander(len(tgt), rng) + _muscle_noise(len(tgt), rng, 0.06)).astype(np.float32)
        info["kitchen_sink"] = True
        return win, tgt, info

    raise ValueError(f"unknown mode {mode!r}")
 

def _plot_stress(mode: str, model, device, seed: int = 11):
    model.eval()
    win, tgt, info = make_window_v3(mode=mode, seed=seed)
    with torch.no_grad():
        x = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).to(device)
        yhat, w = model(x)
        yhat = yhat.squeeze(0).cpu().numpy()
        w = w.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    full = np.concatenate([win, tgt])
    axes[0].plot(full, color="C0", lw=0.8, label="input + true target")
    axes[0].plot(np.arange(len(win), len(full)), tgt, color="green", lw=2, label="true target")
    axes[0].plot(np.arange(len(win), len(win) + len(yhat)), yhat,
                  "--", color="red", lw=2, label="predicted target")
    axes[0].axvline(len(win), color="k", ls=":", alpha=0.5)

 
    if mode == "morphology" and "kinds" in info:
        colours = {"normal": "grey", "pvc": "orange", "af": "magenta", "artefact": "brown"}
        for i, k in enumerate(info["kinds"][:-1]):
            axes[0].axvspan(i * 100, (i + 1) * 100, color=colours.get(k, "grey"), alpha=0.12)
        for name, col in colours.items():
            if name != "normal":
                axes[0].plot([], [], color=col, alpha=0.4, lw=6, label=name)
    if mode == "pvc_chain" and "pvc_position" in info:
        k = info["pvc_position"]
        axes[0].axvspan(k * 100, (k + 1) * 100, color="orange", alpha=0.25, label="PVC")
    if mode == "af_run" and "af_start" in info:
        s, L = info["af_start"], info["af_length"]
        axes[0].axvspan(s * 100, (s + L) * 100, color="magenta", alpha=0.20, label="AF run")

    axes[0].set_title(f"Stress test: {mode}    (final loss in this mode)")
    axes[0].legend(loc="upper right", fontsize=8)

 
    axes[1].stem(np.arange(1, len(w) + 1), w, basefmt=" ")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("per-unit weights")
    axes[1].set_xlabel("unit index")

    plt.tight_layout()
    fname = f"stress_{mode}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"[stress] saved {fname}")


def run_stress_tests(steps_per_test: int = 1500, batch_size: int = 32):
    """Train the same model briefly on each mode and visualise.
    Each test is isolated — a failure in one mode does not stop the others."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KREForm1().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    modes = ["baseline", "wander", "noise", "vrr", "morphology",
             "pvc_chain", "af_run", "unaligned", "all"]

    for mode in modes:
        print(f"\n[stress] mode = {mode}")
        try:
            losses = []
            t0 = time.time()
            for step in range(1, steps_per_test + 1):
                xs, ys = [], []
                for _ in range(batch_size):
                    w, t, _ = make_window_v3(mode=mode, seed=None)
                    xs.append(w); ys.append(t)
                x = torch.from_numpy(np.stack(xs)).unsqueeze(1).to(device)
                y = torch.from_numpy(np.stack(ys)).to(device)
                yhat, _ = model(x)
                loss = ((yhat - y) ** 2).mean()
                optim.zero_grad(); loss.backward(); optim.step()
                losses.append(loss.item())
                if step % 500 == 0:
                    print(f"  step {step:4d}  loss={sum(losses[-500:])/500:.5f}")
            print(f"  ({time.time()-t0:.1f}s)  final_loss={sum(losses[-200:])/200:.5f}")
            _plot_stress(mode, model, device)
        except Exception as e:
            print(f"  [FAIL] mode={mode}  {type(e).__name__}: {e}")
            continue

    save_model(model, "kre_form1_stress.pt")
    print("\n[stress] all modes attempted. PNGs in working directory.")


if __name__ == "__main__":
    visualize_ekg_data()
    train()
    run_stress_tests()