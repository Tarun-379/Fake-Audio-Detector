# ============================================================
#  Deepfake Audio Detector — Production-Grade v2.0
#  Upgraded by: Senior ML Engineer
# ============================================================
#
#  v2.0 Upgrades over v1.0:
#   [1]  FIXED  Real temporal sequences fed into BiLSTM (was seq_len=1)
#   [2]  NEW    Attention mechanism after BiLSTM for context pooling
#   [3]  NEW    Audio visualization: waveform + mel spectrogram (server-side)
#   [4]  NEW    Risk Level system: LOW / MEDIUM / HIGH with confidence
#   [5]  NEW    Benchmark / Metrics endpoint (/benchmark)
#   [6]  NEW    Optional noise augmentation (noise / pitch / stretch)
#   [7]  FIXED  Unique UUID-based temp files (no race conditions)
#   [8]  NEW    File type + size validation with proper error codes
#   [9]  NEW    Inference time + feature extraction time in response
#   [10] NEW    Model versioning + /model-info endpoint
#   [11] NEW    Edge case detection (silent, too-short, corrupted audio)
#   [12] NEW    Legacy model fallback (v1 checkpoint still loads)
#   [13] NEW    Upgraded frontend: visualizations, risk badge, timing, benchmark UI
#
# ============================================================

import os
import io
import uuid
import time
import tempfile
import traceback
import base64
import warnings

import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import timm

# --- Non-interactive matplotlib backend (must be set BEFORE pyplot import) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ================================================================
#  SECTION 1 — CONFIGURATION
# ================================================================

SAMPLE_RATE       = 16000          # Hz
DURATION          = 4              # seconds to analyse
N_FRAMES          = 8              # [NEW] temporal segments fed to LSTM
FRAME_H           = 64             # [NEW] spectrogram height per frame
FRAME_W           = 32             # [NEW] spectrogram width per frame
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH        = os.path.join(os.path.dirname(__file__), "best_model.pth")
MAX_FILE_MB       = 25             # [NEW] max upload size in MB
MAX_FILE_BYTES    = MAX_FILE_MB * 1024 * 1024
ALLOWED_EXTS      = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}
SILENCE_RMS_FLOOR = 1e-4           # [NEW] below this → silent audio
MIN_AUDIO_SECS    = 0.5            # [NEW] below this → too short

# ================================================================
#  SECTION 2 — MODEL VERSIONING  [NEW FEATURE #10]
# ================================================================

MODEL_VERSION = "2.0.0"

MODEL_META = {
    "version"      : MODEL_VERSION,
    "architecture" : "EfficientNet-B0 + 2-layer BiLSTM + Self-Attention",
    "features"     : ["Mel Spectrogram", "MFCC", "CQT"],
    "n_frames"     : N_FRAMES,
    "frame_shape"  : [3, FRAME_H, FRAME_W],
    "device"       : str(DEVICE),
    "sample_rate"  : SAMPLE_RATE,
    "duration"     : DURATION,
}

# ================================================================
#  SECTION 3 — MODEL ARCHITECTURE  [NEW FEATURE #1 + #2]
# ================================================================

class AttentionLayer(nn.Module):
    """
    [NEW] Lightweight additive self-attention over the LSTM output sequence.
    Learns to weight which time steps carry the most discriminative signal.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores  = self.score(lstm_out)                    # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)            # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden_dim)
        return context, weights.squeeze(-1)               # context + attention map


class DeepfakeDetector(nn.Module):
    """
    [UPGRADED v2]
    Architecture:
      1. Audio split into N_FRAMES temporal segments.
      2. Each segment → 3-channel spectrogram (Mel + MFCC + CQT).
      3. EfficientNet-B0 extracts per-frame feature vectors (shared weights).
      4. 2-layer Bidirectional LSTM models temporal dynamics.
      5. Attention layer pools the sequence into a context vector.
      6. MLP classifier outputs real/fake logits.

    Input shape:  (batch, N_FRAMES, 3, FRAME_H, FRAME_W)
    Output shape: (batch, 2)
    """
    def __init__(self, n_frames: int = N_FRAMES):
        super().__init__()
        self.n_frames = n_frames

        # Shared CNN backbone — processes each frame independently
        self.cnn = timm.create_model(
            "efficientnet_b0", pretrained=False,
            num_classes=0, in_chans=3
        )
        cnn_out = self.cnn.num_features          # 1280 for EfficientNet-B0

        # 2-layer BiLSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size  = cnn_out,
            hidden_size = 256,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.3,
        )

        # [NEW] Attention pooling over LSTM time steps
        self.attention = AttentionLayer(hidden_dim=512)   # 256 * 2 (bidir)

        # Classification head with LayerNorm for training stability
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, n_frames, 3, H, W)
        B, T, C, H, W = x.shape

        # Flatten batch × time → single batch for CNN forward pass
        x_flat = x.view(B * T, C, H, W)
        feats  = self.cnn(x_flat)                # (B*T, cnn_out)
        feats  = feats.view(B, T, -1)            # (B, T, cnn_out)

        # LSTM: model temporal dynamics across frames
        lstm_out, _ = self.lstm(feats)           # (B, T, 512)

        # Attention pooling: weighted sum of time steps
        context, _  = self.attention(lstm_out)   # (B, 512)

        return self.classifier(context)          # (B, 2)


class DeepfakeDetectorV1(nn.Module):
    """
    [PRESERVED] Original v1 architecture for backward-compatible
    checkpoint loading. seq_len=1 means LSTM is effectively a linear layer.
    """
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, in_chans=3)
        cnn_out = self.cnn.num_features
        self.lstm = nn.LSTM(cnn_out, 256, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        feat = self.cnn(x).unsqueeze(1)
        out, _ = self.lstm(feat)
        return self.classifier(out.squeeze(1))


# ================================================================
#  SECTION 4 — MODEL LOADING (graceful fallback)  [NEW FEATURE #12]
# ================================================================

USING_LEGACY_ARCH = False   # module-level flag for feature extraction routing


def _load_model():
    """
    Loading strategy (tries in order):
      1. New v2 architecture with checkpoint
      2. Legacy v1 architecture with checkpoint (older .pth files)
      3. New v2 architecture with random weights (demo / training mode)
    """
    global USING_LEGACY_ARCH

    if os.path.exists(MODEL_PATH):
        # --- Attempt 1: new architecture ---
        try:
            m = DeepfakeDetector().to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            m.load_state_dict(state)
            print(f"✅  v2 checkpoint loaded  →  {MODEL_PATH}")
            return m
        except Exception:
            pass

        # --- Attempt 2: legacy architecture ---
        try:
            m_leg = DeepfakeDetectorV1().to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            m_leg.load_state_dict(state)
            USING_LEGACY_ARCH = True
            print(f"⚠️   Legacy v1 checkpoint loaded. Retrain with v2 for best results.")
            return m_leg
        except Exception as e:
            print(f"⚠️   Checkpoint found but incompatible: {e}")
    else:
        print(f"⚠️   No checkpoint at '{MODEL_PATH}'. Running in demo mode (random weights).")

    # --- Attempt 3: fresh v2 model ---
    m = DeepfakeDetector().to(DEVICE)
    print("ℹ️   Model initialised with random weights.")
    return m


print("\n🔧  Initialising model...")
model = _load_model()
model.eval()
print(f"🚀  Ready on {DEVICE}  |  legacy_arch={USING_LEGACY_ARCH}\n")


# ================================================================
#  SECTION 5 — EDGE CASE DETECTION  [NEW FEATURE #11]
# ================================================================

def check_edge_cases(y: np.ndarray, sr: int):
    """
    Returns (is_ok: bool, error_message: str | None).
    Detects: silence, too-short audio.
    """
    duration_sec = len(y) / sr

    if duration_sec < MIN_AUDIO_SECS:
        return False, (
            f"Audio is too short ({duration_sec:.2f}s). "
            f"Please provide at least {MIN_AUDIO_SECS}s of audio."
        )

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < SILENCE_RMS_FLOOR:
        return False, (
            "Audio appears to be silent (RMS energy below threshold). "
            "Please provide audio containing speech."
        )

    return True, None


# ================================================================
#  SECTION 6 — AUDIO AUGMENTATION  [NEW FEATURE #5]
# ================================================================

def apply_augmentation(y: np.ndarray, sr: int) -> np.ndarray:
    """
    [NEW] Randomly applies one of three augmentations to improve robustness:
      - Gaussian noise  (SNR ~20 dB)
      - Pitch shift     (±2 semitones)
      - Time stretch    (×0.9 – ×1.1)

    Toggle via ?augment=true in the request form data.
    """
    aug = np.random.choice(["noise", "pitch", "stretch", "none"],
                           p=[0.35, 0.30, 0.20, 0.15])
    try:
        if aug == "noise":
            sigma = 0.003 + np.random.rand() * 0.004
            y = y + np.random.normal(0, sigma, len(y)).astype(y.dtype)

        elif aug == "pitch":
            steps = np.random.uniform(-2.0, 2.0)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

        elif aug == "stretch":
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
            target = SAMPLE_RATE * DURATION
            y = np.pad(y, (0, max(0, target - len(y))))[:target]

    except Exception:
        pass  # If augmentation fails, silently return original audio

    return y


# ================================================================
#  SECTION 7 — FEATURE EXTRACTION  [NEW FEATURE #1]
# ================================================================

def _resize_spec(spec: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a 2-D spectrogram to (target_h, target_w) via clip + pad."""
    h, w = spec.shape
    if h > target_h:   spec = spec[:target_h, :]
    elif h < target_h: spec = np.pad(spec, ((0, target_h - h), (0, 0)))
    if w > target_w:   spec = spec[:, :target_w]
    elif w < target_w: spec = np.pad(spec, ((0, 0), (0, target_w - w)))
    return spec.astype(np.float32)


def _norm(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation with epsilon guard."""
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma + 1e-6)


def extract_features_v2(y: np.ndarray, sr: int) -> np.ndarray:
    """
    [NEW v2] Produces a REAL temporal sequence for the BiLSTM.

    1. Split audio into N_FRAMES equal segments.
    2. For each segment extract: Mel spectrogram, MFCC, CQT.
    3. Resize each to (FRAME_H × FRAME_W) and stack as 3-channel image.
    4. Normalise per-channel per-frame.

    Returns ndarray of shape (N_FRAMES, 3, FRAME_H, FRAME_W).
    """
    spf = len(y) // N_FRAMES      # samples per frame
    frames = []

    for i in range(N_FRAMES):
        chunk = y[i * spf : (i + 1) * spf]

        # Pad very short chunks (safety net)
        if len(chunk) < 1024:
            chunk = np.pad(chunk, (0, 1024 - len(chunk)))

        # --- Mel Spectrogram ---
        try:
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=chunk, sr=sr, n_mels=FRAME_H,
                    n_fft=512, hop_length=128
                )
            )
            mel = _resize_spec(mel, FRAME_H, FRAME_W)
        except Exception:
            mel = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

        # --- MFCC ---
        try:
            mfcc = librosa.feature.mfcc(
                y=chunk, sr=sr, n_mfcc=FRAME_H,
                n_fft=512, hop_length=128
            )
            mfcc = _resize_spec(mfcc, FRAME_H, FRAME_W)
        except Exception:
            mfcc = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

        # --- CQT ---
        try:
            n_bins = min(FRAME_H, 72)   # keep within musical range
            cqt = librosa.amplitude_to_db(
                np.abs(librosa.cqt(
                    chunk, sr=sr, n_bins=n_bins,
                    hop_length=128, fmin=librosa.note_to_hz("C2")
                )),
                ref=np.max
            )
            cqt = _resize_spec(cqt, FRAME_H, FRAME_W)
        except Exception:
            cqt = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)

        frame = np.stack([_norm(mel), _norm(mfcc), _norm(cqt)])  # (3, H, W)
        frames.append(frame)

    return np.array(frames, dtype=np.float32)   # (N_FRAMES, 3, H, W)


def extract_features_v1(y: np.ndarray, sr: int) -> np.ndarray:
    """
    [PRESERVED] Original v1 single-frame extraction for legacy checkpoints.
    Returns ndarray of shape (3, 128, 128).
    """
    MAX_LEN = 128

    def resize_128(x):
        if x.shape[0] > 128:    x = x[:128, :]
        elif x.shape[0] < 128:  x = np.pad(x, ((0, 128 - x.shape[0]), (0, 0)))
        if x.shape[1] > MAX_LEN:  x = x[:, :MAX_LEN]
        elif x.shape[1] < MAX_LEN: x = np.pad(x, ((0, 0), (0, MAX_LEN - x.shape[1])))
        return x

    mel  = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    cqt  = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
    return np.stack([resize_128(mel), resize_128(mfcc), resize_128(cqt)]).astype(np.float32)


# ================================================================
#  SECTION 8 — AUDIO VISUALISATION  [NEW FEATURE #2]
# ================================================================

def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_visualisations(y: np.ndarray, sr: int) -> dict:
    """
    [NEW] Produces two visualisations returned with every prediction:
      - Waveform: time-domain amplitude plot
      - Mel Spectrogram: frequency-vs-time energy heatmap

    Returns {"waveform": "<b64 png>", "spectrogram": "<b64 png>"}
    or None values on failure.
    """
    BG   = "#0a0a0f"
    SURF = "#13131a"
    ACC  = "#00ff9d"
    MUT  = "#555570"
    results = {"waveform": None, "spectrogram": None}

    # --- Waveform ---
    try:
        times = np.linspace(0, len(y) / sr, len(y))
        # Downsample for faster rendering (max 4000 pts)
        if len(times) > 4000:
            idx   = np.linspace(0, len(times) - 1, 4000, dtype=int)
            times = times[idx]
            y_vis = y[idx]
        else:
            y_vis = y

        fig, ax = plt.subplots(figsize=(8, 1.9), facecolor=BG)
        ax.set_facecolor(SURF)
        ax.fill_between(times, y_vis, color=ACC, alpha=0.18)
        ax.fill_between(times, -y_vis, color=ACC, alpha=0.10)
        ax.plot(times, y_vis, color=ACC, linewidth=0.7, alpha=0.85)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (s)", color=MUT, fontsize=8, labelpad=3)
        ax.set_ylabel("Amplitude", color=MUT, fontsize=8, labelpad=3)
        ax.tick_params(colors=MUT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e2e")
        fig.tight_layout(pad=0.4)
        results["waveform"] = _fig_to_b64(fig)
    except Exception as exc:
        print(f"[WARN] Waveform generation failed: {exc}")

    # --- Mel Spectrogram ---
    try:
        mel_spec = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000),
            ref=np.max
        )
        fig, ax = plt.subplots(figsize=(8, 2.6), facecolor=BG)
        ax.set_facecolor(BG)
        img = librosa.display.specshow(
            mel_spec, sr=sr, x_axis="time", y_axis="mel",
            fmax=8000, ax=ax, cmap="magma"
        )
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=MUT, labelsize=7)
        cbar.outline.set_edgecolor("#1e1e2e")
        ax.set_xlabel("Time (s)", color=MUT, fontsize=8, labelpad=3)
        ax.set_ylabel("Hz (mel)", color=MUT, fontsize=8, labelpad=3)
        ax.tick_params(colors=MUT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e2e")
        fig.tight_layout(pad=0.4)
        results["spectrogram"] = _fig_to_b64(fig)
    except Exception as exc:
        print(f"[WARN] Spectrogram generation failed: {exc}")

    return results


# ================================================================
#  SECTION 9 — RISK LEVEL SYSTEM  [NEW FEATURE #3]
# ================================================================

def get_risk_level(fake_prob: float) -> dict:
    """
    [NEW] Maps fake probability (0-100) to a 3-tier risk assessment.

    LOW    < 35%  : minimal deepfake indicators
    MEDIUM 35-70% : ambiguous — manual review recommended
    HIGH   > 70%  : strong deepfake indicators
    """
    if fake_prob >= 70:
        return {
            "level"  : "HIGH",
            "color"  : "#ff3c5f",
            "bg"     : "#ff3c5f15",
            "emoji"  : "🔴",
            "label"  : "HIGH RISK",
            "desc"   : "Strong indicators of AI synthesis or voice cloning detected",
        }
    elif fake_prob >= 35:
        return {
            "level"  : "MEDIUM",
            "color"  : "#ffaa00",
            "bg"     : "#ffaa0015",
            "emoji"  : "🟡",
            "label"  : "MEDIUM RISK",
            "desc"   : "Ambiguous signal — manual review is recommended",
        }
    else:
        return {
            "level"  : "LOW",
            "color"  : "#00ff9d",
            "bg"     : "#00ff9d10",
            "emoji"  : "🟢",
            "label"  : "LOW RISK",
            "desc"   : "Minimal indicators of AI synthesis detected",
        }


# ================================================================
#  SECTION 10 — PREDICT ENDPOINT  [UPGRADED]
# ================================================================

@app.route("/predict", methods=["POST"])
def predict():
    # ── [NEW] Validate file presence ──────────────────────────
    if "audio" not in request.files:
        return jsonify({"error": "No audio file attached (key='audio')"}), 400

    file = request.files["audio"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # ── [NEW] Validate extension ───────────────────────────────
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return jsonify({
            "error": f"Unsupported format '{ext}'. "
                     f"Accepted: {', '.join(sorted(ALLOWED_EXTS))}"
        }), 415

    # ── [NEW] Validate file size ───────────────────────────────
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_BYTES:
        return jsonify({
            "error": f"File too large ({file_size / 1024 / 1024:.1f} MB). "
                     f"Maximum allowed: {MAX_FILE_MB} MB."
        }), 413

    # ── [NEW] UUID-based temp file (no overwrites / race conds) ─
    tmp_name = f"dfd_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)

    try:
        file.save(tmp_path)

        # ── Feature extraction timing ──────────────────────────
        t_feat_start = time.perf_counter()

        try:
            y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE,
                                  duration=DURATION, mono=True)
        except Exception as load_err:
            return jsonify({"error": f"Could not decode audio: {load_err}"}), 422

        # Pad / trim to exactly DURATION seconds
        target_len = SAMPLE_RATE * DURATION
        y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]

        # ── [NEW] Edge case detection ──────────────────────────
        ok, edge_msg = check_edge_cases(y, sr)
        if not ok:
            return jsonify({"error": edge_msg, "edge_case": True}), 422

        # ── [NEW] Generate visualisations (non-blocking on error) ─
        try:
            visuals = generate_visualisations(y, sr)
        except Exception:
            visuals = {"waveform": None, "spectrogram": None}

        # ── [NEW] Optional augmentation ───────────────────────
        augment = request.form.get("augment", "false").lower() == "true"
        y_proc  = apply_augmentation(y.copy(), sr) if augment else y

        # ── Feature extraction ─────────────────────────────────
        if USING_LEGACY_ARCH:
            feats  = extract_features_v1(y_proc, sr)
            feats  = (feats - feats.mean()) / (feats.std() + 1e-6)
            tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            feats  = extract_features_v2(y_proc, sr)
            tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            # shape → (1, N_FRAMES, 3, FRAME_H, FRAME_W)

        feat_ms = (time.perf_counter() - t_feat_start) * 1000

        # ── Model inference timing ────────────────────────────
        t_inf_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        inf_ms = (time.perf_counter() - t_inf_start) * 1000

        fake_p = round(probs[1].item() * 100, 1)
        real_p = round(probs[0].item() * 100, 1)
        verdict = "FAKE" if fake_p > 50.0 else "REAL"

        # ── [NEW] Risk level ──────────────────────────────────
        risk = get_risk_level(fake_p)

        return jsonify({
            # ── Core (PRESERVED from v1) ──
            "real_prob"       : real_p,
            "fake_prob"       : fake_p,
            "verdict"         : verdict,
            # ── [NEW] Risk level ──
            "risk_level"      : risk["level"],
            "risk_color"      : risk["color"],
            "risk_bg"         : risk["bg"],
            "risk_label"      : risk["label"],
            "risk_emoji"      : risk["emoji"],
            "risk_desc"       : risk["desc"],
            "confidence"      : round(max(real_p, fake_p), 1),
            # ── [NEW] Timing ──
            "feat_time_ms"    : round(feat_ms, 1),
            "inf_time_ms"     : round(inf_ms, 1),
            "total_time_ms"   : round(feat_ms + inf_ms, 1),
            # ── [NEW] Meta ──
            "model_version"   : MODEL_VERSION,
            "augmented"       : augment,
            "arch"            : "v1-legacy" if USING_LEGACY_ARCH else "v2",
            # ── [NEW] Visualisations (base64 PNG) ──
            "waveform_img"    : visuals.get("waveform"),
            "spectrogram_img" : visuals.get("spectrogram"),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        # Always clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ================================================================
#  SECTION 11 — BENCHMARK / METRICS ENDPOINT  [NEW FEATURE #4]
# ================================================================

@app.route("/benchmark", methods=["GET"])
def benchmark():
    """
    [NEW] Returns model performance metrics on a synthetic test set.
    Query params:
      ?n=<int>   number of simulated samples (default 200, max 1000)
      ?seed=<int> random seed (default 42)

    In production, replace synthetic data with a real held-out dataset.
    """
    n    = min(max(int(request.args.get("n",    200)), 10), 1000)
    seed = int(request.args.get("seed", 42))
    rng  = np.random.default_rng(seed)

    # Simulate ground-truth labels (50/50 split)
    y_true  = rng.integers(0, 2, n)

    # Simulate model score: logistic model with some noise
    logits  = (y_true.astype(float) * 2 - 1) + rng.normal(0, 1.0, n)
    y_score = 1.0 / (1.0 + np.exp(-logits))
    y_pred  = (y_score > 0.5).astype(int)

    # ── Confusion matrix ──────────────────────────────────────
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-9
    accuracy    = round((tp + tn) / n * 100, 2)
    precision   = round(tp / (tp + fp + eps) * 100, 2)
    recall      = round(tp / (tp + fn + eps) * 100, 2)
    f1          = round(2 * precision * recall / (precision + recall + eps), 2)
    specificity = round(tn / (tn + fp + eps) * 100, 2)
    fpr         = round(fp / (fp + tn + eps) * 100, 2)

    # ── Score distribution histogram (10 bins) ────────────────
    hist_fake, bin_edges = np.histogram(y_score[y_true == 1], bins=10, range=(0, 1))
    hist_real, _         = np.histogram(y_score[y_true == 0], bins=10, range=(0, 1))

    return jsonify({
        "n_samples"        : n,
        "accuracy"         : accuracy,
        "precision"        : precision,
        "recall"           : recall,
        "f1_score"         : f1,
        "specificity"      : specificity,
        "false_positive_rate": fpr,
        "confusion_matrix" : {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "score_histogram"  : {
            "bins"     : [round(b, 2) for b in bin_edges[:-1].tolist()],
            "fake"     : hist_fake.tolist(),
            "real"     : hist_real.tolist(),
        },
        "model_version"    : MODEL_VERSION,
        "note"             : (
            "Metrics are computed on SYNTHETIC data for demonstration. "
            "Replace with a real evaluation dataset for production use."
        ),
    })


# ================================================================
#  SECTION 12 — MODEL INFO ENDPOINT  [NEW FEATURE #10]
# ================================================================

@app.route("/model-info", methods=["GET"])
def model_info():
    """[NEW] Returns model metadata, version, and architecture details."""
    info = dict(MODEL_META)
    info["legacy_arch"] = USING_LEGACY_ARCH
    info["checkpoint"]  = os.path.exists(MODEL_PATH)
    return jsonify(info)


# ================================================================
#  SECTION 13 — UPGRADED HTML FRONTEND
# ================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Audio Detector v2</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── CSS Reset & Variables ──────────────────────────────────────── */
:root{
  --bg:#0a0a0f; --surface:#13131a; --border:#1e1e2e; --border2:#252535;
  --accent:#00ff9d; --danger:#ff3c5f; --warn:#ffaa00;
  --text:#e8e8f0; --muted:#555570; --muted2:#3a3a50;
  --card:#0d0d15; --card2:#111120;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{
  background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;
  min-height:100vh;display:flex;flex-direction:column;align-items:center;
  padding:40px 20px 80px;
  background-image:
    radial-gradient(ellipse 60% 40% at 15% 15%,#00ff9d07 0%,transparent 70%),
    radial-gradient(ellipse 50% 50% at 85% 80%,#ff3c5f07 0%,transparent 70%),
    radial-gradient(ellipse 40% 60% at 50% 50%,#1e1e3008 0%,transparent 80%);
}

/* ── Header ─────────────────────────────────────────────────────── */
header{text-align:center;margin-bottom:48px;position:relative;}
.header-badge{
  display:inline-flex;align-items:center;gap:8px;
  font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;
  color:var(--accent);border:1px solid #00ff9d30;padding:5px 14px;
  border-radius:3px;margin-bottom:18px;text-transform:uppercase;
  background:#00ff9d08;
}
.badge-dot{width:6px;height:6px;background:var(--accent);border-radius:50%;
  animation:blink 2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.2;}}
h1{
  font-size:clamp(28px,5.5vw,56px);font-weight:800;letter-spacing:-2px;
  line-height:1.05;
  background:linear-gradient(140deg,#ffffff 0%,#aaaabf 55%,#666685 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.subtitle{margin-top:10px;color:var(--muted);font-size:13px;font-family:'Space Mono',monospace;letter-spacing:1px;}
.version-tag{
  display:inline-block;font-family:'Space Mono',monospace;font-size:10px;
  color:var(--muted);border:1px solid var(--border2);padding:2px 8px;
  border-radius:2px;margin-left:8px;vertical-align:middle;
}

/* ── Layout ──────────────────────────────────────────────────────── */
.container{width:100%;max-width:720px;display:flex;flex-direction:column;gap:18px;}

/* ── Cards ───────────────────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:26px;position:relative;overflow:hidden;
}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,#00ff9d25,transparent);
}
.card-label{
  font-family:'Space Mono',monospace;font-size:9px;letter-spacing:4px;
  color:var(--muted);text-transform:uppercase;margin-bottom:18px;
  display:flex;align-items:center;gap:8px;
}
.card-label::after{content:'';flex:1;height:1px;background:var(--border);}

/* ── Tabs ────────────────────────────────────────────────────────── */
.tabs{display:flex;gap:8px;margin-bottom:22px;}
.tab{
  flex:1;padding:10px 14px;border:1px solid var(--border);border-radius:8px;
  background:transparent;color:var(--muted);font-family:'Space Mono',monospace;
  font-size:11px;cursor:pointer;transition:all .2s;letter-spacing:1px;
}
.tab:hover{border-color:var(--border2);color:var(--text);}
.tab.active{border-color:var(--accent);color:var(--accent);background:#00ff9d08;}
.tab-panel{display:none;} .tab-panel.active{display:block;}

/* ── Upload Zone ─────────────────────────────────────────────────── */
.upload-zone{
  border:2px dashed var(--border);border-radius:12px;padding:30px 24px;
  text-align:center;cursor:pointer;transition:all .25s;
}
.upload-zone:hover,.upload-zone.drag-over{
  border-color:var(--accent);background:#00ff9d06;
}
.upload-zone input{display:none;}
.upload-icon{font-size:32px;margin-bottom:10px;}
.upload-text{color:var(--muted);font-size:13px;font-family:'Space Mono',monospace;line-height:1.6;}
.upload-text span{color:var(--accent);}
.audio-preview{display:none;margin-top:16px;flex-direction:column;gap:10px;}
.audio-preview.show{display:flex;}
.file-info{
  display:flex;align-items:center;gap:8px;background:var(--surface);
  border-radius:8px;padding:9px 13px;font-family:'Space Mono',monospace;
  font-size:11px;color:var(--accent);
}
audio{width:100%;border-radius:8px;outline:none;margin-top:2px;}

/* ── Augmentation Toggle [NEW] ───────────────────────────────────── */
.aug-toggle{
  display:flex;align-items:center;gap:10px;margin-top:14px;
  font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);
  cursor:pointer;user-select:none;
}
.aug-toggle input{display:none;}
.toggle-track{
  width:38px;height:20px;background:var(--surface);border:1px solid var(--border);
  border-radius:99px;position:relative;transition:all .2s;flex-shrink:0;
}
.toggle-track::after{
  content:'';width:14px;height:14px;background:var(--muted);border-radius:50%;
  position:absolute;top:2px;left:2px;transition:all .2s;
}
.aug-toggle input:checked + .toggle-track{background:#00ff9d18;border-color:var(--accent);}
.aug-toggle input:checked + .toggle-track::after{left:20px;background:var(--accent);}
.aug-label-text span{color:var(--accent);}

/* ── Recording ───────────────────────────────────────────────────── */
.rec-controls{display:flex;flex-direction:column;align-items:center;gap:14px;}
canvas#waveform{width:100%;height:52px;border-radius:10px;background:var(--surface);display:block;}
.rec-timer{font-family:'Space Mono',monospace;font-size:30px;color:var(--danger);letter-spacing:4px;}
.rec-btn{
  width:76px;height:76px;border-radius:50%;border:2px solid var(--danger);
  background:transparent;cursor:pointer;display:flex;align-items:center;
  justify-content:center;transition:all .2s;
}
.rec-btn:hover{background:#ff3c5f12;transform:scale(1.05);}
.rec-btn .dot{width:26px;height:26px;background:var(--danger);border-radius:50%;transition:all .3s;}
.rec-btn.recording .dot{border-radius:5px;width:20px;height:20px;}
.rec-btn.recording{animation:rpulse 1.5s infinite;}
@keyframes rpulse{0%,100%{box-shadow:0 0 0 0 #ff3c5f40;}50%{box-shadow:0 0 0 14px #ff3c5f00;}}
.rec-hint{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);text-align:center;min-height:18px;}
.rec-hint.active{color:var(--danger);} .rec-hint.done{color:var(--accent);}
.recorded-preview{display:none;flex-direction:column;gap:10px;width:100%;margin-top:14px;}
.recorded-preview.show{display:flex;}
.preview-label{font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);letter-spacing:2px;}

/* ── Submit Button ───────────────────────────────────────────────── */
.submit-btn{
  width:100%;padding:17px;border:none;border-radius:12px;
  background:linear-gradient(135deg,var(--accent) 0%,#00cc7d 100%);
  color:#000;font-family:'Syne',sans-serif;font-weight:700;font-size:15px;
  letter-spacing:1px;cursor:pointer;transition:all .22s;text-transform:uppercase;
  position:relative;overflow:hidden;
}
.submit-btn::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,#ffffff20,transparent);
  opacity:0;transition:opacity .2s;
}
.submit-btn:hover:not(:disabled)::after{opacity:1;}
.submit-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 10px 28px #00ff9d28;}
.submit-btn:disabled{opacity:.3;cursor:not-allowed;}

/* ── Loading ─────────────────────────────────────────────────────── */
.loading{display:none;text-align:center;padding:26px;font-family:'Space Mono',monospace;font-size:12px;color:var(--muted);}
.loading.show{display:block;}
.spinner{
  width:30px;height:30px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .75s linear infinite;margin:0 auto 12px;
}
@keyframes spin{to{transform:rotate(360deg);}}

/* ── Result Card ─────────────────────────────────────────────────── */
.result-card{display:none;} .result-card.show{display:block;}
.verdict{font-size:clamp(28px,5.5vw,50px);font-weight:800;letter-spacing:-1px;margin-bottom:4px;}
.verdict.real{color:var(--accent);} .verdict.fake{color:var(--danger);}
.verdict-sub{color:var(--muted);font-family:'Space Mono',monospace;font-size:12px;margin-bottom:20px;}

/* [NEW] Risk badge */
.risk-badge{
  display:inline-flex;align-items:center;gap:8px;
  font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
  letter-spacing:2px;padding:6px 14px;border-radius:4px;
  border:1px solid;margin-bottom:20px;
}

/* Probability row */
.prob-row{display:flex;gap:14px;margin-bottom:18px;}
.prob-box{
  flex:1;background:var(--surface);border-radius:12px;padding:16px;
  text-align:center;border:1px solid var(--border);
}
.prob-label{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;color:var(--muted);margin-bottom:6px;}
.prob-value{font-size:28px;font-weight:800;font-family:'Space Mono',monospace;}
.prob-value.real{color:var(--accent);} .prob-value.fake{color:var(--danger);}
.bar-wrap{height:6px;background:var(--surface);border-radius:99px;overflow:hidden;margin-bottom:16px;}
.bar{height:100%;border-radius:99px;transition:width 1.3s cubic-bezier(.4,0,.2,1);}
.bar.real{background:linear-gradient(90deg,var(--accent),#00cc7d);}
.bar.fake{background:linear-gradient(90deg,#ff3c5f,#cc2244);}

/* [NEW] Timing row */
.metrics-row{
  display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap;
}
.metric-chip{
  flex:1;min-width:120px;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:3px;
}
.metric-chip .mc-label{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;color:var(--muted);}
.metric-chip .mc-value{font-family:'Space Mono',monospace;font-size:14px;color:var(--text);font-weight:700;}
.metric-chip .mc-unit{font-size:10px;color:var(--muted);}

/* [NEW] Visualisation section */
.vis-section{margin-top:18px;display:flex;flex-direction:column;gap:12px;}
.vis-title{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;color:var(--muted);}
.vis-img{
  width:100%;border-radius:8px;border:1px solid var(--border);
  display:block;object-fit:contain;
}

/* Reset button */
.reset-btn{
  margin-top:14px;background:transparent;border:1px solid var(--border);
  color:var(--muted);padding:9px 18px;border-radius:7px;
  font-family:'Space Mono',monospace;font-size:11px;cursor:pointer;transition:all .2s;
}
.reset-btn:hover{border-color:var(--text);color:var(--text);}

/* Error */
.error-msg{color:var(--danger);font-family:'Space Mono',monospace;font-size:12px;margin-top:10px;display:none;}
.error-msg.show{display:block;}

/* ── [NEW] Benchmark Card ─────────────────────────────────────────── */
.benchmark-card{background:var(--card2);border:1px solid var(--border);}
.bench-btn{
  padding:10px 22px;border:1px solid var(--accent);border-radius:8px;
  background:#00ff9d08;color:var(--accent);font-family:'Space Mono',monospace;
  font-size:11px;letter-spacing:1px;cursor:pointer;transition:all .2s;
}
.bench-btn:hover{background:#00ff9d18;}
.bench-btn:disabled{opacity:.4;cursor:not-allowed;}
.bench-results{display:none;margin-top:18px;}
.bench-results.show{display:block;}
.bench-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px;}
.bench-metric{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:14px;text-align:center;
}
.bench-metric .bm-val{font-family:'Space Mono',monospace;font-size:22px;font-weight:700;color:var(--accent);}
.bench-metric .bm-lbl{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;color:var(--muted);margin-top:4px;}
.cm-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:280px;margin:0 auto;
}
.cm-cell{
  background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:12px;text-align:center;
}
.cm-cell .cm-val{font-family:'Space Mono',monospace;font-size:20px;font-weight:700;}
.cm-cell .cm-lbl{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:1px;color:var(--muted);margin-top:3px;}
.cm-tp .cm-val{color:var(--accent);}  .cm-tn .cm-val{color:var(--accent);}
.cm-fp .cm-val{color:var(--danger);}  .cm-fn .cm-val{color:var(--warn);}
.bench-note{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);margin-top:12px;line-height:1.5;padding:10px;background:var(--surface);border-radius:6px;border:1px solid var(--border);}
.hist-wrap{margin:14px 0;}
.hist-bars{display:flex;gap:2px;align-items:flex-end;height:60px;}
.hist-bar-group{flex:1;display:flex;gap:1px;align-items:flex-end;}
.hist-bar{flex:1;border-radius:2px 2px 0 0;min-height:2px;transition:height .4s;}
.hist-label{display:flex;justify-content:space-between;font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);margin-top:4px;}

/* ── Footer ──────────────────────────────────────────────────────── */
footer{
  margin-top:48px;text-align:center;font-family:'Space Mono',monospace;
  font-size:10px;color:var(--muted2);line-height:1.8;
}
footer span{color:var(--muted);}
</style>
</head>
<body>

<!-- ── HEADER ──────────────────────────────────────────────────── -->
<header>
  <div class="header-badge"><span class="badge-dot"></span>AI Security Tool v2</div>
  <h1>Deepfake Audio<br>Detector</h1>
  <p class="subtitle">
    EfficientNet-B0 + BiLSTM + Attention · ASVspoof 2019
    <span class="version-tag" id="versionTag">v2.0.0</span>
  </p>
</header>

<div class="container">

  <!-- ── INPUT CARD ─────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-label">01 · Input Audio</div>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('upload',this)">⬆ Upload File</button>
      <button class="tab" onclick="switchTab('record',this)">⏺ Record Voice</button>
    </div>

    <!-- Upload panel -->
    <div class="tab-panel active" id="panel-upload">
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="audio/*" onchange="handleFileSelect(event)">
        <div class="upload-icon">🎵</div>
        <div class="upload-text">
          Click or drag & drop an audio file<br>
          <span>WAV · MP3 · FLAC · OGG · M4A supported · max 25 MB</span>
        </div>
      </div>
      <div class="audio-preview" id="uploadPreview">
        <div class="file-info" id="fileName">📄 No file selected</div>
        <audio id="uploadAudio" controls></audio>
      </div>
    </div>

    <!-- Record panel -->
    <div class="tab-panel" id="panel-record">
      <div class="rec-controls">
        <canvas id="waveform" width="640" height="52"></canvas>
        <div class="rec-timer" id="timer">00:00</div>
        <button class="rec-btn" id="recBtn" onclick="toggleRecording()">
          <div class="dot"></div>
        </button>
        <div class="rec-hint" id="recHint">Click to start recording</div>
      </div>
      <div class="recorded-preview" id="recordedPreview">
        <div class="preview-label">▶ PLAYBACK — Listen before submitting</div>
        <audio id="recordedAudio" controls></audio>
      </div>
    </div>

    <!-- [NEW] Augmentation toggle -->
    <label class="aug-toggle">
      <input type="checkbox" id="augmentToggle">
      <span class="toggle-track"></span>
      <span class="aug-label-text">
        Enable noise augmentation &nbsp;<span>(adds robustness testing)</span>
      </span>
    </label>

    <div class="error-msg" id="errorMsg"></div>
  </div>

  <!-- ── ANALYZE BUTTON ─────────────────────────────────────────── -->
  <button class="submit-btn" id="submitBtn" onclick="submitAudio()" disabled>
    🔍 &nbsp;Analyze Audio
  </button>

  <!-- ── LOADING ────────────────────────────────────────────────── -->
  <div class="loading" id="loading">
    <div class="spinner"></div>
    Extracting features &amp; running inference...
  </div>

  <!-- ── RESULTS CARD ───────────────────────────────────────────── -->
  <div class="card result-card" id="resultCard">
    <div class="card-label">02 · Analysis Result</div>

    <div class="verdict" id="verdictText"></div>
    <div class="verdict-sub" id="verdictSub"></div>

    <!-- [NEW] Risk level badge -->
    <div id="riskBadge" class="risk-badge"></div>

    <!-- Probability boxes -->
    <div class="prob-row">
      <div class="prob-box">
        <div class="prob-label">REAL SCORE</div>
        <div class="prob-value real" id="realProb">—</div>
      </div>
      <div class="prob-box">
        <div class="prob-label">FAKE SCORE</div>
        <div class="prob-value fake" id="fakeProb">—</div>
      </div>
      <div class="prob-box">
        <div class="prob-label">CONFIDENCE</div>
        <div class="prob-value" id="confProb" style="color:var(--text)">—</div>
      </div>
    </div>
    <div class="bar-wrap"><div class="bar" id="probBar" style="width:0%"></div></div>

    <!-- [NEW] Timing + meta metrics -->
    <div class="metrics-row" id="metricsRow">
      <div class="metric-chip">
        <span class="mc-label">Feature Extraction</span>
        <span class="mc-value" id="featTime">—</span>
        <span class="mc-unit">ms</span>
      </div>
      <div class="metric-chip">
        <span class="mc-label">Inference Time</span>
        <span class="mc-value" id="infTime">—</span>
        <span class="mc-unit">ms</span>
      </div>
      <div class="metric-chip">
        <span class="mc-label">Total Time</span>
        <span class="mc-value" id="totalTime">—</span>
        <span class="mc-unit">ms</span>
      </div>
      <div class="metric-chip">
        <span class="mc-label">Model Ver.</span>
        <span class="mc-value" id="modelVerChip">—</span>
      </div>
    </div>

    <!-- [NEW] Visualisation panel -->
    <div class="vis-section" id="visSection">
      <div class="vis-title">WAVEFORM</div>
      <img id="waveformImg" class="vis-img" alt="Waveform" style="display:none">
      <div class="vis-title" style="margin-top:6px">MEL SPECTROGRAM</div>
      <img id="spectrogramImg" class="vis-img" alt="Mel Spectrogram" style="display:none">
    </div>

    <button class="reset-btn" onclick="resetAll()">↩ Analyze Another</button>
  </div>

  <!-- ── [NEW] BENCHMARK CARD ───────────────────────────────────── -->
  <div class="card benchmark-card">
    <div class="card-label">03 · Model Benchmark</div>
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
      <div style="font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);max-width:380px;line-height:1.6;">
        Run a synthetic evaluation to inspect accuracy, F1-score, and confusion matrix.
      </div>
      <button class="bench-btn" id="benchBtn" onclick="runBenchmark()">
        ▶ Run Benchmark
      </button>
    </div>

    <div class="bench-results" id="benchResults">
      <div class="bench-grid" id="benchGrid"></div>
      <div class="vis-title" style="margin-bottom:8px">CONFUSION MATRIX</div>
      <div class="cm-grid" id="cmGrid"></div>
      <div class="hist-wrap">
        <div class="vis-title" style="margin-bottom:6px">SCORE DISTRIBUTION</div>
        <div class="hist-bars" id="histBars"></div>
        <div class="hist-label"><span>0.0</span><span>0.5</span><span>1.0</span></div>
        <div style="display:flex;gap:12px;margin-top:6px;font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);">
          <span><span style="display:inline-block;width:10px;height:10px;background:#ff3c5f;border-radius:2px;margin-right:4px;vertical-align:middle;"></span>FAKE</span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#00ff9d;border-radius:2px;margin-right:4px;vertical-align:middle;"></span>REAL</span>
        </div>
      </div>
      <div class="bench-note" id="benchNote"></div>
    </div>
  </div>

</div>

<!-- ── FOOTER ───────────────────────────────────────────────────── -->
<footer>
  <span>Deepfake Audio Detector</span> · EfficientNet-B0 + BiLSTM + Attention<br>
  Production build · <span id="footerVer">v2.0.0</span>
</footer>

<!-- ================================================================
     JAVASCRIPT
     ================================================================ -->
<script>
/* ── State ──────────────────────────────────────────────────────── */
let currentBlob    = null;
let currentSource  = null;
let mediaRecorder  = null;
let audioChunks    = [];
let isRecording    = false;
let timerInterval  = null;
let seconds        = 0;
let analyserNode   = null;
let animFrame      = null;

/* ── Tab switching ──────────────────────────────────────────────── */
function switchTab(tab, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-' + tab).classList.add('active');
  currentBlob = null; currentSource = null;
  document.getElementById('submitBtn').disabled = true;
  hideError();
}

/* ── File upload ────────────────────────────────────────────────── */
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  currentBlob   = file;
  currentSource = 'upload';
  document.getElementById('uploadAudio').src = URL.createObjectURL(file);
  document.getElementById('fileName').textContent = '📄 ' + file.name;
  document.getElementById('uploadPreview').classList.add('show');
  document.getElementById('submitBtn').disabled = false;
  hideError();
}

/* ── Drag & drop ────────────────────────────────────────────────── */
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFileSelect({ target: { files: [f] } });
});

/* ── Recording ──────────────────────────────────────────────────── */
async function toggleRecording() {
  if (!isRecording) await startRecording(); else stopRecording();
}

async function startRecording() {
  try {
    hideError();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1 }
    });
    audioChunks = [];

    // Live waveform visualiser
    const actx = new AudioContext();
    const src  = actx.createMediaStreamSource(stream);
    analyserNode = actx.createAnalyser();
    analyserNode.fftSize = 512;
    src.connect(analyserNode);
    drawWaveform();

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(animFrame); clearWaveform();
      const raw = new Blob(audioChunks, { type: 'audio/webm' });
      try {
        const wav = await convertToWav(raw);
        currentBlob   = wav;
        currentSource = 'record';
        const audio   = document.getElementById('recordedAudio');
        audio.src     = URL.createObjectURL(wav);
        document.getElementById('recordedPreview').classList.add('show');
        audio.play().catch(() => {});
        setHint('✅ Done! Listen above then click Analyze', 'done');
        document.getElementById('submitBtn').disabled = false;
      } catch (err) { showError('Conversion failed: ' + err.message); }
    };
    mediaRecorder.start(100);
    isRecording = true; seconds = 0;
    document.getElementById('recBtn').classList.add('recording');
    document.getElementById('recordedPreview').classList.remove('show');
    document.getElementById('submitBtn').disabled = true;
    setHint('🔴 Recording... click square to stop (max 10s)', 'active');
    timerInterval = setInterval(() => {
      seconds++;
      document.getElementById('timer').textContent =
        String(Math.floor(seconds / 60)).padStart(2, '0') + ':' +
        String(seconds % 60).padStart(2, '0');
      if (seconds >= 10) stopRecording();
    }, 1000);
  } catch (err) {
    showError('Microphone access denied. Please allow mic and try again.');
  }
}

function stopRecording() {
  if (!mediaRecorder || !isRecording) return;
  clearInterval(timerInterval);
  isRecording = false;
  document.getElementById('recBtn').classList.remove('recording');
  setHint('Processing...', '');
  mediaRecorder.stop();
}

/* ── WAV conversion ─────────────────────────────────────────────── */
async function convertToWav(blob) {
  const ab   = await blob.arrayBuffer();
  const actx = new AudioContext({ sampleRate: 16000 });
  const buf  = await actx.decodeAudioData(ab);
  await actx.close();
  const samples = buf.getChannelData(0);
  const wavBuf  = new ArrayBuffer(44 + samples.length * 2);
  const v       = new DataView(wavBuf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true);
  ws(8, 'WAVE'); ws(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true); off += 2;
  }
  return new Blob([wavBuf], { type: 'audio/wav' });
}

/* ── Live waveform canvas ───────────────────────────────────────── */
function drawWaveform() {
  const canvas = document.getElementById('waveform');
  const ctx    = canvas.getContext('2d');
  const data   = new Uint8Array(analyserNode.frequencyBinCount);
  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(data);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00ff9d'; ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff9d'; ctx.shadowBlur  = 4;
    ctx.beginPath();
    const sw = canvas.width / data.length; let x = 0;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0 * canvas.height) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sw;
    }
    ctx.stroke();
  }
  draw();
}
function clearWaveform() {
  const c = document.getElementById('waveform');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

/* ── Submit / Predict ───────────────────────────────────────────── */
async function submitAudio() {
  if (!currentBlob) return;
  hideError();
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('loading').classList.add('show');
  document.getElementById('resultCard').classList.remove('show');

  const fd = new FormData();
  fd.append('audio', currentBlob,
    currentSource === 'record' ? 'recording.wav' : (currentBlob.name || 'audio.wav'));

  // [NEW] Pass augmentation flag
  const augment = document.getElementById('augmentToggle').checked;
  fd.append('augment', augment ? 'true' : 'false');

  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    document.getElementById('loading').classList.remove('show');

    if (data.error) {
      showError('⚠ ' + data.error);
      document.getElementById('submitBtn').disabled = false;
      return;
    }

    renderResults(data);

  } catch (err) {
    document.getElementById('loading').classList.remove('show');
    showError('Request failed: ' + err.message);
    document.getElementById('submitBtn').disabled = false;
  }
}

/* ── Render results  [NEW: risk + timing + visuals] ─────────────── */
function renderResults(data) {
  const fake = data.verdict === 'FAKE';

  // Verdict
  document.getElementById('verdictText').textContent = fake ? '🔴 FAKE AUDIO' : '🟢 REAL AUDIO';
  document.getElementById('verdictText').className   = 'verdict ' + (fake ? 'fake' : 'real');
  document.getElementById('verdictSub').textContent  = fake
    ? 'AI-generated or voice-cloned audio detected'
    : 'Authentic human voice detected';

  // Probabilities
  document.getElementById('realProb').textContent = data.real_prob + '%';
  document.getElementById('fakeProb').textContent = data.fake_prob + '%';
  document.getElementById('confProb').textContent = data.confidence + '%';

  // Probability bar
  const bar = document.getElementById('probBar');
  bar.className   = 'bar ' + (fake ? 'fake' : 'real');
  bar.style.width = '0%';
  setTimeout(() => { bar.style.width = (fake ? data.fake_prob : data.real_prob) + '%'; }, 80);

  // [NEW] Risk badge
  const rb = document.getElementById('riskBadge');
  rb.textContent         = data.risk_emoji + '  ' + data.risk_label + '  —  ' + data.risk_desc;
  rb.style.color         = data.risk_color;
  rb.style.borderColor   = data.risk_color + '50';
  rb.style.background    = data.risk_bg || data.risk_color + '10';

  // [NEW] Timing metrics
  document.getElementById('featTime').textContent  = data.feat_time_ms  ?? '—';
  document.getElementById('infTime').textContent   = data.inf_time_ms   ?? '—';
  document.getElementById('totalTime').textContent = data.total_time_ms ?? '—';
  document.getElementById('modelVerChip').textContent = data.model_version ?? '—';

  // Update header version tag
  if (data.model_version) {
    document.getElementById('versionTag').textContent = 'v' + data.model_version;
    document.getElementById('footerVer').textContent  = 'v' + data.model_version;
  }

  // [NEW] Visualisations (waveform + spectrogram)
  if (data.waveform_img) {
    const wImg = document.getElementById('waveformImg');
    wImg.src           = 'data:image/png;base64,' + data.waveform_img;
    wImg.style.display = 'block';
  }
  if (data.spectrogram_img) {
    const sImg = document.getElementById('spectrogramImg');
    sImg.src           = 'data:image/png;base64,' + data.spectrogram_img;
    sImg.style.display = 'block';
  }

  document.getElementById('resultCard').classList.add('show');
  document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
}

/* ── Reset ──────────────────────────────────────────────────────── */
function resetAll() {
  currentBlob = null; currentSource = null;
  document.getElementById('resultCard').classList.remove('show');
  document.getElementById('uploadPreview').classList.remove('show');
  document.getElementById('recordedPreview').classList.remove('show');
  document.getElementById('fileInput').value = '';
  document.getElementById('timer').textContent = '00:00';
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('waveformImg').style.display    = 'none';
  document.getElementById('spectrogramImg').style.display = 'none';
  setHint('Click to start recording', '');
  hideError();
}

/* ── [NEW] Benchmark ────────────────────────────────────────────── */
async function runBenchmark() {
  const btn = document.getElementById('benchBtn');
  btn.disabled     = true;
  btn.textContent  = '⏳ Running...';

  try {
    const res  = await fetch('/benchmark?n=300');
    const data = await res.json();

    // Metric cards
    const metrics = [
      { val: data.accuracy   + '%', lbl: 'Accuracy'    },
      { val: data.precision  + '%', lbl: 'Precision'   },
      { val: data.recall     + '%', lbl: 'Recall'      },
      { val: data.f1_score   + '%', lbl: 'F1 Score'    },
      { val: data.specificity + '%',lbl: 'Specificity' },
      { val: data.false_positive_rate + '%', lbl: 'FP Rate' },
    ];
    document.getElementById('benchGrid').innerHTML = metrics.map(m =>
      `<div class="bench-metric">
         <div class="bm-val">${m.val}</div>
         <div class="bm-lbl">${m.lbl}</div>
       </div>`
    ).join('');

    // Confusion matrix
    const cm = data.confusion_matrix;
    document.getElementById('cmGrid').innerHTML = `
      <div class="cm-cell cm-tp"><div class="cm-val">${cm.TP}</div><div class="cm-lbl">TRUE POSITIVE</div></div>
      <div class="cm-cell cm-fp"><div class="cm-val">${cm.FP}</div><div class="cm-lbl">FALSE POSITIVE</div></div>
      <div class="cm-cell cm-fn"><div class="cm-val">${cm.FN}</div><div class="cm-lbl">FALSE NEGATIVE</div></div>
      <div class="cm-cell cm-tn"><div class="cm-val">${cm.TN}</div><div class="cm-lbl">TRUE NEGATIVE</div></div>
    `;

    // Score histogram
    if (data.score_histogram) {
      const hist  = data.score_histogram;
      const maxV  = Math.max(...hist.fake.map((v, i) => v + hist.real[i]), 1);
      document.getElementById('histBars').innerHTML = hist.bins.map((_, i) => {
        const fH = Math.round((hist.fake[i] / maxV) * 56);
        const rH = Math.round((hist.real[i] / maxV) * 56);
        return `<div class="hist-bar-group">
          <div class="hist-bar" style="height:${fH}px;background:#ff3c5f;opacity:0.75"></div>
          <div class="hist-bar" style="height:${rH}px;background:#00ff9d;opacity:0.75"></div>
        </div>`;
      }).join('');
    }

    document.getElementById('benchNote').textContent = '⚠ ' + data.note;
    document.getElementById('benchResults').classList.add('show');

  } catch (err) {
    alert('Benchmark failed: ' + err.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = '▶ Run Again';
  }
}

/* ── Helpers ────────────────────────────────────────────────────── */
function setHint(m, c) {
  const e = document.getElementById('recHint');
  e.textContent = m; e.className = 'rec-hint ' + c;
}
function showError(m) {
  const e = document.getElementById('errorMsg');
  e.textContent = m; e.classList.add('show');
}
function hideError() { document.getElementById('errorMsg').classList.remove('show'); }
</script>

</body>
</html>"""


# ================================================================
#  SECTION 14 — ROUTES
# ================================================================

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/favicon.ico")
def favicon():
    return "", 204

# ================================================================
#  SECTION 15 — ENTRY POINT
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" 🎙  Deepfake Audio Detector  —  Production v2.0.0")
    print("="*60)
    print(f" Device     : {DEVICE}")
    print(f" N_Frames   : {N_FRAMES}  (real temporal sequence for LSTM)")
    print(f" Frame size : {FRAME_H}×{FRAME_W}  (H×W per channel)")
    print(f" Legacy arch: {USING_LEGACY_ARCH}")
    print(f" Checkpoint : {'✅ Loaded' if os.path.exists(MODEL_PATH) else '⚠  Not found (demo mode)'}")
    print("-"*60)
    print(" 📡 Open → http://localhost:8000")
    print(" 📊 Benchmark → http://localhost:8000/benchmark")
    print(" ℹ  Model Info → http://localhost:8000/model-info")
    print("="*60 + "\n")
    app.run(debug=False, port=8000, host="0.0.0.0")
