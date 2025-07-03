"""
generate_music_spectro.py
=========================
Auto-encodeur CNN sur |STFT| (256 frames × 513 fréquences).
• Par défaut : charge autoenc_spectro.weights.h5 et génère 3 morceaux.
• Lancez --train UNE fois pour créer les poids.
"""

import argparse, pathlib, sys, numpy as np, pandas as pd, librosa
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.io.wavfile as wavfile

# ────── paramètres CLI ──────
ap = argparse.ArgumentParser()
ap.add_argument("--wav_dir", default="dataset/Jazz_Audio_concurrent")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--latent", type=int, default=128)
ap.add_argument("--frames", type=int, default=256)
ap.add_argument("--target_sec", type=float, default=25)
ap.add_argument("--out", default="out_wav")
ap.add_argument("--weights", default="autoenc_spectro.weights.h5")
ap.add_argument("--train", action="store_true")
args = ap.parse_args()

WAV_DIR  = pathlib.Path(args.wav_dir)
OUT_DIR  = pathlib.Path(args.out); OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS  = pathlib.Path(args.weights)
AMP_FILE = WEIGHTS.with_suffix(".max.npy")

# ────── STFT ──────
SR, N_FFT, HOP = 44100, 1024, 256
FREQ_BINS      = N_FFT // 2 + 1          # 513
FRAME_COUNT    = args.frames             # 256  ≈ 1,49 s

def wav_to_mag(path: pathlib.Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT))
    if S.shape[1] >= FRAME_COUNT:
        S = S[:, :FRAME_COUNT]
    else:
        pad = np.zeros((FREQ_BINS, FRAME_COUNT), dtype="float32")
        pad[:, :S.shape[1]] = S
        S = pad
    return S.T.astype("float32")           # (time, freq)

# ────── modèle ──────
def build_cnn(frames: int, freqs: int, latent: int):
    enc = models.Sequential([
        layers.Input(shape=(frames, freqs, 1)),
        layers.Conv2D(32, (5,5), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (5,5), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(latent, name="latent")
    ], name="encoder")

    dec = models.Sequential([
        layers.Input(shape=(latent,)),
        layers.Dense((frames//4)*(freqs//4)*64, activation="relu"),
        layers.Reshape((frames//4, freqs//4, 64)),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(64, (5,5), activation="relu", padding="same"),
        layers.UpSampling2D((2,2)),
        layers.Conv2D(1, (5,5), activation="sigmoid", padding="same"),
        layers.ZeroPadding2D(padding=((0,0), (0,1)))   # +1 col. → 513
    ], name="decoder")

    inp = layers.Input(shape=(frames, freqs, 1))
    model = models.Model(inp, dec(enc(inp)), name="autoenc")
    model.compile(optimizer="adam", loss="mse")
    return model, dec

auto, decoder = build_cnn(FRAME_COUNT, FREQ_BINS, args.latent)

# ────── entraînement optionnel ──────
if args.train:
    wav_files = list(WAV_DIR.rglob("*.wav"))
    if not wav_files:
        print("Aucun WAV dans", WAV_DIR); sys.exit(1)
    mags = [wav_to_mag(p) for p in wav_files]
    X = np.stack(mags)                    # (N,256,513)
    GLOBAL_MAX = float(X.max())
    X = (X / (GLOBAL_MAX + 1e-8))[..., None]  # (N,256,513,1)

    idx = np.random.permutation(len(X))
    split = int(0.9*len(X))
    auto.fit(X[idx[:split]], X[idx[:split]],
             validation_data=(X[idx[split:]], X[idx[split:]]),
             epochs=args.epochs, batch_size=8)
    auto.save_weights(WEIGHTS)
    np.save(AMP_FILE, GLOBAL_MAX)
    print("✓ Poids sauvegardés :", WEIGHTS)
else:
    if not WEIGHTS.is_file():
        print(f"⚠ Poids absents ({WEIGHTS}). Lancez --train une fois."); sys.exit(0)
    auto.load_weights(WEIGHTS)
    if AMP_FILE.is_file():
        GLOBAL_MAX = float(np.load(AMP_FILE))
    else:
        sample = list(WAV_DIR.rglob("*.wav"))[:10]
        GLOBAL_MAX = max(wav_to_mag(p).max() for p in sample)
    print("✓ Poids chargés :", WEIGHTS)

# ────── génération ──────
seg_sec = FRAME_COUNT * HOP / SR
segments = max(1, int(np.ceil(args.target_sec / seg_sec)))
print(f"\nGénération de 3 morceaux (~{args.target_sec}s)…\n")
for i in range(3):
    parts = []
    for _ in range(segments):
        z = np.random.normal(size=(1, args.latent))
        mag = decoder.predict(z, verbose=0)[0, ..., 0] * GLOBAL_MAX
        wav = librosa.griffinlim(mag.T, hop_length=HOP, win_length=N_FFT)
        parts.append(wav)
    full = np.concatenate(parts)[: int(args.target_sec*SR)]
    full = full / (np.max(np.abs(full)) + 1e-8)
    wavfile.write(OUT_DIR/f"gen_{i}.wav", SR, (full*32767).astype("int16"))
    print("✓", OUT_DIR/f"gen_{i}.wav")

print("\nTerminé — fichiers dans", OUT_DIR)

