"""
generate_music_cnn.py  –  sans ré-entraînement par défaut
=========================================================
• Charge autoenc.weights.h5 et génère 3 morceaux (~target_sec s)
• Si les poids manquent, le script s’arrête (invite à --train)
• Si l’amplitude .max.npy manque, elle est recalculée sur 10 FFT

Une seule phase d’entraînement :
    python generate_music_cnn.py --train --n 120 --epochs 60
Puis génération rapide :
    python generate_music_cnn.py
"""

import argparse, pathlib, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.io.wavfile as wavfile

# ────────── CLI ────────── #
p = argparse.ArgumentParser()
p.add_argument("--csv", default="dataset/jazz_audio_fft_paths.csv")
p.add_argument("--n", type=int, default=100, help="FFT pour l'entraînement")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--latent", type=int, default=64)
p.add_argument("--fft_len", type=int, default=32768)
p.add_argument("--target_sec", type=float, default=25)
p.add_argument("--out", default="out_wav")
p.add_argument("--weights", default="autoenc.weights.h5")
p.add_argument("--train", action="store_true", help="entraîne/ré-entraîne")
args = p.parse_args()

WEIGHTS = pathlib.Path(args.weights)
OUT_DIR = pathlib.Path(args.out); OUT_DIR.mkdir(parents=True, exist_ok=True)
FFT_LEN = args.fft_len

# ────────── Modèle ────────── #
def build_autoenc(fft_len, latent):
    enc = models.Sequential([
        layers.Input(shape=(fft_len,1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPool1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPool1D(2),
        layers.Flatten(),
        layers.Dense(latent, name="latent")
    ], name="encoder")
    dec = models.Sequential([
        layers.Input(shape=(latent,)),
        layers.Dense(fft_len//4*64, activation="relu"),
        layers.Reshape((fft_len//4,64)),
        layers.UpSampling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.UpSampling1D(2),
        layers.Conv1D(1, 5, activation="sigmoid", padding="same")
    ], name="decoder")
    inp = layers.Input(shape=(fft_len,1))
    model = models.Model(inp, dec(enc(inp)), name="autoenc")
    model.compile(optimizer="adam", loss="mse")
    return model, dec

auto, decoder = build_autoenc(FFT_LEN, args.latent)

# ────────── Entraînement (optionnel) ────────── #
if args.train:
    df = pd.read_csv(args.csv).head(args.n)
    arrs = []
    for pth in df["FFT_Npy_Path"]:
        x = np.load(pth).astype("float32")
        x = x[:FFT_LEN] if x.shape[0] >= FFT_LEN else np.pad(x, (0, FFT_LEN-x.shape[0]))
        arrs.append(x)
    X = np.stack(arrs)
    GLOBAL_MAX = float(X.max())
    X = X / (GLOBAL_MAX + 1e-8)
    X = X[..., None]

    idx = np.random.permutation(len(X))
    split = int(0.9*len(X))
    auto.fit(X[idx[:split]], X[idx[:split]],
             validation_data=(X[idx[split:]], X[idx[split:]]),
             epochs=args.epochs, batch_size=8)
    auto.save_weights(WEIGHTS)
    np.save(WEIGHTS.with_suffix(".max.npy"), GLOBAL_MAX)
    print("✓ Poids sauvegardés :", WEIGHTS)
else:
    # --------- mode génération ---------
    if not WEIGHTS.is_file():
        print(f"⚠ Aucun poids trouvé ({WEIGHTS}). Lancez une fois avec --train.")
        sys.exit(0)
    auto.load_weights(WEIGHTS)

    max_path = WEIGHTS.with_suffix(".max.npy")
    if max_path.is_file():
        GLOBAL_MAX = float(np.load(max_path))
    else:
        print("⚠ .max.npy manquant – recalcul sur 10 FFT…")
        sample = pd.read_csv(args.csv)["FFT_Npy_Path"].head(10)
        GLOBAL_MAX = max(np.load(p).max() for p in sample)
    print("✓ Poids chargés :", WEIGHTS, "| GLOBAL_MAX =", GLOBAL_MAX)

# ────────── Génération ────────── #
SR = 44100
seg_sec = (FFT_LEN*2) / SR
segments = max(1, int(np.ceil(args.target_sec / seg_sec)))
for i in range(3):
    pieces = []
    for _ in range(segments):
        z = np.random.normal(size=(1, args.latent))
        spec = decoder.predict(z, verbose=0)[0,:,0] * GLOBAL_MAX
        spec = np.maximum(spec, 0)
        wav = np.fft.irfft(spec)
        wav = wav / (wav.max() + 1e-8)
        pieces.append(wav)
    full = np.concatenate(pieces)[: int(args.target_sec*SR)]
    wavfile.write(OUT_DIR/f"gen_{i}.wav", SR, (full*32767).astype("int16"))
    print("✓", OUT_DIR/f"gen_{i}.wav")

print("\nTerminé — fichiers dans", OUT_DIR)
