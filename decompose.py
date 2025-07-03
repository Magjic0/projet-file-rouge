
"""
decompose.py
============
Décompose chaque fichier audio brut (.npy) en magnitude de série de Fourier.

Structure attendue (créée par ton précédent script) :
dataset/
└─ Jazz_Audio_concurrent/
   ├─ freq_npy/          # fichiers <nom>.npy contenant les échantillons int16
   └─ ...

Le script :
1. Parcourt `freq_npy/`
2. Calcule la FFT avec `np.fft.rfft` (partie réelle positive)
3. Sauvegarde la magnitude dans `fft_npy/<nom>_fft.npy`
4. Construit un CSV `jazz_audio_fft_paths.csv` reliant le morceau (.mid) au chemin de son FFT.

Pour lancer :
    python decompose.py
"""

import os
import numpy as np
import pandas as pd

# ---------------- Répertoires ---------------- #
NPY_DIR = './dataset/Jazz_Audio_concurrent/freq_npy'
FFT_DIR = './dataset/Jazz_Audio_concurrent/fft_npy'
OUT_CSV = './dataset/jazz_audio_fft_paths.csv'

os.makedirs(FFT_DIR, exist_ok=True)

rows = []

for fname in sorted(os.listdir(NPY_DIR)):
    if not fname.endswith('.npy'):
        continue

    npy_path = os.path.join(NPY_DIR, fname)
    print(f'Décomposition FFT de {fname}…')

    # Charger l'audio (int16) et convertir en float32
    audio = np.load(npy_path).astype(np.float32)

    # FFT rapide (uniquement partie réelle positive)
    fft_vals = np.fft.rfft(audio)
    magnitude = np.abs(fft_vals)

    # Sauvegarder la magnitude
    fft_name = fname.replace('.npy', '_fft.npy')
    fft_path = os.path.join(FFT_DIR, fft_name)
    np.save(fft_path, magnitude)

    # Ligne pour le CSV
    rows.append({
        'Name': fname.replace('.npy', '.mid'),
        'FFT_Npy_Path': fft_path
    })

# CSV récapitulatif
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'✓ Séries de Fourier sauvegardées. CSV → {OUT_CSV}')