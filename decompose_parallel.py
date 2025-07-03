
"""
decompose_parallel.py
---------------------
Version « turbo » : décomposition FFT ultra‑rapide, inspirée de ton
script concurrent robuste.

• Exploite tous les cœurs CPU (ProcessPoolExecutor).
• Chaque worker :
    1. Charge le .npy (échantillons int16) → float32.
    2. Calcule np.fft.rfft (magnitude).
    3. Sauvegarde <nom>_fft.npy dans fft_npy/.
    4. Retourne un petit dict : nom, chemin FFT.

• Pas de gros objets picklés.
• Checkpoint CSV toutes les 20 pistes.
• Fallback séquentiel si le pool plante.
• Reprise automatique si le CSV existe déjà.

Usage :
    python decompose_parallel.py
"""

import os, sys, signal, multiprocessing as mp
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ───────── Config ───────── #
NPY_DIR = './dataset/Jazz_Audio_concurrent/freq_npy'
FFT_DIR = './dataset/Jazz_Audio_concurrent/fft_npy'
OUT_CSV = './dataset/jazz_audio_fft_paths.csv'

os.makedirs(FFT_DIR, exist_ok=True)

# ───────── Worker ───────── #
def fft_worker(npy_file: str):
    """Calcule la magnitude FFT et la sauvegarde. Renvoie (dict, err)."""
    try:
        path = os.path.join(NPY_DIR, npy_file)
        audio = np.load(path).astype(np.float32)
        magnitude = np.abs(np.fft.rfft(audio))
        fft_name = npy_file.replace('.npy', '_fft.npy')
        fft_path = os.path.join(FFT_DIR, fft_name)
        np.save(fft_path, magnitude)
        return {'Name': npy_file.replace('.npy', '.mid'),
                'FFT_Npy_Path': fft_path}, None
    except Exception as exc:
        return None, f"{npy_file} | {exc}"

# ───────── Utils ───────── #
def save_csv(df):
    df.to_csv(OUT_CSV, index=False)
    print(f'💾 Checkpoint : {len(df)} FFTs → {OUT_CSV}')

def sequential(files, df):
    for f in files:
        if f.replace('.npy', '.mid') in df['Name'].values:
            continue
        res, err = fft_worker(f)
        if err:
            print('⚠', err)
            continue
        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
        if len(df) % 20 == 0:
            save_csv(df)
    return df

# ───────── Main ───────── #
def main():
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        done = set(df['Name'])
    else:
        df = pd.DataFrame(columns=['Name', 'FFT_Npy_Path'])
        done = set()

    todo = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy') and f.replace('.npy', '.mid') not in done]
    if not todo:
        print('✅ Tous les fichiers sont déjà décomposés.')
        return

    try:
        with ProcessPoolExecutor(mp.cpu_count()) as pool:
            fut2file = {pool.submit(fft_worker, f): f for f in todo}
            for idx, fut in enumerate(as_completed(fut2file), 1):
                try:
                    res, err = fut.result()
                except Exception as exc:
                    file_name = fut2file[fut]
                    print(f'⚠ Worker crash sur {file_name}: {exc}')
                    continue

                if err:
                    print('⚠', err)
                    continue

                df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
                if idx % 20 == 0:
                    save_csv(df)
    except Exception as pool_exc:
        print(f'❌ Problème avec le pool : {pool_exc}. Passage en séquentiel.')
        remaining = [f for f in todo if f.replace('.npy', '.mid') not in df['Name'].values]
        df = sequential(remaining, df)

    save_csv(df)
    print('✓ FFT terminé.')

# ───────── Signals ───────── #
def _sig_handler(sig, frame):
    print('\nInterruption clavier – sauvegarde avant sortie…')
    if "df" in globals():
        save_csv(df)  # type: ignore
    sys.exit(0)

if __name__ == '__main__':
    mp.freeze_support()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    main()
