
"""
midi_to_wav_concurrent_robust_v2.py
-----------------------------------
• NE dépend PLUS de `BrokenProcessPool` (absent avant Python 3.12).
• Si le pool se casse (ou n’importe quelle autre exception globale),
  le code repasse en **séquentiel** pour les fichiers restants.
• Toujours : fréquences écrites en .npy, CSV léger, checkpoints réguliers.
"""

import os, sys, signal, traceback, multiprocessing as mp
import numpy as np
import pandas as pd
import pretty_midi
import scipy.io.wavfile as wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Config ---------------- #
MIDI_FOLDER = './dataset/Jazz Midi'
METADATA_CSV = './dataset/Jazz-midi.csv'
OUT_DIR = './dataset/Jazz_Audio_concurrent'
WAV_DIR = OUT_DIR
NPY_DIR = os.path.join(OUT_DIR, 'freq_npy')
OUT_CSV = './dataset/jazz_audio_frequencies.csv'

os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

metadata_df = pd.read_csv(METADATA_CSV)

# --------------- Worker --------------- #
def worker(midi_file):
    """Traite un fichier, renvoie (row_dict, error_str)"""
    try:
        midi_path = os.path.join(MIDI_FOLDER, midi_file)
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=44100)
        audio_i16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    except Exception as exc:
        return None, f"{midi_file} | Lecture/FluidSynth: {exc} ({type(exc)})"

    wav_name = midi_file.replace('.mid', '.wav')
    wav_path = os.path.join(WAV_DIR, wav_name)
    npy_name = midi_file.replace('.mid', '.npy')
    npy_path = os.path.join(NPY_DIR, npy_name)
    try:
        wavfile.write(wav_path, 44100, audio_i16)
        np.save(npy_path, audio_i16)
    except Exception as exc:
        return None, f"{midi_file} | Sauvegarde: {exc} ({type(exc)})"

    meta = metadata_df[metadata_df['Name'] == midi_file]
    if meta.empty:
        notes = length = uniq = None
    else:
        notes = meta['Notes'].values[0]
        length = meta['Len_Sequence'].values[0]
        uniq = meta['Unique_notes'].values[0]

    return {
        'Name': midi_file,
        'Notes': notes,
        'Len_Sequence': length,
        'Unique_notes': uniq,
        'Wav_Path': wav_path,
        'Freq_Npy_Path': npy_path
    }, None

# --------------- Utils --------------- #
def save_csv(df):
    df.to_csv(OUT_CSV, index=False)
    print(f'💾 Checkpoint ({len(df)} pistes) → {OUT_CSV}')

def sequential(files, df):
    for f in files:
        if f in df['Name'].values:
            continue
        res, err = worker(f)
        if err:
            print('⚠', err)
            continue
        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
        if len(df) % 10 == 0:
            save_csv(df)
    return df

# --------------- Main --------------- #
def main():
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        done = set(df['Name'])
    else:
        df = pd.DataFrame(columns=[
            'Name','Notes','Len_Sequence','Unique_notes','Wav_Path','Freq_Npy_Path'
        ])
        done = set()

    todo = [f for f in os.listdir(MIDI_FOLDER) if f.lower().endswith('.mid') and f not in done]
    if not todo:
        print('✅ Rien à faire.')
        return

    try:
        with ProcessPoolExecutor(mp.cpu_count()) as pool:
            fut2file = {pool.submit(worker, f): f for f in todo}
            for idx, fut in enumerate(as_completed(fut2file), 1):
                try:
                    res, err = fut.result()
                except Exception as exc:
                    # Le futur a levé une exception non interceptée : log + continue
                    file_name = fut2file[fut]
                    print(f'⚠ Worker crash sur {file_name}: {exc} ({type(exc)})')
                    continue

                if err:
                    print('⚠', err)
                    continue

                df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
                if idx % 10 == 0:
                    save_csv(df)
    except Exception as pool_exc:
        # N'importe quelle exception globale (inclut un pool brisé)
        print(f'❌ Problème avec le pool ({pool_exc} - {type(pool_exc)}).')
        remaining = [f for f in todo if f not in df['Name'].values]
        print('⏩ Basculons en séquentiel pour les fichiers restants.')
        df = sequential(remaining, df)

    save_csv(df)
    print('✓ Terminé.')

# ----------- Signals ---------- #
def _sig_handler(sig, frame):
    print('\nInterruption clavier – sauvegarde avant sortie…')
    if 'df' in globals():
        save_csv(df)  # type: ignore
    sys.exit(0)

if __name__ == '__main__':
    mp.freeze_support()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    main()
