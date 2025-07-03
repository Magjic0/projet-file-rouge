"""
extract_instrument.py  –  all-tracks version (patch Path→str)
============================================================
• Parcourt dataset/Jazz Midi/*.mid
• Rend toutes les pistes contenant des notes
• Sauvegarde WAV & FFT par programme GM (0-127) ou 'drums'
• Renseigne dataset/jazz_instrument_fft_paths.csv
"""

import sys, signal, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pretty_midi
import scipy.io.wavfile as wavfile

# ───────── Configuration ───────── #
BASE = Path(__file__).resolve().parent
MIDI_DIR = BASE / "dataset" / "Jazz Midi"          # dossier .mid
WAV_ROOT = BASE / "dataset" / "Instrument_Audio"
FFT_ROOT = BASE / "dataset" / "Instrument_FFT"
CSV_PATH = BASE / "dataset" / "jazz_instrument_fft_paths.csv"

# ─────────── Worker ─────────── #
def process(midi_path: Path):
    rows = []
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))      # ← conversion Path → str
    except Exception as exc:
        return None, f"{midi_path.name} | load : {exc}"

    for inst in pm.instruments:
        if not inst.notes:
            continue

        prog = "drums" if inst.is_drum else inst.program

        # Mini-MIDI pour cette piste
        mini = pretty_midi.PrettyMIDI()
        m_inst = pretty_midi.Instrument(inst.program, inst.is_drum)
        m_inst.notes = inst.notes
        mini.instruments.append(m_inst)

        try:
            audio = mini.fluidsynth(fs=44100)
            audio_i16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
        except Exception as exc:
            return None, f"{midi_path.name} | render ({prog}) : {exc}"

        wav_dir = WAV_ROOT / str(prog)
        fft_dir = FFT_ROOT / str(prog)
        wav_dir.mkdir(parents=True, exist_ok=True)
        fft_dir.mkdir(parents=True, exist_ok=True)

        stem = midi_path.stem + f"_p{prog}"
        wav_path = wav_dir / f"{stem}.wav"
        fft_path = fft_dir / f"{stem}_fft.npy"

        try:
            wavfile.write(wav_path, 44100, audio_i16)
            np.save(fft_path, np.abs(np.fft.rfft(audio_i16)))
        except Exception as exc:
            return None, f"{midi_path.name} | save ({prog}) : {exc}"

        rows.append({
            "Name": midi_path.name,
            "Program": prog,
            "Wav_Path": str(wav_path),
            "FFT_Npy_Path": str(fft_path)
        })
    return rows, None

# ───────── CSV helpers ───────── #
def load_csv():
    if CSV_PATH.is_file():
        df = pd.read_csv(CSV_PATH)
        done = {(n, p) for n, p in zip(df["Name"], df["Program"])}
    else:
        df = pd.DataFrame(columns=["Name", "Program", "Wav_Path", "FFT_Npy_Path"])
        done = set()
    return df, done

def save_csv(df):
    df.to_csv(CSV_PATH, index=False)
    print(f"💾 CSV sauvegardé ({len(df)} lignes) → {CSV_PATH}")

# ─────────── Main ─────────── #
def main():
    df, done = load_csv()
    mids = list(MIDI_DIR.glob("*.mid"))
    todo = [p for p in mids if all((p.name, prog) not in done for prog in range(128))]

    if not todo:
        print("✅ Aucun nouveau MIDI à traiter.")
        return

    with ProcessPoolExecutor(mp.cpu_count()) as pool:
        fut2 = {pool.submit(process, p): p for p in todo}
        for idx, fut in enumerate(as_completed(fut2), 1):
            rows, err = fut.result()
            if err:
                print("⚠", err)
                continue
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            if idx % 10 == 0:
                save_csv(df)

    save_csv(df)
    print("✓ Extraction terminée.")

# ───────── Signals ───────── #
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

if __name__ == "__main__":
    mp.freeze_support()
    main()
