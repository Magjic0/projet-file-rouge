"""
delete_unused_data.py
---------------------
Supprime tous les fichiers audio/FFT/MIDI NON référencés dans
dataset/jazz_instrument_fft_paths.csv.

Cibles nettoyées :
    • dataset/Jazz Midi/*.mid
    • dataset/Jazz_Audio_concurrent/*.wav
    • dataset/Jazz_Audio_concurrent/freq_npy/*.npy
    • dataset/Jazz_Audio_concurrent/fft_npy/**/*.npy
    • dataset/Instrument_Audio/*/*.wav
    • dataset/Instrument_FFT/*/*_fft.npy

Usage
-----
python delete_unused_data.py          # aperçu (dry-run)
python delete_unused_data.py --apply  # suppression réelle
"""

import sys, re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV  = BASE / "dataset" / "jazz_instrument_fft_paths.csv"

DIRS = {
    "midi": BASE / "dataset" / "Jazz Midi",
    "wav_mix": BASE / "dataset" / "Jazz_Audio_concurrent",
    "freq": BASE / "dataset" / "Jazz_Audio_concurrent" / "freq_npy",
    "fft": BASE / "dataset" / "Jazz_Audio_concurrent" / "fft_npy",
    "inst_wav": BASE / "dataset" / "Instrument_Audio",
    "inst_fft": BASE / "dataset" / "Instrument_FFT",
}

def main(apply=False):
    if not CSV.is_file():
        print("⚠ CSV introuvable :", CSV); sys.exit(1)

    keep_stems = {Path(n).stem for n in pd.read_csv(CSV)["Name"]}

    def stem_ok(stem: str) -> bool:
        """True si le début du stem correspond à un morceau conservé."""
        base = stem.split("_")[0]   # pour les fichiers _pXX ou _fft
        return base in keep_stems

    to_del = []

    # --- MIDI ---
    to_del += [p for p in DIRS["midi"].glob("*.mid") if p.stem not in keep_stems]

    # --- mix WAV ---
    to_del += [p for p in DIRS["wav_mix"].glob("*.wav") if p.stem not in keep_stems]

    # --- freq npy ---
    to_del += [p for p in DIRS["freq"].glob("*.npy") if not stem_ok(p.stem)]

    # --- fft npy (récursif) ---
    to_del += [p for p in DIRS["fft"].rglob("*.npy") if not stem_ok(p.stem.split("_fft")[0])]

    # --- instrument WAVs & FFTs ---
    for root in ("inst_wav", "inst_fft"):
        if DIRS[root].is_dir():
            to_del += [p for p in DIRS[root].rglob("*.*") if p.is_file() and not stem_ok(p.stem.split("_p")[0])]

    # -------- dry-run / delete --------
    if not apply:
        print(f"{len(to_del)} fichiers seraient supprimés.")
        for p in to_del[:15]:
            print("  -", p.relative_to(BASE))
        if len(to_del) > 15: print("  ...")
        print("\nLancez : python delete_unused_data.py --apply  pour confirmer.")
        return

    removed = 0
    for p in to_del:
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print("⚠", p, ":", e)
    print(f"✓ {removed} fichiers supprimés.")

if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
