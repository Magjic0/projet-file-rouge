"""
delete_unused_music.py
----------------------
Nettoie automatiquement :

1. Les fichiers .mid dans dataset/Jazz Midi/
2. Les fichiers .wav dans dataset/Jazz_Audio_concurrent/  (récursif)

… en se basant sur les morceaux listés dans
dataset/jazz_instrument_fft_paths.csv  (colonne Name).

Usage
-----
python delete_unused_music.py          # dry-run (affiche seulement)
python delete_unused_music.py --apply  # supprime réellement
"""

import sys
from pathlib import Path
import pandas as pd

# ───────── Chemins ───────── #
BASE        = Path(__file__).resolve().parent
CSV_PATH    = BASE / "dataset" / "jazz_instrument_fft_paths.csv"
MIDI_DIR    = BASE / "dataset" / "Jazz Midi"
AUDIO_DIR   = BASE / "dataset" / "Jazz_Audio_concurrent"

def main(apply: bool = False) -> None:
    # Vérifications rapides
    if not CSV_PATH.is_file():
        print(f"⚠ CSV introuvable : {CSV_PATH}")
        return
    if not MIDI_DIR.is_dir():
        print(f"⚠ Dossier MIDI introuvable : {MIDI_DIR}")
        return
    if not AUDIO_DIR.is_dir():
        print(f"⚠ Dossier audio introuvable : {AUDIO_DIR}")
        return

    df = pd.read_csv(CSV_PATH)

    # Jeux de noms à conserver
    keep_midis = {Path(n).name for n in df["Name"] if str(n).lower().endswith(".mid")}
    keep_wavs  = {Path(n).stem + ".wav" for n in keep_midis}

    # ----------- MIDI à supprimer -----------
    midi_to_del = [p for p in MIDI_DIR.glob("*.mid") if p.name not in keep_midis]

    # ----------- WAV à supprimer ------------
    wav_to_del = [p for p in AUDIO_DIR.rglob("*.wav") if p.name not in keep_wavs]

    if not apply:
        print(f"{len(midi_to_del)} fichiers .mid seraient supprimés.")
        for p in midi_to_del[:8]:
            print("  -", p.relative_to(BASE))
        if len(midi_to_del) > 8:
            print("  ...")

        print(f"\n{len(wav_to_del)} fichiers .wav seraient supprimés.")
        for p in wav_to_del[:8]:
            print("  -", p.relative_to(BASE))
        if len(wav_to_del) > 8:
            print("  ...")

        print("\nRelance avec  --apply  pour supprimer réellement.")
        return

    # Suppression effective
    removed_mid, removed_wav = 0, 0
    for p in midi_to_del:
        try:
            p.unlink()
            removed_mid += 1
        except Exception as e:
            print(f"⚠ Impossible de supprimer {p}: {e}")

    for p in wav_to_del:
        try:
            p.unlink()
            removed_wav += 1
        except Exception as e:
            print(f"⚠ Impossible de supprimer {p}: {e}")

    print(f"✓ {removed_mid} .mid et {removed_wav} .wav supprimés.")

if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
