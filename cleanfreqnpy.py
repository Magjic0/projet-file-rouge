import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset" / "jazz_audio_frequencies.csv"
FREQ_DIR = BASE_DIR / "dataset" / "Jazz_Audio_concurrent" / "freq_npy"

def main(apply: bool = False) -> None:
    if not CSV_PATH.is_file() or not FREQ_DIR.is_dir():
        print("⚠ Chemins invalides :")
        print("  CSV :", CSV_PATH)
        print("  FREQ :", FREQ_DIR)
        return

    # Ensemble des .npy à conserver (on garde juste le nom de fichier)
    keep_set = {Path(p).name for p in pd.read_csv(CSV_PATH)["Freq_Npy_Path"]}

    # Tous les *.npy dans freq_npy (récursif)
    to_delete = [p for p in FREQ_DIR.rglob("*.npy") if p.name not in keep_set]

    if not apply:
        print(f"{len(to_delete)} fichiers *_freq.npy pourraient être supprimés.")
        for p in to_delete[:10]:
            print("  -", p.relative_to(BASE_DIR))
        if len(to_delete) > 10:
            print("  ...")
        print("\nRelance avec  --apply  pour supprimer définitivement.")
        return

    removed = 0
    for p in to_delete:
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"⚠ Impossible de supprimer {p} : {e}")

    print(f"✓ {removed} fichiers *_freq.npy supprimés.")

if __name__ == "__main__":
    main(apply="--apply" in sys.argv)