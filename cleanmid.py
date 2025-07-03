import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset" / "jazz_audio_fft_paths.csv"   # ← adapte si besoin
SEARCH_ROOT = BASE_DIR / "dataset"                             # où sont tes .mid

def main(apply: bool = False) -> None:
    if not CSV_PATH.is_file():
        print(f"⚠ CSV introuvable : {CSV_PATH}")
        return
    if not SEARCH_ROOT.is_dir():
        print(f"⚠ Dossier introuvable : {SEARCH_ROOT}")
        return

    # Ensemble des noms (basename) à conserver
    keep = {
        Path(name).name              # on ignore le dossier, on garde juste <file>.mid
        for name in pd.read_csv(CSV_PATH)["Name"]
        if str(name).lower().endswith(".mid")
    }

    # Tous les .mid présents (récursif)
    to_delete = [
        p for p in SEARCH_ROOT.rglob("*.mid")
        if p.name not in keep
    ]

    if not apply:
        print(f"{len(to_delete)} fichiers .mid seraient supprimés.")
        for p in to_delete[:10]:
            print("  -", p.relative_to(BASE_DIR))
        if len(to_delete) > 10:
            print("  ...")
        print("\nRelance avec --apply pour supprimer réellement.")
        return

    removed = 0
    for p in to_delete:
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"⚠ Impossible de supprimer {p}: {e}")

    print(f"✓ {removed} fichiers .mid supprimés.")

if __name__ == "__main__":
    main(apply="--apply" in sys.argv)