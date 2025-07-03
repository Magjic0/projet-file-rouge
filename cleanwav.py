import os
import sys
import pandas as pd

CSV_PATH = "./dataset/jazz_audio_fft_paths.csv"
AUDIO_DIR = "./dataset/Jazz_Audio_concurrent"

def main(apply: bool = False) -> None:
    # Vérification du CSV
    if not os.path.isfile(CSV_PATH):
        print(f"CSV introuvable : {CSV_PATH}")
        sys.exit(1)

    # Ensemble des .wav à conserver
    df = pd.read_csv(CSV_PATH)
    keep_set = {name.replace(".mid", ".wav") for name in df["Name"]}

    # Fichiers .wav présents
    to_delete = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav") and f not in keep_set
    ]

    if not apply:
        print(f"{len(to_delete)} fichiers .wav pourraient être supprimés.")
        for f in to_delete[:10]:
            print("  -", f)
        if len(to_delete) > 10:
            print("  ...")
        print("\nLancez : python deletewav.py --apply  pour supprimer.")
        return

    # Suppression effective
    removed = 0
    for f in to_delete:
        path = os.path.join(AUDIO_DIR, f)
        try:
            os.remove(path)
            removed += 1
        except Exception as exc:
            print(f"⚠ Impossible de supprimer {f} : {exc}")

    print(f"✓ {removed} fichiers .wav supprimés.")

if __name__ == "__main__":
    apply_flag = "--apply" in sys.argv
    main(apply=apply_flag)