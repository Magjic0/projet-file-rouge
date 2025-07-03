import os
import sys
import pandas as pd

FREQ_CSV = "./dataset/jazz_audio_frequencies.csv"
FFT_CSV  = "./dataset/jazz_audio_fft_paths.csv"

def main(apply: bool = False) -> None:
    if not (os.path.isfile(FREQ_CSV) and os.path.isfile(FFT_CSV)):
        print("CSV manquant ! Vérifie les chemins.")
        sys.exit(1)

    freq_df = pd.read_csv(FREQ_CSV)
    fft_names = set(pd.read_csv(FFT_CSV)["Name"])

    mask_keep = freq_df["Name"].isin(fft_names)
    to_drop = len(freq_df) - mask_keep.sum()

    if not apply:
        print(f"{to_drop} lignes seraient supprimées sur {len(freq_df)}.")
        print("Lance : python cleanfreq.py --apply  pour appliquer.")
        return

    freq_df = freq_df[mask_keep]
    freq_df.to_csv(FREQ_CSV, index=False)
    print(f"✓ CSV nettoyé : {to_drop} lignes supprimées, {len(freq_df)} restantes.")

if __name__ == "__main__":
    main(apply="--apply" in sys.argv)