import os
import numpy as np
import pandas as pd
import pretty_midi
import scipy.io.wavfile as wavfile

midi_folder = './dataset/Jazz Midi'
csv_metadata_path = './dataset/Jazz-midi.csv'
output_audio_folder = './dataset/Jazz_Audio'
os.makedirs(output_audio_folder, exist_ok=True)

metadata_df = pd.read_csv(csv_metadata_path)
data_list = []

for midi_file in os.listdir(midi_folder):
    if midi_file.lower().endswith('.mid'):
        midi_path = os.path.join(midi_folder, midi_file)
        
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            audio = pm.fluidsynth(fs=44100)
        except Exception as e:
            print(f"Erreur avec {midi_file}: {e}")
            continue  # Passe au fichier suivant
        
        audio_norm = np.int16(audio / np.max(np.abs(audio)) * 32767)
        wav_filename = midi_file.replace('.mid', '.wav')
        wav_path = os.path.join(output_audio_folder, wav_filename)
        wavfile.write(wav_path, 44100, audio_norm)
        
        audio_freqs = audio_norm.astype(float).tolist()
        
        meta_row = metadata_df[metadata_df['Name'] == midi_file]
        if not meta_row.empty:
            notes = meta_row['Notes'].values[0]
            len_sequence = meta_row['Len_Sequence'].values[0]
            unique_notes = meta_row['Unique_notes'].values[0]
        else:
            notes = None
            len_sequence = None
            unique_notes = None
        
        data_list.append({
            'Name': midi_file,
            'Notes': notes,
            'Len_Sequence': len_sequence,
            'Unique_notes': unique_notes,
            'Frequencies': audio_freqs
        })

final_df = pd.DataFrame(data_list)
final_csv_path = './dataset/jazz_audio_frequencies.csv'
final_df.to_csv(final_csv_path, index=False)

print(f"Dataset enregistré à : {final_csv_path}")

