import pretty_midi

pm = pretty_midi.PrettyMIDI('./dataset/Jazz Midi/2ndMovementOfSinisterFootwear.mid')
audio = pm.fluidsynth(fs=44100)