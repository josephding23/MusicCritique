import librosa

# filename = librosa.util.example_audio_file()
# filename = '../data/roundabout.mp3'
filename = '../data/roundabout_early.mp3'

y, sr = librosa.load(filename)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

beat_times = librosa.frames_to_time(beat_frames, sr=sr)