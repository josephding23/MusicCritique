import numpy as np
import matplotlib.pyplot as plt

from presets import Preset

import librosa as _librosa
import librosa.display as _display

_librosa.display = _display

librosa = Preset(_librosa)
librosa['sr'] = 44100
librosa['hop_length'] = 1024
librosa['n_fft'] = 4096

filename = '../music/roundabout.mp3'
y, sr = librosa.load(filename, duration=5, offset=35)

M = librosa.feature.melspectrogram(y=y)
M_highres = librosa.feature.melspectrogram(y=y, hop_length=512)

plt.figure(figsize=(6, 6))

ax = plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                         y_axis='mel', x_axis='time')
plt.title('44100/1024/4096')

plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
librosa.display.specshow(librosa.power_to_db(M_highres, ref=np.max),
                         hop_length=512,
                         y_axis='mel', x_axis='time')
plt.title('44100/512/4096')

librosa['sr'] = 11025
y2, sr2 = librosa.load(filename, duration=5, offset=35)
M2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
librosa.display.specshow(librosa.power_to_db(M2, ref=np.max),
                         y_axis='mel', x_axis='time')
plt.title('11025/1024/4096')

plt.tight_layout()
plt.show()

