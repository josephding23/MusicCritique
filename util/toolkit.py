import matplotlib.pyplot as plt
import pretty_midi
import numpy as np

def plot_data(data):
    sample_data = data
    dataX = []
    dataY = []
    for time in range(64):
        for pitch in range(84):
            if sample_data[time][pitch] > 0.5:
                dataX.append(time)
                dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()


def generate_midi_from_data(data, path):
    pm = pretty_midi.PrettyMIDI()
    instr_track = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')
    quarter_length = 60 / 120 / 4
    note_range = 84
    time_step = 64

    for note in range(note_range):
        during_note = False
        note_begin = 0
        for time in range(time_step):
            has_note = data[time, note] >= 0.5

            if has_note:
                if not during_note:
                    during_note = True
                    note_begin = time * quarter_length
                else:
                    if time != time_step-1:
                        continue
                    else:
                        note_end = time * quarter_length
                        instr_track.notes.append(pretty_midi.Note(127, note + 12, note_begin, note_end))
                        during_note = False
            else:
                if not during_note:
                    continue
                else:
                    note_end = time * quarter_length
                    instr_track.notes.append(pretty_midi.Note(64, note + 12, note_begin, note_end))
                    during_note = False
    pm.instruments.append(instr_track)
    pm.write(path)

def evaluate_tonal_scale(data):
    note_range = 84
    time_step = 64
    tonal_distance = [0, 2, 4, 5, 7, 9, 11]
    in_tone_notes = 0
    outta_tone_notes = 0
    for note in range(note_range):
        for time in range(time_step):
            has_note = data[time, note] >= 0.5
            if has_note:
                if note % 12 in tonal_distance:
                    in_tone_notes += 1
                else:
                    outta_tone_notes += 1
    tonality = in_tone_notes / (in_tone_notes + outta_tone_notes)
    return tonality

if __name__ == '__main__':
    path = './data_A2B.npy'
    save_path = './dataA2B.mid'
    data = np.load(path)
    print(data.shape)
    plot_data(data)
    generate_midi_from_data(data, save_path)