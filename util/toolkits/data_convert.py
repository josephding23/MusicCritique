import matplotlib.pyplot as plt
import pretty_midi
import math

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


def generate_midi_segment_from_tensor(data, path):
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
                        instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                        during_note = False
            else:
                if not during_note:
                    continue
                else:
                    note_end = time * quarter_length
                    instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                    during_note = False
    pm.instruments.append(instr_track)
    pm.write(path)


def save_midis(bars, path):
    pm = pretty_midi.PrettyMIDI()

    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 24)),
                                  bars,
                                  np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 20))),
                                 axis=3)
    padded_bars = padded_bars.reshape((-1, padded_bars.shape[1], padded_bars.shape[2], padded_bars.shape[3]))
    print(padded_bars.shape)
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[1]):
        padded_bars_list.append(padded_bars[:, ch_idx, :, :].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[2],
                                                                     padded_bars.shape[3]))

    pianoroll = padded_bars_list[0]
    pianoroll = pianoroll.reshape((pianoroll.shape[0] * pianoroll.shape[1], pianoroll.shape[2]))
    pianoroll_diff = np.concatenate((np.zeros((1, 128), dtype=int), pianoroll, np.zeros((1, 128), dtype=int)))
    pianoroll_search = np.diff(pianoroll_diff.astype(int), axis=0)
    print(pianoroll_search.shape)

    instrument = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')

    tempo = 120
    beat_resolution = 4

    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * pianoroll.shape[0]

    for note_num in range(128):
        start_idx = (pianoroll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))

        end_idx = (pianoroll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))

        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        t.append(j)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])

        start_time = temp_start_time
        end_time = temp_end_time
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx], end=end_time[idx])
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                instrument.notes.append(note)
    instrument.notes.sort(key=lambda note: note.start)

    pm.instruments.append(instrument)
    pm.write(path)


def generate_whole_midi_from_tensor(data, path):
    pm = pretty_midi.PrettyMIDI()

    segment_num = data.shape[0]

    instr_track = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')
    quarter_length = 60 / 120 / 4
    note_range = 84
    time_step = 64

    reshaped = np.reshape(data, (segment_num * time_step, 84))

    for note in range(note_range):
        during_note = False
        note_begin = 0

        for time in range(time_step * segment_num):
            has_note = reshaped[time, note] >= 0.5

            if has_note:
                if not during_note:
                    during_note = True
                    note_begin = time * quarter_length
                else:
                    if time != time_step-1:
                        continue
                    else:
                        note_end = time * quarter_length
                        instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                        during_note = False
            else:
                if not during_note:
                    continue
                else:
                    note_end = time * quarter_length
                    instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                    during_note = False
    pm.instruments.append(instr_track)
    pm.write(path)


def generate_data_from_midi(path):
    pm = pretty_midi.PrettyMIDI(path)
    note_range = (24, 108)
    segment_num = math.ceil(pm.get_end_time() / 8)
    data = np.zeros(shape=(segment_num, 64, 84), dtype=np.float)

    # data = np.zeros((segment_num, 64, 84), np.bool_)
    quarter_length = 60 / 120 / 4
    for instr in pm.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                start = int(round(note.start / quarter_length))
                end = int(round(note.end / quarter_length))
                pitch = note.pitch
                if pitch < note_range[0] or pitch >= note_range[1]:
                    continue
                else:
                    pitch -= 24
                    for time_raw in range(start, end):
                        segment = int(time_raw / 64)
                        time = time_raw % 64
                        data[(segment, time, pitch)] = 1.0
    return data


if __name__ == '__main__':
    pass
