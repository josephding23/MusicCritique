import matplotlib.pyplot as plt
import pretty_midi
import math
from pymongo import MongoClient

from util.data import auxillary as aux
import numpy as np
import music21


def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi


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
                    instr_track.notes.append(pretty_midi.Note(64, note + 24, note_begin, note_end))
                    during_note = False
    pm.instruments.append(instr_track)
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
                start = int(note.start / quarter_length)
                end = int(note.end / quarter_length)
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


def evaluate_tonal_scale(data):

    # should consider minor
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


def get_md5_of(performer, song, genre):
    midi_collection = aux.get_midi_collection()
    try:
        md5 = midi_collection.find_one({'Performer': performer, 'Name': song, 'Genre': genre})['md5']
        return md5
    except Exception:
        raise Exception('No midi Found.')


def get_chord(note_nums):
    notes = []
    for note_num in note_nums:
        name = pretty_midi.note_number_to_name(note_num)
        note = music21.note.pitch.Pitch(name)
        notes.append(note)
    # print(notes)
    chord = music21.chord.Chord(notes)
    # chord.duration = music21.duration.Duration(length)
    return chord


def evaluate_midi_chord(song='21 Guns', performer='Green Day', genre='rock'):
    root_dir = 'E:/free_midi_library/transposed_midi'

    try:
        md5 = get_md5_of(performer, song, genre)
        original_path = root_dir + '/' + genre + '/' + md5 + '.mid'
        print(original_path)
    except Exception as e:
        print(e)
        return

    key_mode = get_midi_collection().find_one({'md5': md5})['KeySignature']['Mode']
    if key_mode == 'major':
        current_key = music21.key.Key('C')
    else:
        current_key = music21.key.Key('a')
    ori_data = generate_data_from_midi(original_path)
    data_piece = ori_data[0, :, :]
    # plot_data(data=ori_data[0, :, :])

    notes = []
    for part in range(ori_data.shape[0]):
        print(f'Part {part}:')
        data_piece = ori_data[part, :, :]
        for time in range(64):
            if time == 0:
                old_notes = []
            else:
                old_notes = notes
            notes = []
            for note in range(84):
                if data_piece[time][note] != 0:
                    notes.append(note + 24)

            if len(notes) == 0:
                notes = old_notes
            elif notes == old_notes:
                continue
            else:
                try:
                    chord = get_chord(notes)
                    print(current_key.getScaleDegreeAndAccidentalFromPitch(chord.root()), chord.commonName)
                except Exception as e:
                    print(e.__traceback__)
                    return
        print()


if __name__ == '__main__':
    evaluate_midi_chord()
