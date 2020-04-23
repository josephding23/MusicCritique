import matplotlib.pyplot as plt
import pretty_midi
import math
from pymongo import MongoClient

from util.data import auxillary as aux
import numpy as np
import music21
from util.toolkit import *

def evaluate_tonal_scale_of_data(data):
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


def evaluate_tonal_scale_of_file(npy_path, type):
    npy_file = np.load(npy_path)
    data = npy_file['arr_0']
    # print(data.shape)
    # should consider minor
    if type == 'major':
        tonal_distance = [0, 2, 4, 5, 7, 9, 11]
        root_note = 0
    else:
        tonal_distance = [0, 2, 3, 5, 7, 8, 10]
        root_note = 9
    in_tone_notes = 0
    outta_tone_notes = 0
    for i in range(data.shape[0]):
        paragraph, time, note = data[i, :]
        if (note - root_note) % 12 in tonal_distance:
            in_tone_notes += 1
        else:
            outta_tone_notes += 1
    tonality = in_tone_notes / (in_tone_notes + outta_tone_notes)
    return tonality


def evaluate_all_tonality():
    root_dir = 'E:/midi_matrix/one_instr'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'TonalityDegree': {'$exists': False}}):
        md5 = midi['md5']
        genre = midi['Genre']
        tonal_type = midi['KeySignature']['Mode']
        path = root_dir + '/' + genre + '/' + md5 + '.npz'

        try:
            tonality = evaluate_tonal_scale_of_file(path, tonal_type)
            print(tonality)

            midi_collection.update_one(
                {'_id': midi['_id']},
                {'$set': {'TonalityDegree': tonality}}
            )

            print('Progress: {:.2%}\n'.format(midi_collection.count({'TonalityDegree': {'$exists': True}}) / midi_collection.count()))
        except:
            print(path, midi['Name'])


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


def get_genre_tonality():
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        name = genre['Name']
        qualified_num = midi_collection.count({'Genre': name, 'TonalityDegree': {'$gte': 0.6}})
        whole_num = midi_collection.count({'Genre': name})
        qualified_percent = qualified_num / whole_num
        print(name, qualified_percent)


def print_overall_tonality():
    tonality_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_info = ['0~10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~100']
    midi_collection = get_midi_collection()
    classical_collection = get_classical_collection()
    jazz_collection = get_jazz_collection()
    jazzkar_collection = get_jazzkar_collection()

    for collection in [midi_collection, classical_collection, jazz_collection, jazzkar_collection]:
        for midi in collection.find({'TonalityDegree': {'$exists': True}}):
            tonality = midi['TonalityDegree'] * 100
            if tonality == 100:
                tonality_list[9] += 1
            else:
                tonality_list[int(tonality / 10)] += 1

    plt.bar(range(len(tonality_list)), tonality_list)
    # plt.xlabel('符合调性音符比例')
    plt.xticks([i for i in range(10)], x_info)
    plt.show()


if __name__ == '__main__':
    # get_midi_collection().update_many({}, {'$unset': {'TonalityDegree': ''}})
    print_overall_tonality()
