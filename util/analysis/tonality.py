from util.toolkits.database import *
import numpy as np
import matplotlib.pyplot as plt


def evaluate_tonal_scale_of_data(whole_data):
    # should consider minor
    note_range = 84
    time_step = 64
    tonal_distance = [0, 2, 4, 5, 7, 9, 11]
    in_tone_notes = 0
    outta_tone_notes = 0
    for i in range(whole_data.shape[0]):
        data = whole_data[i, :, :]
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

def evaluate_tonal_scale_of_data_advanced(whole_data):
    # should consider minor
    note_range = 84
    time_step = 64
    tonal_distance = [0, 2, 4, 5, 7, 9, 11]
    in_tone_notes = 0
    outta_tone_notes = 0

    for i in range(whole_data.shape[0]):
        data = whole_data[i, :, :]
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


def get_mode_type_of_song(md5):
    midi_collection = get_midi_collection()
    return midi_collection.find_one({'md5': md5})['KeySignature']['Mode']


def evaluate_tonal_scale_of_file(npy_path, mode_type):
    npy_file = np.load(npy_path)
    data = npy_file['arr_0']
    # print(data.shape)
    # should consider minor
    if mode_type == 'major':
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


def evaluate_all_other_tonality():
    root_dir = 'E:/jazz_midkar/npy_files'
    midi_collection = get_jazzkar_collection()
    midi_collection.update_many({}, {'$unset': {'TonalityDegree': ''}})
    for midi in midi_collection.find({'TonalityDegree': {'$exists': False}}):
        md5 = midi['md5']
        tonal_type = midi['KeySignature']['Mode']
        path = root_dir + '/' + md5 + '.npz'

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


def get_genre_tonality():
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        name = genre['Name']
        qualified_num = midi_collection.count({'Genre': name, 'TonalityDegree': {'$gte': 0.5}})
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

    plt.rcParams['xtick.labelsize'] = 9
    plt.bar(range(len(tonality_list)), tonality_list, width=0.7)
    plt.ylabel('Music files num')
    plt.xlabel('In tune notes ratio (%)')
    plt.xticks([i for i in range(10)], x_info)
    plt.show()


if __name__ == '__main__':
    # get_midi_collection().update_many({}, {'$unset': {'TonalityDegree': ''}})
    get_genre_tonality()
