from pymongo import MongoClient
import os
import pretty_midi
import pypianoroll
import traceback
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import scipy.sparse as ss

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres

def generate_multi_instr_numpy(time_step=120, bar_length=4, valid_range = (24, 108), genre='rock'):
    root_dir = 'E:/merged_midi/'
    npy_file_root_Dir = 'E:/midi_matrix/' + genre + '/'

    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}, 'MultiInstrNpyGenerated': False}, no_cursor_timeout = True):
        non_zeros = []
        path = root_dir + genre + '/' + midi['md5'] + '.mid'
        mult = pypianoroll.parse(path)
        instr_tracks = {
            'Drums': None,
            'Piano': None,
            'Guitar': None,
            'Bass': None,
            'Strings': None
        }
        length = 0
        for track in mult.tracks:
            length = track.pianoroll.shape[0]
            valid_matrix = track.pianoroll[:, valid_range[0]:valid_range[1]].copy()
            instr_tracks[track.name] = valid_matrix
        for name, content in instr_tracks.items():
            if content is None:
                instr_tracks[name] = np.zeros((length, valid_range[1]-valid_range[0]))


        merged_instr_matrix = np.dstack((instr_tracks['Drums'], instr_tracks['Piano'],
                                         instr_tracks['Guitar'], instr_tracks['Bass'],
                                         instr_tracks['Strings']))
        whole_paragraphs = length // (time_step * bar_length) + 1
        try:
            for track_num in range(5):
                instr_track = merged_instr_matrix[:, :, track_num]
                for current_time in range(length):
                    for note in range(valid_range[1]-valid_range[0]):
                        if instr_track[current_time][note] != 0:
                            paragraph_num = current_time // (time_step * bar_length)
                            bar_num = current_time % (time_step * bar_length) // time_step
                            time_node = current_time % time_step

                            non_zeros.append([paragraph_num, bar_num, time_node, note, track_num])

            non_zero_temp_matrix = np.array(non_zeros).transpose()
            print(non_zero_temp_matrix.shape)
            save_path = npy_file_root_Dir + midi['md5'] + '.npz'
            np.savez_compressed(save_path, non_zero_temp_matrix)
            # last_segment_number += whole_paragraphs
            # print(last_segment_number)
            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'MultiInstrNpyGenerated': True}})
            # prossed += 1
            print('Progress: {:.2%}\n'.format(midi_collection.count({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}, 'MultiInstrNpyGenerated': True}) / midi_collection.count({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}})))
        except:
            pass
        # print(merged.shape)


def divide_into_groups(group_num=10):
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        total_amount = midi_collection.count({'Genre': genre['Name']})
        num_per_group = math.ceil(total_amount // group_num)
        current_num = 0
        for midi in midi_collection.find({'Genre': genre['Name']},
                                         no_cursor_timeout=True):
            group = current_num // num_per_group
            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'DataGroup': group}})
            current_num += 1

def test_multiple_genres_midi():
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    one_genre_num = midi_collection.count({'TotalGenres': ['pop', 'rock']})
    print(one_genre_num / midi_collection.count())



def merge_all_sparse_matrices():
    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    root_dir = 'E:/midi_matrix/one_instr/'

    time_step = 64
    valid_range = (24, 108)


    for genre in genre_collection.find({'DatasetGenerated': False}):
        save_dir = 'd:/data/' + genre['Name']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print(genre['Name'])
        whole_length = genre['PiecesNum']
        shape = np.array([whole_length, time_step, valid_range[1]-valid_range[0]])

        processed = 0
        last_piece_num = 0
        whole_num = midi_collection.count({'Genre': genre['Name']})

        non_zeros = []
        for midi in midi_collection.find({'Genre': genre['Name']}, no_cursor_timeout=True):

            path = root_dir + genre['Name'] + '/' + midi['md5'] + '.npz'
            pieces_num = midi['PiecesNum']

            f = np.load(path)
            matrix = f['arr_0'].copy()
            print(pieces_num, matrix.shape[0])
            for data in matrix:
                try:
                    data = data.tolist()
                    piece_order = last_piece_num + data[0]
                    non_zeros.append([piece_order, data[1], data[2]])
                except:
                    print(path)

            last_piece_num += pieces_num
            processed += 1

            print('Progress: {:.2%}\n'.format(processed / whole_num))

        non_zeros = np.array(non_zeros)
        print(non_zeros.shape)
        np.savez_compressed(save_dir + '/data_sparse' + '.npz', nonzeros=non_zeros, shape=shape)

        genre_collection.update_one({'_id': genre['_id']}, {'$set': {'DatasetGenerated': True}})



def print_all_genres_num():
    genres_collection = get_genre_collection()
    midi_collection = get_midi_collection()

    for genre in genres_collection.find():
        whole_num = 0
        for midi in midi_collection.find({'Genre': genre['Name']}):
           whole_num += midi['PiecesNum']
        genres_collection.update_one(
            {'_id': genre['_id']},
            {'$set': {'PiecesNum': whole_num}}
        )
        print(genre['Name'], whole_num)

def get_genre_pieces(genre):
    genres_collection = get_genre_collection()

    pieces_num = genres_collection.find_one({'Name': genre})

    return pieces_num['PiecesNum']['IgnoreLessThan4']


def generate_data_from_sparse_data(root_dir='d:/data', genre='rock', parts_num=10):
    shape = np.load(root_dir + '/' +genre + '/shape.npy')
    result = np.zeros(shape, np.bool_)
    paths = [root_dir+ '/'  + genre + '/data_sparse_part' + str(num) + '.npz' for num in range(parts_num)]


    for path in paths:
        with np.load(path) as npfile:
            print(path)
            sparse = npfile['nonzeros']
            for data in sparse:
                # print(data)
                result[data] = True
    print(result.shape)
    return result


def find_data_with_no_empty_tracks():
    root_dir = 'E:/merged_midi/'
    total = 0
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'NotEmptyTracksNum': {'$exists': False}}):
        instr_tracks = {
            'Drums': None,
            'Piano': None,
            'Guitar': None,
            'Bass': None,
            'Strings': None
        }
        num = 0
        try:
            path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
            mult = pypianoroll.parse(path)
            for track in mult.tracks:
                num += 1
            midi_collection.update_one(
                {'_id': midi['_id']},
                {'$set': {'NotEmptyTracksNum': num}}
            )
            print('Progress: {:.2%}\n'.format(midi_collection.count({'NotEmptyTracksNum': {'$exists': True}}) / midi_collection.count()))
        except:
            total += 1
            # midi_collection.delete_one({'_id': midi['_id']})
    print(total)

def add_paragraph_num_info(time_step=120, bar_length=4, valid_range = (24, 108)):
    root_dir = 'E:/merged_midi/'

    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'PiecesNum': {'$exists': False}}, no_cursor_timeout = True):
        path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
        mult = pypianoroll.parse(path)
        instr_tracks = {
            'Drums': None,
            'Piano': None,
            'Guitar': None,
            'Bass': None,
            'Strings': None
        }
        length = 0
        for track in mult.tracks:
            length = track.pianoroll.shape[0]
            valid_matrix = track.pianoroll[:, valid_range[0]:valid_range[1]].copy()
            instr_tracks[track.name] = valid_matrix
        for name, content in instr_tracks.items():
            if content is None:
                instr_tracks[name] = np.zeros((length, valid_range[1]-valid_range[0]))

        piece_num = math.ceil(length / (time_step * bar_length) + 1)
        print(piece_num)

        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'PiecesNum': piece_num}})
        print('Progress: {:.2%}\n'.format( midi_collection.count({'PiecesNum': {'$exists': True}}) / midi_collection.count()))


def reset_paragraph_num_info():
    midi_collection = get_midi_collection()
    root_dir = 'E:/merged_midi/'
    for midi in midi_collection.find({'PiecesNum': {'$exists': False}}, no_cursor_timeout = True):
        path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
        pm = pretty_midi.PrettyMIDI(path)
        length = pm.get_end_time()

        piece_num = math.ceil(length / 8)
        print(piece_num)

        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'PiecesNum': piece_num}})
        print('Progress: {:.2%}\n'.format( midi_collection.count({'PiecesNum': {'$exists': True}}) / midi_collection.count()))


def find_music_with_multiple_genres():
    root_dir = 'E:/merged_midi/'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'GenresNum': {'$exists': False}}):
        performer = midi['Performer']
        name = midi['Name']
        genres = []
        total_genres_num = midi_collection.count({'Name': name, 'Performer': performer})
        for md in midi_collection.find({'Name': name, 'Performer': performer}):
            genres.append(md['Genre'])
        print(total_genres_num, genres)
        midi_collection.update_one({'_id': midi['_id']},
                                   {'$set': {'GenresNum': total_genres_num, 'TotalGenres': genres}})
        print('Progress: {:.2%}\n'.format(midi_collection.count({'GenresNum': {'$exists': True}}) / midi_collection.count()))

def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.zeros(f['shape'], np.bool_)
        print(f['nonzero'].shape) # (5, 156149621)
        for i in f['nonzero']:
            print(i)
        data[[x for x in f['nonzero']]] = True
    return data

def test_example_data():
    path = 'E:/data/train_x_lpd_5_phr.npz'
    data = load_data_from_npz(path)
    print(data.shape)  # (102378, 4, 48, 84, 5)

def get_music_with_no_empty_tracks():
    midi_collection = get_midi_collection()
    print(midi_collection.count({'Genre': 'rock', 'NotEmptyTracksNum': {'$gte': 4}}), midi_collection.count())

def build_single_tensor_from_sparse(path):
    midi_collection = get_midi_collection()
    nonzeros = np.load(path)['arr_0']
    midi = midi_collection.find_one({'md5': path[:-4]})
    result = np.zeros((midi['PiecesNum'] + 1, 4, 120, 84, 5))
    result[[data for data in nonzeros]] = True
    return result

def get_original_tempo(md5):
    midi = get_midi_collection().find_one({'md5': md5})
    print(midi['Info']['tempo'][0])
    return midi['Info']['tempo'][0]

def build_midi_from_tensor(src_path, save_path, time_step=120, bar_length=4, valid_range = (24, 108)):
    data = build_single_tensor_from_sparse(src_path)
    piece_num = data.shape[0]
    instr_list = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    program_list = [0, 0, 24, 32, 48]
    pm = pretty_midi.PrettyMIDI()
    for i in range(5):
        instr = instr_list[i]
        is_drum = (instr == 'Drums')
        instr_track = pretty_midi.Instrument(program_list[i], is_drum=is_drum, name=instr)
        track_data = data[:, :, :, :, i]

        for piece in range(piece_num):
            for bar in range(bar_length):
                init_time = piece * (bar_length * time_step) + bar * time_step
                print(init_time)
                for note in range(valid_range[1]-valid_range[0]):

                    during_note = False
                    note_begin = init_time

                    for time in range(time_step):
                        has_note = track_data[piece, bar, time, note]
                        if has_note:
                            if not during_note:
                                during_note = True
                                note_begin = time + init_time
                            else:
                                if time != time_step-1:
                                    continue
                                else:
                                    note_end = time + init_time
                                    print(note_begin / 60, note_end / 60)
                                    instr_track.notes.append(pretty_midi.Note(64, note + 12,
                                                                              note_begin / 48,
                                                                              note_end / 48))
                        else:
                            if not during_note:
                                continue
                            else:
                                note_end = time + init_time
                                print(note_begin / 60, note_end / 60)
                                instr_track.notes.append(pretty_midi.Note(64, note + 12,
                                                                          note_begin / 48,
                                                                          note_end / 48))
                                during_note = False

        pm.instruments.append(instr_track)

    pm.write(save_path)

def generate_nonzeros_by_notes():
    root_dir = 'E:/merged_midi/'

    midi_collection = get_midi_collection()
    genre_collection = get_genre_collection()
    for genre in genre_collection.find():
        genre_name = genre['Name']
        print(genre_name)
        npy_file_root_dir = 'E:/midi_matrix/one_instr/' + genre_name + '/'
        if not os.path.exists(npy_file_root_dir):
            os.mkdir(npy_file_root_dir)
        print('Progress: {:.2%}'.format(
            midi_collection.count({'Genre': genre_name, 'OneInstrNpyGenerated': True}) / midi_collection.count({'Genre': genre_name})),
            end='\n')

        for midi in midi_collection.find({'Genre': genre_name, 'OneInstrNpyGenerated': False}, no_cursor_timeout = True):
            path = root_dir + genre_name + '/' + midi['md5'] + '.mid'
            save_path = npy_file_root_dir + midi['md5'] + '.npz'
            pm = pretty_midi.PrettyMIDI(path)
            segment_num = math.ceil(pm.get_end_time() / 8)
            note_range = (24, 108)

            # data = np.zeros((segment_num, 64, 84), np.bool_)
            nonzeros = []

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
                                nonzeros.append([segment, time, pitch])

            nonzeros = np.array(nonzeros)
            np.savez_compressed(save_path, nonzeros)
            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'OneInstrNpyGenerated': True}})
            print('Progress: {:.2%}'.format(
                midi_collection.count({'Genre': genre_name, 'OneInstrNpyGenerated': True}) / midi_collection.count()), end='\n')



def test_build_numpy_by_note_length():
    path = './e56b7b3e51ee03bab6fedbebcc90ed00.mid'
    pm = pretty_midi.PrettyMIDI(path)
    segment_num = math.ceil(pm.get_end_time() / 8)
    note_range = (24, 108)

    npy_path = './data.npz'

    data = np.zeros((segment_num, 64, 84), np.bool_)
    nonzeros = []

    quarter_length = (60 / 120) / 4
    for instr in pm.instruments:
        print(instr.name)
        if not instr.is_drum:
            for note in instr.notes:
                start = int(note.start / quarter_length)
                end = int(note.end / quarter_length)
                pitch = note.pitch
                if pitch < 24 or pitch >= 108:
                    continue
                else:
                    pitch -= 24
                    for time_raw in range(start, end):
                        segment = int(time_raw / 64)
                        time = time_raw % 64
                        data[segment, time, pitch] = True
                        nonzeros.append([segment, time, pitch])

    nonzeros = np.array(nonzeros)
    print(nonzeros.shape)
    np.savez_compressed(npy_path, nonzeros)



def generate_sparse_matrix_of_genre(genre):
    npy_path = 'D:/data/' + genre + '/data_sparse.npz'


    with np.load(npy_path) as f:
        shape = f['shape']
        data = np.zeros(shape, np.float_)
        nonzeros = f['nonzeros']
        for x in nonzeros:
            data[(x[0], x[1], x[2])] = 1.

    return data


def generate_sparse_matrix_from_multiple_genres(genres):
    length = 0

    genre_collection = get_genre_collection()
    for genre in genre_collection.find({'Name': {'$in': genres}}):
        length += genre['PiecesNum']
    data = np.zeros([length, 64, 84], np.float_)

    for genre in genres:
        npy_path = 'D:/data/' + genre + '/data_sparse.npz'
        with np.load(npy_path) as f:
            nonzeros = f['nonzeros']
            for x in nonzeros:
                data[(x[0], x[1], x[2])] = 1.
    return data

def pretty_midi_test():
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(0)
    instr.notes.append(pretty_midi.Note(64, 64, 0, 2))
    pm.instruments.append(instr)
    pm.write('./pm_test.mid')

def test_build_midi():
    test_path = '4e074b1b1470a0f4b58a94272f1f06fa.npz'
    data = build_single_tensor_from_sparse(test_path)
    save_path = 'test.mid'
    build_midi_from_tensor(test_path, save_path)

def label_all_numpy_existed():
    root_dir = 'e:/midi_matrix/rock'
    for file in os.listdir(root_dir):
        md5 = file[:-4]
        get_midi_collection().update_one({'md5': md5, 'Genre': 'rock'}, {'$set': {'MultiInstrNpyGenerated': True}})

def test_sample_data():
    path = './classic_piano_train_1.npy'
    data = np.load(path)
    print(data.shape)


if __name__ == '__main__':
    import time
    time1 = time.time()
    genres = ['metal', 'punk', 'folk', 'newage', 'country', 'bluegrass']
    data = generate_sparse_matrix_from_multiple_genres(genres)
    time2 = time.time()
    print(time2-time1)
    np.random.shuffle(data)
    time3 = time.time()
    print(data.shape)
    print(time3-time1)