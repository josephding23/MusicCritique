from pymongo import MongoClient
import os
import pretty_midi
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
        whole_length = genre['ValidPiecesNum']

        shape = np.array([whole_length, time_step, valid_range[1]-valid_range[0]])

        processed = 0
        last_piece_num = 0
        whole_num = midi_collection.count({'Genre': genre['Name']})

        non_zeros = []
        for midi in midi_collection.find({'Genre': genre['Name']}, no_cursor_timeout=True):

            path = root_dir + genre['Name'] + '/' + midi['md5'] + '.npz'
            valid_pieces_num = midi['PiecesNum'] - 1

            f = np.load(path)
            matrix = f['arr_0'].copy()
            print(valid_pieces_num, matrix.shape[0])
            for data in matrix:
                try:
                    data = data.tolist()

                    if data[0] < valid_pieces_num:
                        piece_order = last_piece_num + data[0]
                        non_zeros.append([piece_order, data[1], data[2]])
                except:
                    print(path)

            last_piece_num += valid_pieces_num
            processed += 1

            print('Progress: {:.2%}\n'.format(processed / whole_num))

        non_zeros = np.array(non_zeros)
        print(non_zeros.shape)
        np.savez_compressed(save_dir + '/data_sparse' + '.npz', nonzeros=non_zeros, shape=shape)

        genre_collection.update_one({'_id': genre['_id']}, {'$set': {'DatasetGenerated': True}})


def set_paragraph_num_info():
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


def build_single_tensor_from_sparse(path):
    midi_collection = get_midi_collection()
    nonzeros = np.load(path)['arr_0']
    midi = midi_collection.find_one({'md5': path[:-4]})
    result = np.zeros((midi['PiecesNum'] + 1, 4, 120, 84, 5))
    result[[data for data in nonzeros]] = True
    return result


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
            # segment_num = math.ceil(pm.get_end_time() / 8)
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
        length += genre['ValidPiecesNum']
    data = np.zeros([length, 64, 84], np.float_)
    for genre in genres:
        npy_path = 'D:/data/' + genre + '/data_sparse.npz'
        with np.load(npy_path) as f:
            nonzeros = f['nonzeros']
            for x in nonzeros:
                data[(x[0], x[1], x[2])] = 1.
    return data


if __name__ == '__main__':
    # get_genre_collection().update_many({}, {'$set': {'DatasetGenerated': False}})
    merge_all_sparse_matrices()