from pymongo import MongoClient
import os
import pretty_midi
import pypianoroll
import traceback
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse as ss

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_genre_collection():
    client = MongoClient(connect=False)
    return client.free_midi.genres

def get_whole_genre_numpy(time_step=120, bar_length=4, valid_range = (24, 108), genre='rock'):

    root_dir = 'E:/merged_midi/'
    npy_file_root_Dir = 'E:/midi_matrix/' + genre + '/'

    last_segment_number = 0

    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}, 'NpyGenerated': False}, no_cursor_timeout = True):
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
            print(last_segment_number)
            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'NpyGenerated': True}})
            # prossed += 1
            print('Progress: {:.2%}\n'.format(midi_collection.count({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}, 'NpyGenerated': True}) / midi_collection.count({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}})))
        except:
            pass
        # print(merged.shape)


def divide_into_groups(genre='rock', group_num=10):
    midi_collection = get_midi_collection()
    total_amount = midi_collection.count({'Genre': 'rock', 'NotEmptyTracksNum': {'$gte': 4}})
    num_per_group = total_amount // group_num + 1
    current_num = 0
    for midi in midi_collection.find({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}},
                                     no_cursor_timeout=True):
        group = current_num // num_per_group
        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'DataGroup': group}})
        current_num += 1


def merge_all_sparse_matrices(root_dir='E:/midi_matrix/', genre='rock', group_num=10, save_dir='d:/data/', time_step=120, bar_length=4, valid_range = (24, 108)):
    midi_collection = get_midi_collection()
    divide_into_groups(genre, group_num)

    whole_length = 132700
    shape = np.array([whole_length, bar_length, time_step, valid_range[1]-valid_range[0], 5])

    processed = 0
    last_piece_num = 0
    whole_num = midi_collection.count({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}})

    for group in range(group_num):
        non_zeros = []
        for midi in midi_collection.find({'Genre': genre, 'NotEmptyTracksNum': {'$gte': 4}, 'DataGroup': group}, no_cursor_timeout=True):

            path = root_dir + genre + '/' + midi['md5'] + '.npz'
            pieces_num = midi['PiecesNum']

            f = np.load(path)
            matrix = f['arr_0'].copy().transpose()
            print(pieces_num, matrix.shape[0])
            for data in matrix:
                try:
                    data = data.tolist()
                    piece_order = last_piece_num + data[0]
                    non_zeros.append([piece_order, data[1], data[2], data[3], data[4]])
                except:
                    print(path)
            # print(matrix['arr_0'].shape)

            # f.close()
            last_piece_num += pieces_num
            processed += 1
            print('Progress: {:.2%}\n'.format(processed / whole_num))
        non_zeros = np.array(non_zeros)
        print(last_piece_num)
        np.savez_compressed(save_dir + genre + '/data_sparse_part' + str(group) + '.npz', nonzeros=non_zeros)
    np.save(save_dir + genre + '/shape.npy', shape)



def print_all_genres_num():
    genres_collection = get_genre_collection()
    midi_collection = get_midi_collection()

    for genre in genres_collection.find():
        pieces_num = {'IgnoreLessThan3': 0, 'IgnoreLessThan4': 0}
        for midi in midi_collection.find({'Genre': genre['Name'], 'NotEmptyTracksNum': {'$gte': 4}}):
            pieces_num['IgnoreLessThan4'] += midi['PiecesNum']
        for midi in midi_collection.find({'Genre': genre['Name'], 'NotEmptyTracksNum': {'$gte': 3}}):
            pieces_num['IgnoreLessThan3'] += midi['PiecesNum']
        genres_collection.update_one(
            {'_id': genre['_id']},
            {'$set': {'PiecesNum': pieces_num}}
        )
        print(genre['Name'], pieces_num)

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
                sparse[data] = True

    return result


def divide_track_test(time_step=120, bar_length=4, valid_range = (24, 108)):
    paths = ['./test.mid']
    for path in paths:
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
            valid_matrix = track.pianoroll[:, valid_range[0]:valid_range[1]]
            instr_tracks[track.name] = valid_matrix
        for name, content in instr_tracks.items():
            if content is None:
                instr_tracks[name] = np.zeros((length, valid_range[1]-valid_range[0]))
        merged_instr_matrix = np.dstack((instr_tracks['Drums'], instr_tracks['Piano'],
                                        instr_tracks['Guitar'], instr_tracks['Bass'],
                                        instr_tracks['Strings']))
        whole_paragraphs = length // (time_step * bar_length)
        nonzeros = []
        for track_num in range(5):
            instr_track = merged_instr_matrix[:, :, track_num]
            for current_time in range(length):
                for note in range(valid_range[1]-valid_range[0]):
                    if instr_track[current_time][note] != 0:
                        paragraph_num = current_time // (time_step * bar_length)
                        bar_num = current_time % (time_step * bar_length) // time_step
                        time_node = current_time % time_step

                        nonzeros.append([paragraph_num, bar_num, time_node, note, track_num])
                        # print(paragraph_num, bar_num, time_node, note, track_num)

        reconstruct = np.zeros([whole_paragraphs, bar_length, time_step, valid_range[1]-valid_range[0], 5], np.bool_)
        print(reconstruct.shape)
        nonzeros_matrix = np.array(nonzeros).transpose()
        # print(nonzeros_matrix.transpose().shape)
        reconstruct[[x for x in nonzeros_matrix]] = True
        for i in reconstruct[2, 1, :, :, 2]:
            print(i)
        '''
        divided_in_time_step_list = []
        for index in range(segment_num):
            time_step_slice = merged_instr_matrix[index*time_step:(index+1)*time_step, :, :]
            divided_in_time_step_list.append(time_step_slice)
        divided_in_time_step = np.array(divided_in_time_step_list)

        divided_in_bar_list = []
        for index in range(bar_num):
            bar_slice = divided_in_time_step[index*bar_length:(index+1)*bar_length, :, :, :]
            divided_in_bar_list.append(bar_slice)
        divided_in_bar = np.array(divided_in_bar_list)
        first_dim = divided_in_bar.shape[0]
        print(divided_in_bar.shape)

        nonzeros = []

        final_shape = divided_in_bar.shape
        for a in range(final_shape[0]):
            for b in range(final_shape[1]):
                for c in range(final_shape[2]):
                    for d in range(final_shape[3]):
                        for e in range(final_shape[4]):
                            if divided_in_bar[a][b][c][d][e] != 0:
                                nonzeros.append([a, b, c, d, e])

        nonzeros = np.array(nonzeros)
        np.save('./save_test.npy', nonzeros)
        print(nonzeros.shape)
        '''



def test_notes_range_in_single_file():
    valid_range_length = 84
    # valid_range = ((108 - 84) / 2, 108 - (108 - 84) / 2)
    valid_range = (24, 108)
    print(valid_range)
    midi = pretty_midi.PrettyMIDI('./test.mid')
    for instr in midi.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                if note.pitch > valid_range[1] or note.pitch < valid_range[0]:
                    print(note.pitch)

def test_notes_range_in_all():
    midi_collection = get_midi_collection()

    valid_range_length = 84
    # valid_range = ((108 - 84) / 2, 108 - (108 - 84) / 2)
    valid_range = (24, 108)
    root_dir = 'E:/merged_midi/'
    for midi in midi_collection.find():
        total_notes_num = 0
        outta_range_num = 0
        path = root_dir + midi['Genre'] + '/' + midi['md5'] + '.mid'
        pm = pretty_midi.PrettyMIDI(path)
        for instr in pm.instruments:
            if not instr.is_drum:
                for note in instr.notes:
                    total_notes_num += 1
                    if note.pitch > valid_range[1] or note.pitch < valid_range[0]:
                        outta_range_num += 1
        print(round(outta_range_num / total_notes_num, 5))

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


if __name__ == '__main__':
    get_midi_collection().update_many({}, {'$set': {'NpyGenerated': False}})
    get_whole_genre_numpy()
    merge_all_sparse_matrices()