from pymongo import MongoClient
import os
import pretty_midi
import pypianoroll
import numpy as np

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_whole_genre_numpy(time_step=120, bar_length=4, note_valid_length=84):
    genre = 'rock'
    root_dir = 'E:/merged_midi/'
    valid_range = ((108 - note_valid_length) // 2, 108 - (108 - note_valid_length) // 2)
    merged = None
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Genre': genre}):
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
            valid_matrix = track.pianoroll[:, valid_range[0]:valid_range[1]]
            instr_tracks[track.name] = valid_matrix
        for name, content in instr_tracks.items():
            if content is None:
                instr_tracks[name] = np.zeros((length, note_valid_length))
        segment_num = length // time_step
        bar_num = segment_num // bar_length
        merged_instr_matrix = np.dstack((instr_tracks['Drums'], instr_tracks['Piano'],
                                         instr_tracks['Guitar'], instr_tracks['Bass'],
                                         instr_tracks['Strings']))

        divided_in_time_step_list = []
        for index in range(segment_num):
            time_step_slice = merged_instr_matrix[index * time_step:(index + 1) * time_step, :, :]
            divided_in_time_step_list.append(time_step_slice)
        divided_in_time_step = np.array(divided_in_time_step_list)

        divided_in_bar_list = []
        for index in range(bar_num):
            bar_slice = divided_in_time_step[index * bar_length:(index + 1) * bar_length, :, :, :]
            divided_in_bar_list.append(bar_slice)
        divided_in_bar = np.array(divided_in_bar_list)

        if merged is None:
            merged = divided_in_bar
        else:
            merged = np.concatenate((merged, divided_in_bar), axis=0)
        print(merged.shape)

def divide_track_test(time_step=120, bar_length=4, note_valid_length=84):
    paths = ['./test.mid', './test.mid']
    valid_range = ((108 - note_valid_length) // 2, 108 - (108 - note_valid_length) // 2)
    merged = None
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
                instr_tracks[name] = np.zeros((length, note_valid_length))
        segment_num = length // time_step
        bar_num = segment_num // bar_length
        merged_instr_matrix = np.dstack((instr_tracks['Drums'], instr_tracks['Piano'],
                                        instr_tracks['Guitar'], instr_tracks['Bass'],
                                        instr_tracks['Strings']))

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
        print(divided_in_bar.shape)
        if merged is None:
            merged = divided_in_bar
        else:
            merged = np.concatenate((merged, divided_in_bar), axis=0)
    print(merged.shape)

def test_notes_range_in_single_file():
    valid_range_length = 84
    valid_range = ((108 - 84) / 2, 108 - (108 - 84) / 2)
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
    valid_range = ((108 - 84) / 2, 108 - (108 - 84) / 2)
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
    midi_collection = get_midi_collection()


if __name__ == '__main__':
    get_whole_genre_numpy()