from pypianoroll import Multitrack, Track
from pypianoroll.plot import plot_pianoroll
import matplotlib.pyplot as plt
from pretty_midi import PrettyMIDI
import pypianoroll
import os
import errno
import tqdm
import json
import numpy as np
import pretty_midi
import traceback
from pymongo import MongoClient
from data_process.util import *

CONFIG = {
    'multicore': 40, # the number of cores to use (1 to disable multiprocessing)
    'beat_resolution': 24, # temporal resolution (in time step per beat)
    'time_signatures': ['4/4'] # '3/4', '2/4'
}

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_merged(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)"""
    category_list = {'Bass': [], 'Drums': [], 'Guitar': [], 'Piano': [], 'Strings': []}
    program_dict = {'Piano': 0, 'Drums': 0, 'Guitar': 24, 'Bass': 32, 'Strings': 48}
    multi = Multitrack(tempo=multitrack.tempo, downbeat=multitrack.downbeat,
                       beat_resolution=multitrack.beat_resolution, name=multitrack.name)
    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list['Drums'].append(idx)
        elif track.program//8 == 0:
            category_list['Piano'].append(idx)
        elif track.program//8 == 3:
            category_list['Guitar'].append(idx)
        elif track.program//8 == 4:
            category_list['Bass'].append(idx)
        else:
            category_list['Strings'].append(idx)

    for key in category_list.keys():
        is_drum = key == 'Drums'
        if category_list[key]:
            merged = multitrack[category_list[key]].get_merged_pianoroll()
            track = Track(merged, program_dict[key], is_drum=is_drum, name=key)
            # track.plot()
            multi.append_track(track)
        else:
            track = Track(None, program_dict[key], is_drum=is_drum, name=key)
            multi.append_track(track)
    return multi


get_genres = lambda : ['pop', 'rock', 'hip-hop-rap', 'jazz', 'blues', 'classical', 'rnb-soul',
                       'bluegrass',  'country', 'christian-gospel',  'dance-eletric', 'newage',
                       'reggae-ska',  'folk', 'punk', 'disco', 'metal']

def convert_midi_files():
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    converter_root_dir = 'E:/MIDI_converted'
    root_dir = 'E:/free_MIDI'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Converted': False}):
        genre = midi['Genre']
        name = midi['Name']
        performer = midi['Performer']
        filepath = root_dir + '/' + genre + '/' + name + ' - ' + performer + '.mid'
        try:
            midi_name = os.path.splitext(os.path.basename(filepath))[0]
            multitrack = Multitrack(beat_resolution=24, name=midi_name)

            pm = PrettyMIDI(filepath)
            midi_info = get_midi_info(pm)
            multitrack = Multitrack(filepath)
            merged = get_merged(multitrack)
            os.chdir(converter_root_dir)
            if not os.path.exists(converter_root_dir + '/' + genre):
                os.mkdir(converter_root_dir + '/' + genre)
            converter_path =converter_root_dir + '/' + genre + '/' + midi_name + '.npz'
            # merged.save(converter_path)
            print(get_midi_info(pm))
            '''
            midi_collection.update_one(
                {'_id', midi['_id']},
                {'$set' :{'Converted': True}}
            )
            '''
            # print([midi_name, midi_info])
        except:
            print(filepath)
            print(traceback.format_exc())

def midi_filter(midi_info):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_info['first_beat_time'] > 0.0:
        return False
    elif midi_info['num_time_signature_change'] > 1:
        return False
    elif midi_info['time_signature'] not in ['4/4']:
        return False
    return True

def get_midi_info(pm):
    """Return useful information from a pretty_midi.PrettyMIDI instance"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = []
        for i in range(len(pm.time_signature_changes)):
            time_sign.append('{}/{}'.format(pm.time_signature_changes[i].numerator,
                                   pm.time_signature_changes[i].denominator))

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi.tolist()
    }

    return midi_info

def add_info_to_database():
    root_dir = 'E:/free_midi_library/'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'InfoAdded': False}, no_cursor_timeout = True):
        path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        try:
            info = get_midi_info(pretty_midi.PrettyMIDI(path))

            midi_collection.update_one({'_id': midi['_id']}, {'$set': {
                'Info': info,
                'InfoAdded': True
            }})
            print('Progress: {:.2%}\n'.format(midi_collection.count({'InfoAdded': True}) / midi_collection.count()))
        except:
            print(path)
            print(traceback.format_exc())

def tempo_unify():
    midi_collection = get_midi_collection()
    root_dir = 'E:/free_midi_library/'
    test_path = './test.mid'
    unify_tempo = 90
    original = Multitrack(test_path)
    original_tempo = get_midi_info(original.to_pretty_midi())['tempo'][0]
    merged_multi = get_merged(original)
    tracks_dict = {}
    length = 0
    for track in merged_multi.tracks:
        if track.pianoroll.shape[0] != 0:
            length = track.pianoroll.shape[0]
            old_pianoroll = track.pianoroll
            for i in range(128):
                for j in range(length):
                    print(track.pianoroll[j][i], end=' ')
                print()

    '''
    tracks_dict = {}
    for midi in midi_collection.find({}, no_cursor_timeout = True):
        path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        try:
            multi = Multitrack(path)
            merged_multi = get_merged(multi)
            info = get_midi_info(merged_multi.to_pretty_midi())
            print(info['tempo'])
            length = 0
            for track in merged_multi.tracks:
                if track.pianoroll.shape[0] != 0:
                    length = track.pianoroll.shape[0]
            for track in merged_multi.tracks:
                if track.name not in ['unknown']:
                    if track.pianoroll.shape[0] == 0:
                        tracks_dict[track.name] = np.zeros((length, 128))
                    else:
                        tracks_dict[track.name] = track.pianoroll
            # track_path = os.path.join(converted_dir + '/', track.name + '.npy')
        except:
            print(path)
            print(traceback.format_exc())
    '''
def merge_tracks():
    root_dir = 'E:/free_midi_library/'
    converted_root_dir = 'E:/converted_MIDI/'
    midi_collection = get_midi_collection()
    min_length = 10000
    for midi in midi_collection.find({}, no_cursor_timeout = True):
        path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        converted_dir = os.path.join(converted_root_dir, midi['Genre'] + '/', midi['md5'])
        tracks_dict = {}
        '''
        if not os.path.exists(os.path.join(converted_root_dir, midi['Genre'])):
            os.mkdir(os.path.join(converted_root_dir, midi['Genre']))
        if not os.path.exists(converted_dir):
            os.mkdir(converted_dir)
            '''
        try:

            multi = Multitrack(path)
            merged_multi = get_merged(multi)
            # info = get_midi_info(pretty_midi.PrettyMIDI(path))

            length = 0
            for track in merged_multi.tracks:
                if track.pianoroll.shape[0] != 0:
                    length = track.pianoroll.shape[0]
            print(length)

            for track in merged_multi.tracks:

                if track.name not in ['unknown']:
                    if track.pianoroll.shape[0] == 0:
                        tracks_dict[track.name] = np.zeros((length, 128))
                    else:
                        tracks_dict[track.name] = track.pianoroll
            # track_path = os.path.join(converted_dir + '/', track.name + '.npy')
            try:
                tracks = np.dstack((tracks_dict['Guitar'], tracks_dict['Piano'], tracks_dict['Bass'], tracks_dict['Strings'], tracks_dict['Drums']))
                print(tracks.shape)
            except:
                for _, track in tracks_dict.items():
                    print(track.shape)

        except:
            print(path)
            print(traceback.format_exc())

if __name__ == '__main__':
    tempo_unify()