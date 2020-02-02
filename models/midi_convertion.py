from pypianoroll import Multitrack, Track
import matplotlib.pyplot as plt
from pretty_midi import PrettyMIDI
import pypianoroll
import os
import errno
import traceback
from pymongo import MongoClient

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
            track.plot()
            multi.append_track(track)
        else:
            track = Track(None, program_dict[key], is_drum=is_drum, name=key)
            multi.append_track(track)
    return multi


def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

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
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info

if __name__ == '__main__':
   file_path = 'E:/free_MIDI/rock/21st Century Schizoid Man - King Crimson.mid'
   multitrack = Multitrack(file_path)
   merged = get_merged(multitrack)
   merged.save('E:/MIDI_converted/rock/21st Century Schizoid Man - King Crimson.mid')
   # merged.plot()
   plt.show()
   for track in merged.tracks:
       print(track.name)

