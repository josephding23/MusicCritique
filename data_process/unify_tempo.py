from pymongo import MongoClient
import pretty_midi
import os
from pypianoroll import Multitrack
from pypianoroll import Multitrack, Track
import mido
import music21

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

TRACK_INFO = (
    ('Drums', 0),
    ('Piano', 0),
    ('Guitar', 24),
    ('Bass', 32),
    ('Strings', 48),
)

def get_merged(multitrack):
    """Merge the multitrack pianorolls into five instrument families and
    return the resulting multitrack pianoroll object."""
    track_lists_to_merge = [[] for _ in range(5)]
    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            track_lists_to_merge[0].append(idx)
        elif track.program//8 == 0:
            track_lists_to_merge[1].append(idx)
        elif track.program//8 == 3:
            track_lists_to_merge[2].append(idx)
        elif track.program//8 == 4:
            track_lists_to_merge[3].append(idx)
        elif track.program < 96 or 104 <= track.program < 112:
            track_lists_to_merge[4].append(idx)

    tracks = []
    for idx, track_list_to_merge in enumerate(track_lists_to_merge):
        if track_list_to_merge:
            merged = multitrack[track_list_to_merge].get_merged_pianoroll('max')
            tracks.append(Track(merged, TRACK_INFO[idx][1], (idx == 0),
                                TRACK_INFO[idx][0]))
        else:
            tracks.append(Track(None, TRACK_INFO[idx][1], (idx == 0),
                                TRACK_INFO[idx][0]))
    return Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat,
                      multitrack.beat_resolution, multitrack.name)

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def get_tempo(path):
    pm = pretty_midi.PrettyMIDI(path)
    return get_midi_info(pm)['tempo']


def merge_then_tempo_unify():

    midi_collection = get_midi_collection()
    root_dir = 'E:/free_midi_library/'
    merged_root_dir = 'E:/merged_midi/'

    for midi in midi_collection.find({'MergedAndScaled': False}, no_cursor_timeout = True):
        original_path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        original_tempo = get_tempo(original_path)[0]
        changed_rate = original_tempo / 120

        if not os.path.exists(os.path.join(merged_root_dir, midi['Genre'])):
            os.mkdir(os.path.join(merged_root_dir, midi['Genre']))
        merged_path = os.path.join(merged_root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        merged_multi = get_merged(Multitrack(original_path))
        merged_multi.write(merged_path)

        score = music21.converter.parse(merged_path)
        new_score = score.scaleOffsets(changed_rate).scaleDurations(changed_rate)
        new_score.write('midi', merged_path)

        midi_collection.update_one({'_id': midi['_id']}, {'$set': {'MergedAndScaled': True}})

        print('Progress: {:.2%}\n'.format(midi_collection.count({'MergedAndScaled': True}) / midi_collection.count()))

def change_tempo_in_metadata():
    test_path = './test4_changed_tempo_dirty.mid'
    dst_path = './changed_tempo_in_meta.mid'
    test = mido.MidiFile(test_path)
    dst_midi = mido.MidiFile()

    changed_rate = get_tempo(test_path)[0] / 120
    dst_tempo = mido.bpm2tempo(120)
    for track in test.tracks:
        new_track = mido.MidiTrack()
        new_track.name = track.name
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                msg = mido.MetaMessage('set_tempo', tempo=dst_tempo)
                new_track.append(msg)
            else:
                msg.time = int(msg.time * changed_rate)
                new_track.append(msg)
        dst_midi.tracks.append(new_track)


    dst_midi.save(dst_path)

    pm = pretty_midi.PrettyMIDI(dst_path)
    print(get_tempo(test_path), get_tempo(dst_path))

if __name__ == '__main__':
    merge_then_tempo_unify()