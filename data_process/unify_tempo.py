from pymongo import MongoClient
from pypianoroll import Multitrack
from data_process.track_merge import get_merged
import midi
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


def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi


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