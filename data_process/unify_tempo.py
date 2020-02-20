from pymongo import MongoClient
import pretty_midi
import os
from pypianoroll import Multitrack, Track
import pypianoroll
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


def get_merged_from_pm(pm):
    new_pm = pretty_midi.PrettyMIDI()

    drums = pretty_midi.Instrument(0, is_drum=True, name='Drums')
    piano = pretty_midi.Instrument(0, name='Piano')
    guitar = pretty_midi.Instrument(24, name='Guitar')
    bass = pretty_midi.Instrument(32, name='Bass')
    strings = pretty_midi.Instrument(48, name='Strings')

    for instr in pm.instruments:
        if instr.is_drum:
            for note in instr.notes:
                drums.notes.append(note)
        elif instr.program//8 == 0:
            for note in instr.notes:
                piano.notes.append(note)
        elif instr.program//8 == 3:
            for note in instr.notes:
                guitar.notes.append(note)
        elif instr.program//8 == 4:
            for note in instr.notes:
                bass.notes.append(note)
        elif instr.program < 96 or 104 <= instr.program < 112:
            for note in instr.notes:
                strings.notes.append(note)

    for instr in [drums, piano, guitar, bass, strings]:
        new_pm.instruments.append(instr)
    return new_pm


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

def get_music_with_tempo_changes():
    root_dir = 'E:/free_midi_library/'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find():
        path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        midi_file = mido.MidiFile(path)
        tempos = []
        times = []
        for msg in midi_file:
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = mido.tempo2bpm(msg.tempo)
                time = msg.time
                tempos.append(tempo)
                times.append(time)
        print(tempos)
        print(times)
        print()
        '''
        pm = pretty_midi.PrettyMIDI(path)
        tempo_change_times, tempo = pm.get_tempo_changes()
        if len(tempo_change_times) > 1:
            print(tempo_change_times.tolist(), '\n', tempo.tolist(), '\n', path, '\n')
            '''

def get_tempo(path):
    pm = pretty_midi.PrettyMIDI(path)
    # tempo_estimated = pm.estimate_tempo()
    _, tempo = pm.get_tempo_changes()
    return tempo.tolist()

def tempo_unify_and_merge_test():
    test_path = './test.mid'
    original_tempo = get_tempo(test_path)[0]
    changed_rate = original_tempo / 120

    '''
    score = music21.converter.parse(test_path)
    new_score = score.scaleOffsets(changed_rate).scaleDurations(changed_rate)
    new_score.write('midi', './test_merged.mid')
    '''
    pm = pretty_midi.PrettyMIDI(test_path)
    for instr in pm.instruments:
        for note in instr.notes:
            note.start *= changed_rate
            note.end *= changed_rate
    merged_pm = get_merged_from_pm(pm)
    # merged_multi = get_merged(mult)
    merged_pm.write('./test_merged.mid')

    # new_pm = pretty_midi.PrettyMIDI()


def tempo_unify_and_merge():

    midi_collection = get_midi_collection()
    root_dir = 'E:/free_midi_library/'
    merged_root_dir = 'E:/merged_midi/'

    for midi in midi_collection.find({'MergedAndScaled': False}, no_cursor_timeout = True):
        original_path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        try:
            original_tempo = get_tempo(original_path)[0]
            changed_rate = original_tempo / 120

            if not os.path.exists(os.path.join(merged_root_dir, midi['Genre'])):
                os.mkdir(os.path.join(merged_root_dir, midi['Genre']))

            pm = pretty_midi.PrettyMIDI(original_path)
            for instr in pm.instruments:
                for note in instr.notes:
                    note.start *= changed_rate
                    note.end *= changed_rate

            merged_path = os.path.join(merged_root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
            merged = get_merged_from_pm(pm)
            merged.write(merged_path)

            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'MergedAndScaled': True}})

            print('Progress: {:.2%}\n'.format(midi_collection.count({'MergedAndScaled': True}) / midi_collection.count()))
            
        except:
            pass

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
    tempo_unify_and_merge()