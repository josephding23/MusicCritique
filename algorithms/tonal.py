from util.data import auxillary as aux
from util.toolkits.database import *
from util.toolkits.midi import *
from music21 import analysis, converter
import pretty_midi
import numpy as np
import scipy.stats as stats
import math


def test_tonal_analysis():
    root_note_names = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭G', 'G', '♭A', 'A', '♭B', 'B']

    scales = []
    for i in range(12):
        root_name = root_note_names[i]
        root_note = i

        for scale_name, scale in get_mode_dict()['Heptatonic'].items():
            name = root_name + '_' + scale_name
            scales.append({name: [root_note + note for note in scale]})

        for scale_name, scale in get_mode_dict()['Pentatonic'].items():
            name = root_name + '_' + scale_name
            scales.append({name: [root_note + note for note in scale]})

    print(scales)


def get_note_lengths(path):
    notes_length = [0 for _ in range(12)]
    pm = pretty_midi.PrettyMIDI(path)
    for instr in pm.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                length = note.end - note.start
                pitch = note.pitch
                notes_length[pitch % 12] += length

    return notes_length


def get_weights(mode, name='ks'):
    if name == 'kk':
        a = analysis.discrete.KrumhanslKessler()
        # Strong tendancy to identify the dominant key as the tonic.
    elif name == 'ks':
        a = analysis.discrete.KrumhanslSchmuckler()
    elif name == 'ae':
        a = analysis.discrete.AardenEssen()
        # Weak tendancy to identify the subdominant key as the tonic.
    elif name == 'bb':
        a = analysis.discrete.BellmanBudge()
        # No particular tendancies for confusions with neighboring keys.
    elif name == 'tkp':
        a = analysis.discrete.TemperleyKostkaPayne()
        # Strong tendancy to identify the relative major as the tonic in minor keys. Well-balanced for major keys.
    else:
        assert name == 's'
        a = analysis.discrete.SimpleWeights()
        # Performs most consistently with large regions of music, becomes noiser with smaller regions of music.
    return a.getWeights(mode)


def get_key_name(index):
    if index // 12 == 0:
        mode = 'major'
    else:
        mode = 'minor'

    tonic_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    tonic = tonic_list[index % 12]
    return tonic + ' ' + mode


def krumhansl_schmuckler(path):
    note_lengths = get_note_lengths(path)
    key_profiles = [0 for _ in range(24)]

    for key_index in range(24):

        if key_index // 12 == 0:
            mode = 'major'
        else:
            mode = 'minor'
        weights = get_weights(mode, 'kk')

        current_note_length = note_lengths[key_index:] + note_lengths[:key_index]

        pearson = stats.pearsonr(current_note_length, weights)[0]

        key_profiles[key_index] = math.fabs(pearson)

    key_name = get_key_name(np.argmax(key_profiles))
    print(key_name)
    # print(key_profiles, '\n', note_lengths)


def find_meta(path):
    pm = pretty_midi.PrettyMIDI(path)
    for ks in pm.key_signature_changes:
        print(ks)


def test():
    path = '../data/midi/read/All You Need Is Love - Beatles.mid'

    find_meta(path)

    s = converter.parse(path)
    s.plot('histogram', 'pitch')
    p = analysis.discrete.KrumhanslKessler()
    print(p.getSolution(s))

    krumhansl_schmuckler(path)


if __name__ == '__main__':
    test()
