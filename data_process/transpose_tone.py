import mido
from music21 import *
import pretty_midi
import os
from pymongo import MongoClient
import traceback

def get_midi_collection():
    client = MongoClient(connect=False)
    return client.free_midi.midi

def transpose_tone_mido():
    midi = mido.MidiFile('./test.mid')
    for msg in midi:
        if msg.is_meta:
            print(msg)

def estimate_key_test():
    test_stream = converter.parse('./test.mid')
    estimate_key = test_stream.analyze('key')
    estimate_tone, estimate_mode = (estimate_key.tonic, estimate_key.mode)
    c_key = key.Key('C', 'major')
    c_tone, c_mode = (c_key.tonic, c_key.mode)
    margin = interval.Interval(estimate_tone, c_tone)

    semitones = margin.semitones

    mid = pretty_midi.PrettyMIDI('test.mid')
    for instr in mid.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                note.pitch += semitones

    mid.write('test_transposed.mid')
    print(converter.parse('./test_transposed.mid').analyze('key'))


def transpose_to_c():
    root_dir = 'E:/free_midi_library/'
    transpose_root_dir = 'E:/transposed_midi/'
    midi_collection = get_midi_collection()
    for midi in midi_collection.find({'Transposed': False}, no_cursor_timeout = True):
        original_path = os.path.join(root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')

        if not os.path.exists(os.path.join(transpose_root_dir, midi['Genre'])):
            os.mkdir(os.path.join(transpose_root_dir, midi['Genre']))

        transposed_path = os.path.join(transpose_root_dir, midi['Genre'] + '/', midi['md5'] + '.mid')
        try:
            original_stream = converter.parse(original_path)

            estimate_key = original_stream.analyze('key')

            estimate_tone, estimate_mode = (estimate_key.tonic, estimate_key.mode)

            c_key = key.Key('C', 'major')
            c_tone, c_mode = (c_key.tonic, c_key.mode)
            margin = interval.Interval(estimate_tone, c_tone)
            semitones = margin.semitones

            mid = pretty_midi.PrettyMIDI(original_path)
            for instr in mid.instruments:
                if not instr.is_drum:
                    for note in instr.notes:
                        if note.pitch + semitones < 128 and note.pitch + semitones > 0:
                            note.pitch += semitones

            mid.write(transposed_path)
            midi_collection.update_one({'_id': midi['_id']}, {'$set': {'Transposed': True}})
            print('Progress: {:.2%}\n'.format(midi_collection.count({'Transposed': True}) / midi_collection.count()))
        except:
            print(traceback.format_exc())



if __name__ == '__main__':
    transpose_to_c()