from midi_extended.MidiFileExtended import MidiFileExtended
from midi_extended.UtilityBox import *
import pretty_midi
import pypianoroll

def freedom():
    midi_path = '../data/midi/read/RATM/freedom.mid'
    numpy_dir = '../data/numpy/RATM/'
    plots_dir = '../data/plots/RATM/'
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    print(midi_data.estimate_tempo())

    '''
    midi_file = MidiFileExtended(midi_path)
    for track in midi_file.tracks:
        calculate_track_duration(track)
        '''

def vietnow():
    midi_path = '../data/midi/read/RATM/vietnow.mid'
    midi_file = MidiFileExtended(midi_path, charset='')
    track = midi_file.get_track_by_name('Guitar-1')
    calculate_track_duration(track)
        # midi_file.turn_track_into_numpy_matrix()

if __name__ == '__main__':
    vietnow()