from midi_extended.MidiFileExtended import MidiFileExtended
from util.toolkits.midi import *


class RocknRoll(object):
    def __init__(self):
        self.bpm = 138
        self.time_signature = '4/4'
        self.key = 'E'
        self.key_note = 40
        self.file_path = '../data/midi/write/rock.mid'
        self.mid = MidiFileExtended(self.file_path, type=1, mode='w')

    def write(self):
        self.mid.add_new_track('RhythmGuitar', self.time_signature, self.bpm, self.key, {'0': 29})
        self.add_riff()

    def add_riff(self):
        pattern = [0, 0, 1, 1, 0, 0, 2, 3]

        arrange_list = [['1', '3', '1', '4', '4'],
                        ['3', '5', '3', '6', '6'],
                        ['3', '5', '3', '6', '7', '6'],
                        ['1', '3', '1', '4', '5', '4']
                        ]

        time_list = [[1/8, 1/8, 1/8, 1/4, 1/8],
                     [1/8, 1/8, 1/8, 1/4, 1/8],
                     [1/8, 1/8, 1/8, 1/8, 1/8, 1/8],
                     [1/8, 1/8, 1/8, 1/8, 1/8, 1/8]
                     ]

        rhythm = self.mid.get_extended_track('RhythmGuitar')

        for p in pattern:
            arrange = arrange_list[p]
            time_arrange = time_list[p]
            for i in range(len(arrange)):
                time = time_arrange[i]
                note = self.key_note + get_note_from_arrange(arrange[i])

                rhythm.add_chord(get_power_chord(note, time), time)

if __name__ == '__main__':
    golden_wind = RocknRoll()
    golden_wind.write()
    golden_wind.mid.save_midi()
    golden_wind.mid.play_it()