from midi_extended.MidiFileExtended import MidiFileExtended

class GoldenWind(object):
    def __init__(self):
        self.bpm = 127
        self.time_signature = '4/4'
        self.key = 'Am'
        self.file_path = '../data/midi/write/golden_wind.mid'
        self.mid = MidiFileExtended(self.file_path, type=1, mode='w')

    def write(self):
        self.mid.add_new_track('Piano1', self.time_signature, self.bpm, self.key, {'0': 0})
        track1 = self.mid.get_extended_track('Piano1')
        track1.add_meta_info()
        self.mid.add_new_track('Piano2', self.time_signature, self.bpm, self.key, {'0': 0})
        track2 = self.mid.get_extended_track('Piano2')
        track2.add_meta_info()

        for i in [1, 2, 1, 3]:
            self.intro(i, False)
        for i in [1, 2, 1, 4]:
            self.intro(i, True)
        for i in [1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3]:
            self.piano2_pattern(i)
        for i in [1, 2, 3, 4]:
            self.piano1_parapraph(i)
        for i in [1, 2, 1, 3]:
            self.intro(i, False)
        for i in [1, 2, 1, 4]:
            self.intro(i, True)

    def intro(self, pattern, both):
        tracks = [self.mid.get_extended_track('Piano1'), self.mid.get_extended_track('Piano2')]
        for i in range(2):
            track = tracks[i]
            if not both and i == 0:
                track.wait(1)
                continue
            if pattern == 4:
                track.add_note(7, 1/8, base_num=-1-i)
                track.add_note(7, 1/8, base_num=-1-i)
                track.add_note(7, 1/8, base_num=-1-i)
                track.add_note(6, 1/16, base_num=-1-i)
                track.add_note(7, 1/8, base_num=-1-i)
                track.wait(7/16)
            else:
                track.add_note(7, 1/8, base_num=-1-i)
                track.add_note(7, 1/8, base_num=-1-i)
                track.add_note(7, 1/16, base_num=-1-i)
                track.add_note(6, 1/16, base_num=-1-i)
                track.wait(1/16)
                track.add_note(7, 1/16, base_num=-1-i)
                track.wait(1/16)
                if pattern == 1:
                    track.add_note(2, 1/16, base_num=-i)
                    track.wait(1/16)
                    track.add_note(7, 1/16, base_num=-1-i)
                    track.wait(1/16)
                    track.add_note(4, 1/16, base_num=-1-i, alt=1)
                elif pattern == 2:
                    track.add_note(4, 1/16, base_num=-i)
                    track.wait(1 / 16)
                    track.add_note(3, 1/16, base_num=-i)
                    track.wait(1 / 16)
                    track.add_note(2, 1/16, base_num=-i)
                elif pattern == 3:
                    track.add_note(4, 1/16, base_num=-i)
                    track.wait(1 / 16)
                    track.add_note(3, 1/16, base_num=-i)
                    track.wait(1 / 16)
                    track.add_note(7, 1/16, base_num=-1-i)
                track.add_note(6, 1/8, base_num=-1-i)

    def piano2_pattern(self, pattern):
        track = self.mid.get_extended_track('Piano2')
        if pattern == 1:
            track.add_note([7, 4, 7], 1/4+1/8, alt=[0, 1, 0], base_num=[-1, -1, -2])
            track.add_note([4, 2, 5], 1/8+1/4, alt=[0, 0, 1], base_num=[-1, -1, -2])
            track.wait(1/4)
        elif pattern == 2:
            track.add_note([7, 7], 1/2, base_num=[-1, -2])
            track.add_note([4, 4], 1/2, base_num=[-1, -2], alt=[1, 1])
        elif pattern == 3:
            track.add_note([1, 1], 1/2, base_num=[0, -1], alt=[1, 1])
            track.add_note([4, 4], 1/2, base_num=[-1, -2], alt=[1, 1])
        elif pattern == 4:
            track.add_note([4, 7], 1/4, base_num=[-1, -2], alt=[1, 0])
            track.wait(3/4)

    def piano1_parapraph(self, paragraph):
        track = self.mid.get_extended_track('Piano1')
        if paragraph == 1:
            track.add_note(4, 1/4+1/8, base_num=1, alt=1)
            track.add_note(4, 1/8+1/4, base_num=1)
            track.wait(1/8)
            track.add_note(2, 1/16, base_num=1)
            track.add_note(3, 1/16, base_num=1)

            track.add_note(4, 1/8+1/16, base_num=1)
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note(1, 1/8+1/16, base_num=1, alt=1)
            track.add_note(2, 1/16+1/8, base_num=1)
            track.add_note(3, 1/8, base_num=1)

            track.add_note(4, 1/4+1/8, base_num=1, alt=1)
            track.add_note(7, 1/8+1/4, base_num=1)
            track.add_note(7, 1/8)
            track.add_note(1, 1/8, base_num=1, alt=1)

            track.add_note(2, 1/8+1/16, base_num=1)
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note(1, 1/8+1/16, base_num=1, alt=1)
            track.add_note(6, 1/16+1/8, base_num=1)
            track.add_note(5, 1/8, base_num=1)

        if paragraph == 2:
            track.add_note([4, 2, 7], 1/4+1/8, alt=[1, 0, 0], base_num=[1, 1, 0])
            track.add_note([4, 2, 7], 1/8+1/4, base_num=[1, 1, 0])
            track.wait(1/8)
            track.add_note(2, 1/16, base_num=1)
            track.add_note(3, 1/16, base_num=1)

            track.add_note([4, 1, 5], 1/8+1/16, alt=[0, 1, 0], base_num=[1, 1, 0])
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note([1, 6], 1/8+1/16, alt=[1, 1], base_num=[1, 0])
            track.add_note(2, 1/16+1/8, base_num=1)
            track.add_note(3, 1/8, base_num=1)

            track.add_note([4, 7], 1/4+1/8, alt=[1, 0], base_num=[1, 0])
            track.add_note([7, 4], 1/8+1/4, base_num=[1, 1])
            track.add_note(7, 1/8, base_num=1)
            track.add_note(1, 1/8, base_num=2, alt=1)

            track.add_note([2, 7], 1/8+1/16, base_num=[1, 0])
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(5, 1/8)
            track.add_note(4, 1/8+1/16, alt=1)
            track.add_note(2, 1/16+1/8, base_num=1)
            track.add_note(3, 1/8, base_num=1)

        if paragraph == 3:
            track.add_note([4, 2, 7], 1/ 4 + 1/8, alt=[1, 0, 0], base_num=[1, 1, 0])
            track.add_note([4, 2, 7], 1/8 + 1/4, base_num=[1, 1, 0])
            track.wait(1/8)
            track.add_note(2, 1/16, base_num=1)
            track.add_note(3, 1/16, base_num=1)

            track.add_note([4, 7, 5], 1/8+1/16, base_num=[1, 0, 0])
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note([1, 5], 1/8+1/16, alt=[1, 0], base_num=[1, 0])
            track.add_note(2, 1/16+1/8, base_num=1)
            track.add_note(3, 1/8, base_num=1)

            track.add_note([4, 2, 7], 1/4+1/8, alt=[1, 0, 0], base_num=[1, 1, 0])
            track.add_note([7, 4, 1], 1/8+1/4, base_num=[1, 1, 1], alt=[0, 0, 1])
            track.add_note(7, 1/8, base_num=1)
            track.add_note(1, 1/8, base_num=2, alt=1)

            track.add_note(2, 1/8+1/16, base_num=1)
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note(1, 1/8+1/16, base_num=1, alt=1)
            track.add_note(6, 1/16+1/8, base_num=1)
            track.add_note(5, 1/8, base_num=1)

        if paragraph == 4:
            track.add_note([4, 2, 7], 1/4+1/8, alt=[1, 0, 0], base_num=[1, 1, 0])
            track.add_note([4, 2, 7], 1/8+1/4, base_num=[1, 1, 0])
            track.wait(1 / 8)
            track.add_note(2, 1/16, base_num=1)
            track.add_note(3, 1/16, base_num=1)

            track.add_note([4, 1, 5], 1/8+1/16, alt=[0, 1, 0], base_num=[1, 1, 0])
            track.add_note(3, 1/16+1/8, base_num=1)
            track.add_note(2, 1/8, base_num=1)
            track.add_note([1, 6], 1/8+1/16, alt=[1, 1], base_num=[1, 0])
            track.add_note(2, 1/16+1/8, base_num=1)
            track.add_note(3, 1/8, base_num=1)

            track.add_note([4, 2], 1/4+1/8, base_num=1, alt=[1, 0])
            track.add_note([7, 4], 1/8+1/4, base_num=1)
            track.add_note(7, 1/8)
            track.add_note(1, 1/8, base_num=1, alt=1)

            track.add_note(2, 1/8+1/16, base_num=1)
            track.add_note(5, 1/16+1/8, base_num=1)
            track.add_note(4, 1/8, base_num=1, alt=1)
            track.add_note(4, 1/8+1/16, base_num=1)
            track.add_note(2, 1/16+1/8, base_num=2)
            track.add_note(6, 1/8, base_num=1, alt=1)

if __name__ == '__main__':
    golden_wind = GoldenWind()
    golden_wind.write()
    golden_wind.mid.save_midi()
    golden_wind.mid.play_it()