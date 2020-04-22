from midi_extended.MidiFileExtended import MidiFileExtended

class CoffinDance():
    def __init__(self):
        self.bpm = 120
        self.time_signature = '4/4'
        self.key = 'C'
        self.file_path = '../data/midi/write/coffin_dance.mid'
        self.mid = MidiFileExtended(self.file_path, type=1, mode='w')

    def write_coffin(self):
        self.mid.add_new_track('Lead', self.time_signature, self.bpm, self.key, {'0': 81})
        self.intro()
        self.verse()
        self.verse()
        self.verse()
        self.end()

    def intro(self):
        track_lead = self.mid.get_extended_track('Lead')
        track_lead.add_meta_info()
        for i in range(16):
            track_lead.add_note(6, 0.125)
        for i in range(8):
            track_lead.add_note(1, 0.125, base_num=1)

        for i in range(4):
            track_lead.add_note(6, 0.125)
        for i in range(4):
            track_lead.add_note(3, 0.125, base_num=1)
        for i in range(4):
            track_lead.add_note(2, 0.125, base_num=1)
        for i in range(4):
            track_lead.add_note(5, 0.125, base_num=1)

        for i in range(12):
            track_lead.add_note(6, 0.125, base_num=1)
        track_lead.add_note(2, 0.125, base_num=1)
        track_lead.add_note(1, 0.125, base_num=1)
        track_lead.add_note(7, 0.125)
        track_lead.add_note(5, 0.125)

    def verse(self):
        track_lead = self.mid.get_extended_track('Lead')

        track_lead.add_note(6, 0.125, base_num=-1)
        track_lead.wait(0.125)
        track_lead.add_note(6, 0.125, base_num=-1)
        track_lead.add_note(3, 0.125)
        track_lead.add_note(2, 0.125)
        track_lead.wait(0.125)
        track_lead.add_note(1, 0.125)
        track_lead.wait(0.125)

        track_lead.add_note(7, 0.125, -1)
        track_lead.wait(0.125)
        track_lead.add_note(7, 0.125, -1)
        track_lead.add_note(7, 0.125, -1)
        track_lead.add_note(2, 0.125)
        track_lead.wait(0.125)
        track_lead.add_note(1, 0.125)
        track_lead.add_note(7, 0.125, -1)

        for i in range(2):
            track_lead.add_note(6, 0.125, base_num=-1)
            track_lead.wait(0.125)
            track_lead.add_note(6, 0.125, base_num=-1)
            track_lead.add_note(1, 0.125, base_num=1)
            track_lead.add_note(7, 0.125)
            track_lead.add_note(1, 0.125, base_num=1)
            track_lead.add_note(7, 0.125)
            track_lead.add_note(1, 0.125, base_num=1)

    def end(self):
        track_lead = self.mid.get_extended_track('Lead')
        for i in range(4):
            track_lead.add_note(1, 0.125)
        for i in range(4):
            track_lead.add_note(3, 0.125)

        for i in range(4):
            track_lead.add_note(2, 0.125)
        for i in range(4):
            track_lead.add_note(5, 0.125)

        for i in range(12):
            track_lead.add_note(6, 0.125)
        track_lead.add_note(2, 0.125, base_num=1)
        track_lead.add_note(1, 0.125, base_num=1)
        track_lead.add_note(7, 0.125)
        track_lead.add_note(5, 0.125)

        track_lead.add_note(5, 0.125, base_num=-1)
        track_lead.wait(7.875)

if __name__ == '__main__':
    coffin = CoffinDance()
    coffin.write_coffin()
    coffin.mid.save_midi()
    coffin.mid.play_it()