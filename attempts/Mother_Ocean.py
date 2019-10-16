from midi_extended.UtilityBox import *
from midi_extended.MidiFileExtended import MidiFileExtended

class Mother_Ocean():
    def __init__(self):
        self.bpm = 75
        self.time_signature = '3/4'
        self.key = 'C'
        self.file_path = '../data/midi/write/mother_ocean.mid'
        self.mid = MidiFileExtended(self.file_path, type=1, mode='w')

    def verse(self):
        
        track = self.mid.get_extended_track('Melody')
        track.add_note(1, 0.5)       # 小
        track.add_note(1, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.3, 2, 0.3, 0]})       # 时
        track.add_note(1, 1.5, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})       # 候
        track.add_note(7, 0.25, -1)  # 妈
        track.add_note(6, 0.25, -1)  # 妈
    
        track.add_note(5, 0.5, -1, channel=1)  # 对
        track.add_note(2, 0.5, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.8, 2, 0, 0]})      # 我
        track.add_note(3, 2, channel=1, pitch_type=1, tremble_setting={'pitch': 640, 'wheel_times': 8})        # 讲
    
        track.add_note(3, 0.5)           # 大
        track.add_note(3, 0.5, pitch_type=2, bend_setting={'pitch': 3000, 'PASDA': [0.1, 0.8, 2, 0.3, 0]})
        track.add_note(3, 1.5, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 6})           # 海
        track.add_note(2, 0.25)          # 就
        track.add_note(1, 0.25)          # 是
    
        track.add_note(6, 0.5, -1, channel=1)  # 我
        track.add_note(1, 0.5, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.2, 0.8, 2, 0, 0]})      # 故
        track.add_note(2, 2, channel=1, pitch_type=1, tremble_setting={'pitch': 600, 'wheel_times': 8})        # 乡
    
        track.add_note(7, 0.5, -1)  # 海
        track.add_note(1, 0.5)
        track.add_note(7, 1.5, -1, tremble_setting={'pitch': 500, 'wheel_times': 6})  # 边
        track.add_note(6, 0.25, -1)
        track.add_note(5, 0.25, -1)
    
        track.add_note(5, 0.5, -1, channel=1)  # 出
        track.add_note(1, 0.5, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.2, 1.5, 3, 0, 0]})
        track.add_note(2, 2, channel=1, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 8})        # 生
    
        track.add_note(3, 1.5, pitch_type=2, bend_setting={'pitch': 3000, 'PASDA': [0, 0.3, 3, 0, 0]})       # 海
        track.add_note(3, 0.5)       # 里
        track.add_note(1, 0.5, channel=1)       # 成
        track.add_note(6, 0.5, -1, channel=1)
    
        track.add_note(1, 3, channel=1, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})         # 长
    
    
    def chorus(self, num):
        track = self.mid.get_extended_track('Melody')
        track.add_note(5, 0.5)  # 大
        track.add_note(5, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.5, 2, 0, 0]})
        track.add_note(5, 1.5, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})  # 海
        track.add_note(3, 0.5)  # 啊
    
        track.add_note(5, 0.5, channel=1)  # 大
        if num == 1:
            track.add_note(5, 0.6, channel=1, velocity=1.2, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.6, 2, 0.5, 0.3]})
            track.add_note(5, 1.9, channel=1, pitch_type=1, tremble_setting={'pitch': 1000, 'wheel_times': 6})    # 海
        if num == 2:
            track.add_note(5, 0.65, channel=1, velocity=1.2, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 0.9, 2, 0.9, 0.3]})
            track.add_note(5, 1.85, channel=1, pitch_type=1, tremble_setting={'pitch': 1300, 'wheel_times': 6})  # 海
        track.add_note(5, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.8, 1.3, 1.8, 0.8, 0]})  # 是（就）
        track.add_note(5, 0.5)  # 我（像）
        track.add_note(4, 0.5)  # 生（妈）
        if num == 1:
            track.add_note(1, 0.25, channel=1) # 活
            track.add_note(1, 0.25, channel=1) # 的
        if num == 2:
            track.add_note(1, 0.5, channel=1)  # (妈)
        track.add_note(5, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0, 1.7, 1.8, 0.8, 0]})      # 地（一）
        track.add_note(5, 0.6, channel=1)
    
        track.add_note(5, 2.9, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})        # 方（样）
    
        track.add_note(3, 0.5)  # 海（走）
        track.add_note(4, 0.5)  # 风（遍）
        track.add_note(3, 1.5, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 6})  # 吹（天）
        track.add_note(2, 0.25) # (涯)
        track.add_note(1, 0.25)
    
        track.add_note(6, 0.5, -1, channel=1)  # 海（海）
        track.add_note(1, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0.1, 1.4, 1.8, 0.8, 0]})      # 浪
        track.add_note(2, 2, channel=1, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})        # 涌（角）
    
        if num == 2:
            track.set_bpm(70)
        track.add_note(4, 0.5)              # 随（总）
        track.add_note(4, 0.5, pitch_type=2, bend_setting={'pitch': 6000, 'PASDA': [0, 0.7, 1.8, 0.6, 0]})              # 我（在）
        if num == 2:
            track.set_bpm(65)
        track.add_note(4, 0.5)              # 漂（我）
        track.add_note(4, 0.5, pitch_type=2, bend_setting={'pitch': -6000, 'PASDA': [0, 0.5, 1.8, 0.8, 0]})              # 流（的）
        if num == 2:
            track.set_bpm(60)
        track.add_note(1, 0.5, channel=1)   # 四（身）
        track.add_note(6, 0.5, -1, channel=1)
    
        if num == 2:
            track.set_bpm(50)
        track.add_note(1, 3, channel=1, pitch_type=1, tremble_setting={'pitch': 1000, 'wheel_times': 8})     # 方（旁）

    def verse_simple(self):

        track = self.mid.get_extended_track('Melody')
        track.add_note(1, 0.5)  # 小
        track.add_note(2, 0.5)  # 时
        track.add_note(1, 1.5)  # 候
        track.add_note(7, 0.25, -1)  # 妈
        track.add_note(6, 0.25, -1)  # 妈

        track.add_note(5, 0.5, -1)  # 对
        track.add_note(3, 0.5)  # 我
        track.add_note(3, 2)  # 讲

        track.add_note(3, 0.5)  # 大
        track.add_note(4, 0.5)
        track.add_note(3, 1.5)  # 海
        track.add_note(2, 0.25)  # 就
        track.add_note(1, 0.25)  # 是

        track.add_note(6, 0.5, -1)  # 我
        track.add_note(2, 0.5)  # 故
        track.add_note(2, 2)  # 乡

        track.add_note(7, 0.5, -1)  # 海
        track.add_note(1, 0.5)
        track.add_note(7, 1.5, -1)  # 边
        track.add_note(6, 0.25, -1)
        track.add_note(5, 0.25, -1)

        track.add_note(5, 0.5, -1)  # 出
        track.add_note(2, 0.5)
        track.add_note(2, 2)  # 生

        track.add_note(4, 1.5)  # 海
        track.add_note(3, 0.5)  # 里
        track.add_note(1, 0.5)  # 成
        track.add_note(6, 0.5, -1)

        track.add_note(1, 3)  # 长

    def chorus_simple(self, num):
        track = self.mid.get_extended_track('Melody')
        track.add_note(5, 0.5)  # 大
        track.add_note(6, 0.5)
        track.add_note(5, 1.5)  # 海
        track.add_note(3, 0.5)  # 啊

        track.add_note(5, 0.5)  # 大
        track.add_note(6, 0.5)
        track.add_note(5, 2)  # 海
        track.add_note(6, 0.5)  # 是（就）
        track.add_note(5, 0.5)  # 我（像）
        track.add_note(4, 0.5)  # 生（妈）
        if num == 1:
            track.add_note(1, 0.25)  # 活
            track.add_note(1, 0.25)  # 的
        if num == 2:
            track.add_note(1, 0.5)  # (妈)
        track.add_note(6, 0.5)  # 地（一）
        track.add_note(5, 0.5)

        track.add_note(5, 3)  # 方（样）

        track.add_note(3, 0.5)  # 海（走）
        track.add_note(4, 0.5)  # 风（遍）
        track.add_note(3, 1.5)  # 吹（天）
        track.add_note(2, 0.25)  # (涯)
        track.add_note(1, 0.25)

        track.add_note(6, 0.5, -1)  # 海（海）
        track.add_note(2, 0.5)  # 浪
        track.add_note(2, 2)  # 涌（角）

        track.add_note(4, 0.5)  # 随（总）
        track.add_note(5, 0.5)  # 我（在）
        track.add_note(4, 0.5)  # 漂（我）
        track.add_note(3, 0.5)  # 流（的）
        track.add_note(1, 0.5)  # 四（身）
        track.add_note(6, 0.5, -1)

        track.add_note(1, 3)  # 方（旁）
    
    def chord(self):
        format = [0, 1, 2, 3, 2, 1]
        track = self.mid.get_extended_track('Chord')
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('E', 'm7', format, 3)
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('D', 'm7', format, 3)
        track.add_chord('G', 'Mm7', format, 3, -1)
        track.add_chord('D', 'm7', format, 3)
        track.add_chord('F', 'M7', format, 3, -1)
        track.add_chord('C', 'M7', format, 3)
    
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('G', 'M7', format, 3, -1)
        track.add_chord('F', 'M7', format, 3, -1)
        track.add_chord('G', 'M7', format, 3, -1)
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('D', 'm7', format, 3)
        track.add_chord('G', 'Mm7', format, 3)
        track.add_chord('C', 'M7', format, 3)
    
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('G', 'M7', format, 3, -1)
        track.add_chord('F', 'M7', format, 3, -1)
        track.add_chord('G', 'M7', format, 3, -1)
        track.add_chord('C', 'M7', format, 3)
        track.add_chord('D', 'm7', format, 3)
        track.add_chord('G', 'Mm7', format, 3)
        track.add_chord('C', 'M7', format, 3)
    
    def bass_line1(self):
        track = self.mid.get_extended_track('Bass')
        track.add_bass(1, 0.25)
        track.add_bass(1, 1.25)
        track.add_bass(5, 1.25, base_num=-2)
        track.add_bass(1, 0.25, base_num=0, channel=8, velocity=0.9)
        track.add_bass(3, 0.25, -1)
        track.add_bass(3, 1.25, -1)
        track.add_bass(5, 1.25, base_num=-2)
        track.add_bass(5, 0.25, channel=8, velocity=0.9)
    
        track.add_bass(1, 0.25)
        track.add_bass(1, 1.25)
        track.add_bass(3, 1.25, base_num=-2)
        track.add_bass(5, 0.25, channel=8, velocity=0.9)
        track.add_bass(2, 0.25, -1)
        track.add_bass(2, 1.25, -1)
        track.add_bass(4, 1.25, base_num=-2)
        track.add_bass(1, 0.25, channel=8, velocity=0.9)
    
        track.add_bass(5, 0.25, -2)
        track.add_bass(5, 1.25, -2)
        track.add_bass(3, 1.25, base_num=-2)
        track.add_bass(3, 0.25, base_num=0, channel=8, velocity=0.9)
        track.add_bass(2, 0.25, -1)
        track.add_bass(2, 1.25, -1)
        track.add_bass(4, 1.25, base_num=-2)
        track.add_bass(1, 0.25, channel=8, velocity=0.9)
    
        track.add_bass(5, 0.25, -2)
        track.add_bass(5, 1.25, -2)
        track.add_bass(3, 1.25, base_num=-2)
        track.add_bass(1, 0.25, base_num=0, channel=8, velocity=0.9)
        track.add_bass(5, 0.25, -2)
        track.add_bass(5, 0.25, -2)
        track.add_bass(3, 0.25, -2)
        track.add_bass(5, 0.25, channel=8, velocity=0.9)
        track.add_bass(5, 0.25, -2)
        track.add_bass(7, 0.25, channel=8, velocity=0.9)
        track.add_bass(7, 0.25, -2)
        track.add_bass(1, 0.25, base_num=0, channel=8, velocity=0.9)
        track.add_bass(3, 0.25)
        track.add_bass(3, 0.25, base_num=0, channel=8, velocity=0.9)
        track.add_bass(5, 0.25)
        track.add_bass(6, 0.25, base_num=0, channel=8, velocity=0.9)
    
    def bass_line2(track):
        pass
    
    def hi_hat(self):
        track = self.mid.get_extended_track('Hi-Hat')
        for i in range(8):
            track.add_drum('open_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
            track.add_drum('open_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
        for i in range(16):
            track.add_drum('open_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.25, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.25, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.25, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.25, velocity=0.6)
            track.add_drum('open_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
            track.add_drum('closed_hi-hat', 0.5, velocity=0.6)
    
    
    def tom_and_snare_pt1(self):
        track = self.mid.get_extended_track('Tom And Snare')
        for i in range(7):
            track.add_drum('acoustic_snare', 0.5, velocity=0.8)
            track.add_drum('low_tom', 0.5, velocity=0.6)
            track.add_drum('low-mid_tom', 0.5, velocity=0.6)
            track.add_drum('acoustic_snare', 0.5, velocity=0.6)
            track.add_drum('low_tom', 0.25, velocity=0.8)
            track.add_drum('acoustic_snare', 0.5, velocity=0.8)
            track.add_drum('low-mid_tom', 0.25, velocity=0.6)
        track.add_drum('acoustic_snare', 0.5, velocity=0.8)
        track.add_drum('low_tom', 0.25, velocity=0.6)
        track.add_drum('low-mid_tom', 0.25, velocity=0.6)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('acoustic_snare', 0.125, velocity=0.8)
        track.add_drum('acoustic_snare', 0.125, velocity=0.8)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('crash_cymbal1', 0.25, velocity=1.2)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('crash_cymbal2', 0.25, velocity=1.2)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
    
    
    def tom_and_snare_pt2(self, num):
        track = self.mid.get_extended_track('Tom And Snare')
        for i in range(3): #6
            track.add_drum('acoustic_snare', 0.25, velocity=0.8)
            track.add_drum('acoustic_snare', 0.25, velocity=0.8)
            track.add_drum('low_tom', 0.5, velocity=0.6)
            track.add_drum('high_tom', 0.25, velocity=0.6)
            track.add_drum('low-mid_tom', 0.25, velocity=0.6)
    
            track.add_drum('acoustic_snare', 0.5, velocity=0.6)
            track.add_drum('low_tom', 0.25, velocity=0.8)
            track.add_drum('acoustic_snare', 0.5, velocity=0.9)
            track.add_drum('low-mid_tom', 0.125, velocity=0.6)
            track.add_drum('hi-mid_tom', 0.125, velocity=0.6)
    
            track.add_drum('acoustic_snare', 0.5, velocity=0.8)
            track.add_drum('low_tom', 0.25, velocity=0.6)
            track.add_drum('low-mid_tom', 0.25, velocity=0.6)
            track.add_drum('high_tom', 0.25, velocity=0.6)
            track.add_drum('low-mid_tom', 0.25, velocity=0.6)
    
            track.add_drum('acoustic_snare', 0.125, velocity=0.8)
            track.add_drum('acoustic_snare', 0.125, velocity=0.8)
            track.add_drum('acoustic_snare', 0.125, velocity=0.8)
            track.add_drum('acoustic_snare', 0.125, velocity=0.8)
            track.add_drum('crash_cymbal1', 0.25, velocity=1.2)
            track.add_drum('high_tom', 0.25, velocity=0.6)
            track.add_drum('crash_cymbal2', 0.25, velocity=1.2)
            track.add_drum('low-mid_tom', 0.25, velocity=0.6)
    
        track.add_drum('acoustic_snare', 0.5, velocity=0.8)
        track.add_drum('low_tom', 0.25, velocity=0.6)
        track.add_drum('low-mid_tom', 0.25, velocity=0.6)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('acoustic_snare', 0.125, velocity=0.8)
        track.add_drum('acoustic_snare', 0.125, velocity=0.8)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('crash_cymbal1', 0.25, velocity=1.2)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
        track.add_drum('crash_cymbal2', 0.25, velocity=1.2)
        track.add_drum('acoustic_snare', 0.25, velocity=0.8)
    
        track.add_drum('high_tom', 0.125, velocity=0.9)
        track.add_drum('high_tom', 0.125, velocity=0.9)
        track.add_drum('hi-mid_tom', 0.125, velocity=0.9)
        track.add_drum('hi-mid_tom', 0.125, velocity=0.9)
        track.add_drum('low-mid_tom', 0.125, velocity=0.9)
        track.add_drum('low-mid_tom', 0.125, velocity=0.9)
        track.add_drum('low_tom', 0.125, velocity=0.9)
        track.add_drum('low_tom', 0.125, velocity=0.9)
    
        track.add_drum('high_floor_tom', 0.25, velocity=1.1)
        track.add_drum('high_floor_tom', 0.25, velocity=1.1)
        track.add_drum('low_floor_tom', 0.25, velocity=1.2)
        track.add_drum('low_floor_tom', 0.25, velocity=1.2)
    
        if num == 1:
            track.add_drum('crash_cymbal1', 0.25, velocity=1.2)
            track.add_drum('crash_cymbal2', 0.25, velocity=1.2)
            track.add_drum('crash_cymbal1', 0.25, velocity=1.2)
            track.add_drum('crash_cymbal2', 0.25, velocity=1.2)
        if num == 2:
            track.add_drum('low_floor_tom', 0.25, velocity=1.3)
            track.add_drum('low_floor_tom', 0.25, velocity=1.3)
            track.add_drum('chinese_cymbal', 1, velocity=1.9)
    
    
    def write_song(self):

        self.mid.add_new_track('Melody', self.time_signature, self.bpm, self.key, {'0': 30, '1': 30})
        self.verse()
        self.chorus(1)
        self.chorus(2)

        self.mid.add_new_track('Chord', self.time_signature, self.bpm, self.key, {'3': 25})
        self.chord()

        self.mid.add_new_track('Bass', self.time_signature, self.bpm, self.key, {'6': 32, '7': 35, '8': 36})
        self.bass_line1()

        self.mid.add_new_track('Hi-Hat', self.time_signature, self.bpm, self.key, {})
        self.hi_hat()

        self.mid.add_new_track('Tom And Snare', self.time_signature, self.bpm, self.key, {})
        self.tom_and_snare_pt1()
        self.tom_and_snare_pt2(1)
        self.tom_and_snare_pt2(2)

def piano_roll_test():
    path = '../data/midi/write/mother_ocean_simple.mid'
    mid = MidiFileExtended(path, 'r')

    mid.turn_track_into_numpy_matrix('Melody', "../data/numpy/mother_ocean/melody.npy")
    mid.generate_track_from_numpy_matrix("../data/numpy/mother_ocean/melody.npy", (288, 128), 'Melody', False)

    mid.turn_track_into_numpy_matrix('Chord', "../data/numpy/mother_ocean/chord.npy")
    mid.generate_track_from_numpy_matrix("../data/numpy/mother_ocean/chord.npy", (288, 128), 'Chord', False, True, '../data/plots/mother_ocean/chord.png')

    mid.turn_track_into_numpy_matrix('Hi-Hat', "../data/numpy/mother_ocean/hi-hat.npy")
    mid.generate_track_from_numpy_matrix("../data/numpy/mother_ocean/hi-hat.npy", (288, 128), 'Hi-Hat', True, True, '../data/plots/mother_ocean/hi-hat.png')

    mid.turn_track_into_numpy_matrix('Tom And Snare', "../data/numpy/mother_ocean/tom_and_snare.npy")
    mid.generate_track_from_numpy_matrix("../data/numpy/mother_ocean/tom_and_snare.npy", (288, 128), 'Tom And Snare', True, True, '../data/plots/mother_ocean/tom_and_snare.png')

    '''
    mid.generate_multiple_tracks_from_numpy_matrices(4, "../data/numpy/mother_ocean/",
                                                     ['melody.npy', 'chord.npy', 'hi-hat.npy', 'tom_and_snare.npy'],
                                                     (288, 128),
                                                     ['Melody', 'Chord', 'Hi-Hat', 'Tom And Snare'],
                                                     [False, False, True, True],
                                                     75.0, [0, 96, 192], 24, save_fig=True,
                                                     save_path='../data/plots/mother_ocean/multi.png')
    '''
if __name__ == '__main__':
    # piano_roll_test()

    mother_ocean = Mother_Ocean()
    mother_ocean.write_song()

    mother_ocean.mid.save_midi()
    mother_ocean.mid.play_it()
    # mother_ocean.mid.get_track_by_name('Melody').print_msgs()
    # mother_ocean.mid.play_it()

