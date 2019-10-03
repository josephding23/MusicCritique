from midi.utility import *

bpm = 75

def verse(track):
    add_note(1, 0.5, track)       # 小
    add_note(1, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.3, 0.3, 2, 0.3, 0]})       # 时
    add_note(1, 1.5, track, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})       # 候
    add_note(7, 0.25, track, -1)  # 妈
    add_note(6, 0.25, track, -1)  # 妈

    add_note(5, 0.5, track, -1, channel=1)  # 对
    add_note(2, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.4, 0.8, 2, 0, 0]})      # 我
    add_note(3, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 640, 'wheel_times': 8})        # 讲

    add_note(3, 0.5, track)           # 大
    add_note(3, 0.5, track, pitch_type=2, bend_setting={'pitch': 3000, 'PADRA': [0.7, 0.3, 2, 0.3, 0]})
    add_note(3, 1.5, track, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 6})           # 海
    add_note(2, 0.25, track)          # 就
    add_note(1, 0.25, track)          # 是

    add_note(6, 0.5, track, -1, channel=1)  # 我
    add_note(1, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.6, 0.8, 2, 0, 0]})      # 故
    add_note(2, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 600, 'wheel_times': 8})        # 乡

    add_note(7, 0.5, track, -1)  # 海
    add_note(1, 0.5, track)
    add_note(7, 1.5, track, -1, tremble_setting={'pitch': 500, 'wheel_times': 6})  # 边
    add_note(6, 0.25, track, -1)
    add_note(5, 0.25, track, -1)

    add_note(5, 0.5, track, -1, channel=1)  # 出
    add_note(1, 0.5, track, channel=1, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.6, 0.8, 2, 0, 0]})
    add_note(2, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 400, 'wheel_times': 8})        # 生

    add_note(3, 1.5, track, pitch_type=2, bend_setting={'pitch': 3000, 'PADRA': [0.1, 0.3, 3, 0, 0]})       # 海
    add_note(3, 0.5, track)       # 里
    add_note(1, 0.5, track, channel=1)       # 成
    add_note(6, 0.5, track, -1, channel=1)

    add_note(1, 3, track, channel=1, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})         # 长


def chorus(track, num):
    add_note(5, 0.5, track)  # 大
    add_note(5, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.3, 0.5, 2, 0, 0]})
    add_note(5, 1.5, track, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})  # 海
    add_note(3, 0.5, track)  # 啊

    add_note(5, 0.5, track, channel=1)  # 大
    if num == 1:
        add_note(5, 0.6, track, channel=1, velocity=1.2, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.1, 0.6, 2, 0.5, 0.3]})
        add_note(5, 1.9, track, channel=1, pitch_type=1, tremble_setting={'pitch': 1000, 'wheel_times': 6})    # 海
    if num == 2:
        add_note(5, 0.65, track, channel=1, velocity=1.2, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.1, 0.9, 2, 0.9, 0.3]})
        add_note(5, 1.85, track, channel=1, pitch_type=1, tremble_setting={'pitch': 1300, 'wheel_times': 6})  # 海
    add_note(5, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.8, 1.3, 1.8, 0.8, 0]})  # 是（就）
    add_note(5, 0.5, track)  # 我（像）
    add_note(4, 0.5, track)  # 生（妈）
    if num == 1:
        add_note(1, 0.25, track, channel=1) # 活
        add_note(1, 0.25, track, channel=1) # 的
    if num == 2:
        add_note(1, 0.5, track, channel=1)  # (妈)
    add_note(5, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.8, 1.7, 1.8, 0.8, 0]})      # 地（一）
    add_note(5, 0.6, track, channel=1)

    add_note(5, 2.9, track, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})        # 方（样）

    add_note(3, 0.5, track)  # 海（走）
    add_note(4, 0.5, track)  # 风（遍）
    add_note(3, 1.5, track, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 6})  # 吹（天）
    add_note(2, 0.25, track) # (涯)
    add_note(1, 0.25, track)

    add_note(6, 0.5, track, -1, channel=1)  # 海（海）
    add_note(1, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0.3, 1.4, 1.8, 0.8, 0]})      # 浪
    add_note(2, 2, track, channel=1, pitch_type=1, tremble_setting={'pitch': 800, 'wheel_times': 10})        # 涌（角）

    if num == 2:
        set_bpm(track, 70)
    add_note(4, 0.5, track)              # 随（总）
    add_note(4, 0.5, track, pitch_type=2, bend_setting={'pitch': 6000, 'PADRA': [0, 0.7, 1.8, 0.6, 0]})              # 我（在）
    if num == 2:
        set_bpm(track, 65)
    add_note(4, 0.5, track)              # 漂（我）
    add_note(4, 0.5, track, pitch_type=2, bend_setting={'pitch': -6000, 'PADRA': [0, 0.5, 1.8, 0.8, 0]})              # 流（的）
    if num == 2:
        set_bpm(track, 60)
    add_note(1, 0.5, track, channel=1)   # 四（身）
    add_note(6, 0.5, track, -1, channel=1)

    if num == 2:
        set_bpm(track, 50)
    add_note(1, 3, track, channel=1, pitch_type=1, tremble_setting={'pitch': 1000, 'wheel_times': 8})     # 方（旁）

def chord(track):
    format = [0, 1, 2, 3, 2, 1]
    add_chord('C', 'M7', format, 3, track)
    add_chord('E', 'm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track, -1)
    add_chord('D', 'm7', format, 3, track)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3,  track)

    add_chord('C', 'M7', format, 3,  track)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)

    add_chord('C', 'M7', format, 3,  track)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('F', 'M7', format, 3, track, -1)
    add_chord('G', 'M7', format, 3, track, -1)
    add_chord('C', 'M7', format, 3, track)
    add_chord('D', 'm7', format, 3, track)
    add_chord('G', 'Mm7', format, 3, track)
    add_chord('C', 'M7', format, 3, track)

def bass_line1(track):
    add_bass(1, 0.25, track)
    add_bass(1, 1.25, track)
    add_bass(5, 1.25, track, base_num=-2)
    add_bass(1, 0.25, track, base_num=0, channel=8, velocity=0.9)
    add_bass(3, 0.25, track, -1)
    add_bass(3, 1.25, track, -1)
    add_bass(5, 1.25, track, base_num=-2)
    add_bass(5, 0.25, track, channel=8, velocity=0.9)

    add_bass(1, 0.25, track)
    add_bass(1, 1.25, track)
    add_bass(3, 1.25, track, base_num=-2)
    add_bass(5, 0.25, track, channel=8, velocity=0.9)
    add_bass(2, 0.25, track, -1)
    add_bass(2, 1.25, track, -1)
    add_bass(4, 1.25, track, base_num=-2)
    add_bass(1, 0.25, track, channel=8, velocity=0.9)

    add_bass(5, 0.25, track, -2)
    add_bass(5, 1.25, track, -2)
    add_bass(3, 1.25, track, base_num=-2)
    add_bass(3, 0.25, track, base_num=0, channel=8, velocity=0.9)
    add_bass(2, 0.25, track, -1)
    add_bass(2, 1.25, track, -1)
    add_bass(4, 1.25, track, base_num=-2)
    add_bass(1, 0.25, track, channel=8, velocity=0.9)

    add_bass(5, 0.25, track, -2)
    add_bass(5, 1.25, track, -2)
    add_bass(3, 1.25, track, base_num=-2)
    add_bass(1, 0.25, track, base_num=0, channel=8, velocity=0.9)
    add_bass(5, 0.25, track, -2)
    add_bass(5, 0.25, track, -2)
    add_bass(3, 0.25, track, -2)
    add_bass(5, 0.25, track, channel=8, velocity=0.9)
    add_bass(5, 0.25, track, -2)
    add_bass(7, 0.25, track, channel=8, velocity=0.9)
    add_bass(7, 0.25, track, -2)
    add_bass(1, 0.25, track, base_num=0, channel=8, velocity=0.9)
    add_bass(3, 0.25, track)
    add_bass(3, 0.25, track, base_num=0, channel=8, velocity=0.9)
    add_bass(5, 0.25, track)
    add_bass(6, 0.25, track, base_num=0, channel=8, velocity=0.9)

def bass_line2(track):
    pass

def hi_hat(track):
    for i in range(8):
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
    for i in range(16):
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.25, track, velocity=0.6)
        add_drum('open_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)
        add_drum('closed_hi-hat', 0.5, track, velocity=0.6)


def tom_and_snare_pt1(track):
    for i in range(7):
        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low_tom', 0.5, track, velocity=0.6)
        add_drum('low-mid_tom', 0.5, track, velocity=0.6)
        add_drum('acoustic_snare', 0.5, track, velocity=0.6)
        add_drum('low_tom', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.5, track, velocity=0.8)
    add_drum('low_tom', 0.25, track, velocity=0.6)
    add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)


def tom_and_snare_pt2(track, num):
    for i in range(3): #6
        add_drum('acoustic_snare', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.25, track, velocity=0.8)
        add_drum('low_tom', 0.5, track, velocity=0.6)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

        add_drum('acoustic_snare', 0.5, track, velocity=0.6)
        add_drum('low_tom', 0.25, track, velocity=0.8)
        add_drum('acoustic_snare', 0.5, track, velocity=0.9)
        add_drum('low-mid_tom', 0.125, track, velocity=0.6)
        add_drum('hi-mid_tom', 0.125, track, velocity=0.6)

        add_drum('acoustic_snare', 0.5, track, velocity=0.8)
        add_drum('low_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('acoustic_snare', 0.125, track, velocity=0.8)
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('high_tom', 0.25, track, velocity=0.6)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
        add_drum('low-mid_tom', 0.25, track, velocity=0.6)

    add_drum('acoustic_snare', 0.5, track, velocity=0.8)
    add_drum('low_tom', 0.25, track, velocity=0.6)
    add_drum('low-mid_tom', 0.25, track, velocity=0.6)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.125, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)
    add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    add_drum('acoustic_snare', 0.25, track, velocity=0.8)

    add_drum('high_tom', 0.125, track, velocity=0.9)
    add_drum('high_tom', 0.125, track, velocity=0.9)
    add_drum('hi-mid_tom', 0.125, track, velocity=0.9)
    add_drum('hi-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low-mid_tom', 0.125, track, velocity=0.9)
    add_drum('low_tom', 0.125, track, velocity=0.9)
    add_drum('low_tom', 0.125, track, velocity=0.9)

    add_drum('high_floor_tom', 0.25, track, velocity=1.1)
    add_drum('high_floor_tom', 0.25, track, velocity=1.1)
    add_drum('low_floor_tom', 0.25, track, velocity=1.2)
    add_drum('low_floor_tom', 0.25, track, velocity=1.2)

    if num == 1:
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal1', 0.25, track, velocity=1.2)
        add_drum('crash_cymbal2', 0.25, track, velocity=1.2)
    if num == 2:
        add_drum('low_floor_tom', 0.25, track, velocity=1.3)
        add_drum('low_floor_tom', 0.25, track, velocity=1.3)
        add_drum('chinese_cymbal', 1, track, velocity=1.9)


def write_song(file):
    mid = MidiFile(type=1)
    mid.charset = 'utf-8'

    melody_track = MidiTrack()
    set_track_meta_info(melody_track, 'Melody', '3/4', bpm, 'C', {'0': 30, '1': 30})
    mid.tracks.append(melody_track)
    verse(melody_track)
    # chorus(melody_track, 1)
    # chorus(melody_track, 2)

    chord_track = MidiTrack()
    set_track_meta_info(chord_track, 'Chord', '3/4', bpm, 'C', {'3': 25})
    mid.tracks.append(chord_track)
    chord(chord_track)

    bass_track = MidiTrack()
    set_track_meta_info(bass_track, 'Bass', '3/4', bpm, 'C', {'6': 32, '7': 35, '8': 36})
    mid.tracks.append(bass_track)
    bass_line1(bass_track)


    hi_hat_drum_track = MidiTrack()
    set_track_meta_info(hi_hat_drum_track, 'Hi-Hat', '3/4', bpm, 'C', {})
    mid.tracks.append(hi_hat_drum_track)
    hi_hat(hi_hat_drum_track)

    tom_and_snare_track = MidiTrack()
    set_track_meta_info(tom_and_snare_track, 'Tom And Snare', '3/4', bpm, 'C', {})
    mid.tracks.append(tom_and_snare_track)
    tom_and_snare_pt1(tom_and_snare_track)
    # tom_and_snare_pt2(tom_and_snare_track, 1)
    # tom_and_snare_pt2(tom_and_snare_track, 2)

    mid.save(file)

if __name__ == '__main__':
    roundabout_ori = '../../music/midi/read/roundabout.mid'
    roundabout_bass_only = '../../music/midi/read/roundabout_bass_only.mid'
    single_track_attempt = '../music/midi/write/attempt.mid'
    # play_midi(roundabout_ori)
    # print_track_msgs(roundabout_ori, 4)
    write_song(single_track_attempt)
    play_midi(single_track_attempt)