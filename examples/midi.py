from mido import Message, MidiFile, MidiTrack, MetaMessage
import mido
import pygame

bpm = 75

def print_tracks(file):
    cv1 = MidiFile(file)
    for track in cv1.tracks:
        print(track)

def play_note(note, length, track, base_num=0, delay=0, velocity=1.0, channel=0):
    meta_time = 60 * 60 * 10 / bpm
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    base_note = 60
    track.append(Message('note_on', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(delay*meta_time), channel=channel))
    track.append(Message('note_off', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(meta_time*length), channel=channel))

def wait(time, track):
    meta_time = 60 * 60 * 10 / bpm
    track.append(Message('note_off', time=round(meta_time * time)))

def verse(track):
    play_note(1, 0.5, track)       # 小
    play_note(2, 0.5, track)       # 时
    play_note(1, 1.5, track)       # 候
    play_note(7, 0.25, track, -1)  # 妈
    play_note(6, 0.25, track, -1)  # 妈

    play_note(5, 0.5, track, -1, channel=1)  # 对
    play_note(3, 0.5, track, channel=1)      # 我
    play_note(3, 2, track, channel=1)        # 讲

    play_note(3, 0.5, track)           # 大
    play_note(4, 0.5, track)
    play_note(3, 1.5, track)           # 海
    play_note(2, 0.25, track)          # 就
    play_note(1, 0.25, track)          # 是

    play_note(6, 0.5, track, -1, channel=1)  # 我
    play_note(2, 0.5, track, channel=1)      # 故
    play_note(2, 2, track, channel=1)        # 乡

    play_note(7, 0.5, track, -1)  # 海
    play_note(1, 0.5, track)
    play_note(7, 1.5, track, -1)  # 边
    play_note(6, 0.25, track, -1)
    play_note(5, 0.25, track, -1)

    play_note(5, 0.5, track, -1, channel=1)  # 出
    play_note(2, 0.5, track, channel=1)
    play_note(2, 2, track, channel=1)        # 生

    play_note(4, 1.5, track)       # 海
    play_note(3, 0.5, track)       # 里
    play_note(1, 0.5, track, channel=1)       # 成
    play_note(6, 0.5, track, -1, channel=1)

    play_note(1, 3, track, channel=1)         # 长


def chorus(track, num):
    play_note(5, 0.5, track)  # 大
    play_note(6, 0.5, track)
    play_note(5, 1.5, track)  # 海
    play_note(3, 0.5, track)  # 啊

    play_note(5, 0.5, track, channel=1)  # 大
    play_note(6, 0.5, track, channel=1)
    play_note(5, 2, track, channel=1)    # 海

    play_note(6, 0.5, track)  # 是（就）
    play_note(5, 0.5, track)  # 我（像）
    play_note(4, 0.5, track)  # 生（妈）
    if num == 1:
        play_note(1, 0.25, track, channel=1) # 活
        play_note(1, 0.25, track, channel=1) # 的
    if num == 2:
        play_note(1, 0.5, track, channel=1)  # (妈)
    play_note(6, 0.5, track, channel=1)      # 地（一）
    play_note(5, 0.5, track, channel=1)

    play_note(5, 3, track, channel=1)        # 方（样）

    play_note(3, 0.5, track)  # 海（走）
    play_note(4, 0.5, track)  # 风（遍）
    play_note(3, 1.5, track)  # 吹（天）
    play_note(2, 0.25, track) # (涯)
    play_note(1, 0.25, track)

    play_note(6, 0.5, track, -1, channel=1)  # 海（海）
    play_note(2, 0.5, track, channel=1)      # 浪
    play_note(2, 2, track, channel=1)        # 涌（角）

    play_note(4, 0.5, track)              # 随（总）
    play_note(5, 0.5, track)              # 我（在）
    play_note(4, 0.5, track)              # 漂（我）
    play_note(3, 0.5, track)              # 流（的）
    play_note(1, 0.5, track, channel=1)   # 四（身）
    play_note(6, 0.5, track, -1, channel=1)

    play_note(1, 3, track, channel=1)     # 方（旁）

def write_midi_with_single_track(file):
    mid = MidiFile()
    mid.charset = 'utf-8'

    tempo = mido.bpm2tempo(bpm)

    meta_instrument = MetaMessage('instrument_name', name='Slap Bass 1')
    meta_time = MetaMessage('time_signature', numerator=3, denominator=4)
    meta_tempo = MetaMessage('set_tempo', tempo = tempo, time=0)
    meta_tone = MetaMessage('key_signature', key='C')


    track = MidiTrack()
    track.name = 'Mono'
    mid.tracks.append(track)

    track.append(meta_instrument)
    track.append(meta_time)
    track.append(meta_tone)
    track.append(meta_tempo)

    track.append(Message('program_change', channel=0, program=33, time=0))
    track.append(Message('program_change', channel=1, program=36, time=0))

    verse(track)
    chorus(track, 1)
    chorus(track, 2)

    mid.save(file)

def print_track_msgs(file, no):
    for msg in MidiFile(file).tracks[no]:
        print(msg)

    # cv3 = mido.MidiFile()
    # cv3.tracks.append(cv1.tracks[6])
    # cv3.save('../music/midi/roundabout_bass_only.mid')

def play_midi(file):
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(file)
    except:
        import traceback
        print(traceback.format_exc())
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

if __name__ == '__main__':
    roundabout_ori = '../music/midi/read/roundabout.mid'
    roundabout_bass_only = '../music/midi/read/roundabout_bass_only.mid'
    single_track_attempt = '../music/midi/write/attempt.mid'
    # play_midi(roundabout_ori)
    # print_track_msgs(roundabout_ori, 4)
    write_midi_with_single_track(single_track_attempt)
    play_midi(single_track_attempt)