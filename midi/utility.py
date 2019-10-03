from mido import Message, MidiFile, MidiTrack, MetaMessage
from fractions import Fraction
import mido
import pygame
import traceback

def print_tracks(file):
    cv1 = MidiFile(file)
    for track in cv1.tracks:
        print(track)

def get_bpm(track):
    for msg in track:
        if msg.type == 'set_tempo':
            return mido.tempo2bpm(msg.tempo)
    return 0

def set_bpm(track, bpm):
    tempo = mido.bpm2tempo(bpm)
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

def set_track_meta_info(track, name, time, bpm, key, instruments):
    track.name = name
    tempo = mido.bpm2tempo(bpm)
    numerator = Fraction(time).numerator
    denominator = Fraction(time).denominator
    track.append(MetaMessage('time_signature', numerator=numerator, denominator=denominator))
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(MetaMessage('key_signature', key=key))
    for channel, program in instruments.items():
        track.append(Message('program_change', channel=int(channel), program=program, time=0))

def get_chord_arrangement(name):
    maj3 = [0, 4, 7, 0]  # 大三和弦 根音-大三度-纯五度
    min3 = [0, 3, 7, 0]  # 小三和弦 根音-小三度-纯五度
    aug3 = [0, 4, 6, 0]  # 增三和弦 根音-大三度-三全音
    dim3 = [0, 3, 6, 0]  # 减三和弦 根音-小三度-三全音

    M7 = [0, 4, 7, 11]  # 大七和弦 根音-大三度-纯五度-大七度
    Mm7 = [0, 4, 7, 10]  # 属七和弦 根音-大三度-纯五度-小七度
    m7 = [0, 3, 7, 10]  # 小七和弦 根音-小三度-纯五度-小七度
    mM7 = [0, 3, 7, 11]  # 小大七和弦 根音-小三度-纯五度-大七度
    aug7 = [0, 4, 6, 11]  # 增大七和弦 根音-大三度-三全音-大七度
    m7b5 = [0, 3, 6, 10]  # 半减七和弦 根音-小三度-三全音-小七度
    dim7 = [0, 3, 6, 9]  # 减减七和弦 根音-小三度-三全音-减七度

    chord = [0, 0, 0, 0]
    if name == 'maj3':
        chord = maj3
    if name == 'min3':
        chord = min3
    if name == 'aug3':
        chord = aug3
    if name == 'dim3':
        chord = dim3
    if name == 'M7':
        chord = M7
    if name == 'Mm7':
        chord = Mm7
    if name == 'm7':
        chord = m7
    if name == 'mM7':
        chord = mM7
    if name == 'aug7':
        chord = aug7
    if name == 'm7b5':
        chord = m7b5
    if name == 'dim7':
        chord = dim7

    return chord

def add_chord(root, name, format, length, track, root_base=0, channel=3):
    bpm = get_bpm(track)
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    notes_dict = {}
    for i, note in enumerate(notes):
        notes_dict[note] = 60 + sum(major_notes[0:i+1])
    root_note = notes_dict[root] + root_base*12
    chord = get_chord_arrangement(name)
    meta_time = 60 * 60 * 10 / bpm
    time = round(length / len(format) * meta_time)

    for dis in format:
        note = root_note + chord[dis]
        track.append(Message('note_on', note=note, velocity=56, time=0, channel=channel))
        track.append(Message('note_off', note=note, velocity=56, time=time, channel=channel))

def add_bass(note, length, track, base_num=-1, velocity=0.7, channel=6, delay=0):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    base_note = 60
    track.append(
        Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]), velocity=round(64 * velocity),
                time=round(delay * meta_time), channel=channel))
    track.append(
        Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]), velocity=round(64 * velocity),
                time=round(meta_time * length), channel=channel))

def add_note(note, length, track, base_num=0, delay=0, velocity=1.0, channel=0, pitch_type=0, tremble_setting=None, bend_setting=None):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
    base_note = 60
    if pitch_type == 0: # No Pitch Wheel Message
        track.append(Message('note_on', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(delay*meta_time), channel=channel))
        track.append(Message('note_off', note=base_note + base_num*12 + sum(major_notes[0:note]), velocity=round(64*velocity), time=round(meta_time*length), channel=channel))
    elif pitch_type == 1: # Tremble
        try:
            pitch = tremble_setting['pitch']
            wheel_times = tremble_setting['wheel_times']
            track.append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity),
                                 time=round(delay * meta_time), channel=channel))
            for i in range(wheel_times):
                track.append(Message('pitchwheel', pitch=pitch, time=round(meta_time * length / (2 * wheel_times)),
                                     channel=channel))
                track.append(Message('pitchwheel', pitch=0, time=0, channel=channel))
                track.append(Message('pitchwheel', pitch=-pitch, time=round(meta_time * length / (2 * wheel_times)),
                                     channel=channel))
            track.append(Message('pitchwheel', pitch=0, time=0, channel=channel))
            track.append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=0, channel=channel))
        except:
            print(traceback.format_exc())
    elif pitch_type == 2: # Bend
        try:
            pitch = bend_setting['pitch']
            PASDA = bend_setting['PADRA'] # Prepare-Attack-Sustain-Decay-Aftermath (Taken the notion of ADSR)
            prepare_rate = PASDA[0] / sum(PASDA)
            attack_rate = PASDA[1] / sum(PASDA)
            sustain_rate = PASDA[2] / sum(PASDA)
            decay_rate = PASDA[3] / sum(PASDA)
            aftermath_rate = PASDA[4] / sum(PASDA)
            track.append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(delay * meta_time), channel=channel))
            track.append(Message('pitchwheel', pitch=0, time=round(meta_time * length * prepare_rate), channel=channel))
            track.append(Message('pitchwheel', pitch=pitch, time=round(meta_time * length * attack_rate), channel=channel))
            track.append(Message('aftertouch', time=round(meta_time * length * sustain_rate), channel=channel))
            track.append(Message('pitchwheel', pitch=0, time=round(meta_time * length * decay_rate), channel=channel))
            #track.append(Message('pitchwheel', pitch=0, time=round(meta_time * length * aftermath_rate), channel=channel))
            track.append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(meta_time * length * aftermath_rate), channel=channel))
        except:
            print(traceback.format_exc())

def wait(time, track):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    track.append(Message('note_off', time=round(meta_time * time)))

def get_drum_dict():
    drum_dict = {
        'acoustic_bass': 35,
        'bass1': 36,
        'side_stick': 37,
        'acoustic_snare': 38,
        'hand_clap': 39,
        'electric_snare': 40,
        'low_floor_tom': 41,
        'closed_hi-hat': 42,
        'high_floor_tom': 43,
        'pedal_hi-hat': 44,
        'low_tom': 45,
        'open_hi-hat': 46,
        'low-mid_tom': 47,
        'hi-mid_tom': 48,
        'crash_cymbal1': 49,
        'high_tom': 50,
        'ride_cymbal1': 51,
        'chinese_cymbal': 52,
        'ride_bell': 53,
        'tambourine': 54,
        'splash_cymbal': 55,
        'cowbell': 56,
        'crash_cymbal2': 57,
        'vibraslap': 58,
        'ride_cymbal2': 59,
        'hi_bongo': 60,
        'low_bongo': 61,
        'mute_hi_bongo': 62,
        'open_hi_bongo': 63,
        'low_conga': 64,
        'high_timbale': 65,
        'low_timbale': 66,
        'high_agogo': 67,
        'low_agogo': 68,
        'cabasa': 69,
        'maracas': 70,
        'short_whistle': 71,
        'long_whistle': 72,
        'short_guiro': 73,
        'long_guiro': 74,
        'claves': 75,
        'hi_wood_block': 76,
        'low_wood_block': 77,
        'mute_cuica': 78,
        'open_cuica': 79,
        'mute_triangle': 80,
        'open_triangle': 81
    }

    return drum_dict

def add_drum(name, time, track, delay=0, velocity=1):
    bpm = get_bpm(track)
    meta_time = 60 * 60 * 10 / bpm
    drum_dict = get_drum_dict()
    try:
        note = drum_dict[name]
    except:
        print(traceback.format_exc())
        return
    track.append(
        Message('note_on', note=note, velocity=round(64 * velocity),
                time=delay, channel=9))
    track.append(
        Message('note_off', note=note, velocity=round(64 * velocity),
                time=round(meta_time * time), channel=9))


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
    attempt_bend_path = '../music/midi/bent_experiment.mid'
    mid = MidiFile(type=1)
    mid.charset = 'utf-8'

    melody_track = MidiTrack()
    set_track_meta_info(melody_track, 'Melody', '3/4', 75, 'C', {'0': 30, '1': 30})
    mid.tracks.append(melody_track)
