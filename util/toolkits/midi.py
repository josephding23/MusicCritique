import traceback
import mido


def get_bpm_from_track(track):
    for msg in track:
        if msg.type == 'set_tempo':
            try:
                return mido.tempo2bpm(msg.tempo)
            except:
                print(traceback.format_exc())
                continue
    return None


def get_key_from_track(track):
    for msg in track:
        if msg.type == 'key_signature':
            try:
                return msg.key
            except:
                print(traceback.format_exc())
                continue
    return None


def get_time_signature_from_track(track):
    for msg in track:
        if msg.type == 'time_signature':
            try:
                numerator = msg.numerator
                denominator = msg.denominator
                return str(numerator) + '/' + str(denominator)
            except:
                print(traceback.format_exc())
                continue
    return None


def get_instruments_from_track(track):
    instruments = {}
    for msg in track:
        if msg.type == 'program_change':
            try:
                channel = str(msg.channel)
                program = msg.program
                instruments[channel] = program
            except:
                print(traceback.format_exc())
                continue
    return instruments


def calculate_track_duration(track):
    duration = 0
    for msg in track:
        if msg.type in ['note_on']:
            print(msg)
    print(duration)


def get_note_name_by_midi_value(midi, mode=1):
    name_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    if mode == 1:
        pass
    else:
        name_list = ['C', '♯C', 'D', '♯D', 'E', 'F', '♯F', 'G', '♯G', 'A', '♯A', 'B']
    range_num = 4
    name = 'C'
    for num in range(8):
        min_note = 12 + num * 12
        max_note = 12 + (num + 1) * 12
        if midi < max_note and midi >= min_note:
            range_num = num
            name = name_list[midi - min_note]
            break
    if midi > 108:
        range_num = 8
    return name + str(range_num)


def get_chord_arrangement(root, name):
    chord_dict = {
        'maj3': [0, 4, 7],  # 大三和弦 根音-大三度-纯五度
        'min3': [0, 3, 7],  # 小三和弦 根音-小三度-纯五度
        'aug3': [0, 4, 8],  # 增三和弦 根音-大三度-增五度
        'dim3': [0, 3, 6],  # 减三和弦 根音-小三度-减五度

        'M7': [0, 4, 7, 11],  # 大七和弦 根音-大三度-纯五度-大七度
        'Mm7': [0, 4, 7, 10],  # 属七和弦 根音-大三度-纯五度-小七度
        'm7': [0, 3, 7, 10],  # 小七和弦 根音-小三度-纯五度-小七度
        'mM7': [0, 3, 7, 11],  # 小大七和弦 根音-小三度-纯五度-大七度
        'aug7': [0, 4, 8, 10],  # 增七和弦 根音-大三度-增五度-小七度
        'augM7': [0, 4, 8, 11],  # 增大七和弦 根音-大三度-增五度-小七度
        'm7b5': [0, 3, 6, 10],  # 半减七和弦 根音-小三度-减五度-减七度
        'dim7': [0, 3, 6, 9]  # 减减七和弦 根音-小三度-减五度-减七度
    }

    pattern = [0, 0, 0, 0]
    try:
        pattern = chord_dict[name]
    except:
        print(traceback.format_exc())
    return [note + root for note in pattern]


def get_power_chord(root, _type=1):
    if _type == 1:
        pattern = [0, 7]
    elif _type == 2:
        pattern = [0, 7, 12]
    else:
        pattern = [0, 12]

    return [(note + root) for note in pattern]


def get_note_from_arrange(degree):
    scale = [0, 2, 2, 1, 2, 2, 2,
             1, 2, 2, 1, 2, 2, 2]
    note = sum(scale[0:int(degree[0])-1])
    if '-' in degree:
        note -= 1
    if '+' in degree:
        note += 1
    return note


get_mode_dict = lambda: {
    'Heptatonic': {
        'Ionian': [0, 2, 2, 1, 2, 2, 2],  # 全 - 全 - 半 - 全 - 全 - 全 - 半
        'Dorian': [0, 2, 1, 2, 2, 2, 1],  # 全 - 半 - 全 - 全 - 全 - 半 - 全
        'Phrygian': [0, 1, 2, 2, 2, 1, 2],  # 半 - 全 - 全 - 全 - 半 - 全 - 全
        'Lydian': [0, 2, 2, 2, 1, 2, 2],  # 全 - 全 - 全 - 半 - 全 - 全 - 半
        'Mixolydian': [0, 2, 2, 1, 2, 2, 1],  # 全 - 全 - 半 - 全 - 全 - 半 - 全
        'Aeolian': [0, 2, 1, 2, 2, 1, 2],  # 全 - 半 - 全 - 全 - 半 - 全 - 全
        'Locrian': [0, 1, 2, 2, 1, 2, 2]  # 半 - 全 - 全 - 半 - 全 - 全 - 全
    },
    'Pentatonic': {
        'Major': [0, 2, 2, 3, 2],
        'Egyptian': [0, 2, 3, 2, 3],
        'Blues Minor': [0, 3, 2, 3, 2],
        'Blues Major': [0, 2, 3, 2, 2],
        'Minor': [0, 3, 2, 2, 3]
    }
}

get_mode_types = lambda: [type for type in get_mode_dict().keys()]

get_mode_name_list = lambda: [[name for name, mode in get_mode_dict()[type].items()] for type in get_mode_types()]


def get_mode_pattern_list():
    whole_list = []
    for group in [[mode for name, mode in get_mode_dict()[type].items()] for type in get_mode_types()]:
        for pattern in group:
            whole_list.append(pattern)
    return whole_list


get_mode_margin = lambda: [len(mode) for type, mode in get_mode_dict().items()]

get_drum_dict = lambda: {
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

get_instrument_types = lambda: [type for type in get_instrument_dict().keys()]

get_instrument_list = lambda: [[instr for num, instr in get_instrument_dict()[type].items()] for type in
                               get_instrument_types()]

get_instrument_margin = lambda: [len(instruments) for type, instruments in get_instrument_dict().items()]

get_instrument_dict = lambda: {
    'Piano': {
        0: 'Acoustic Grand Piano',
        1: 'Bright Acoustic Piano',
        2: 'Electric Grand Piano',
        3: 'Honky-tonk Piano',
        4: 'Electric Piano 1',
        5: 'Electric Piano 2',
        6: 'Harpsichord',
        7: 'Clavinet'
    },
    'Chromatic Percussion': {
        8: 'Celesta',
        9: 'Glockenspiel',
        10: 'Music Box',
        11: 'Vibraphone',
        12: 'Marimba',
        13: 'Xylophone',
        14: 'Tubular Bells',
        15: 'Dulcimer'
    },
    'Organ': {
        16: 'Drawbar Organ',
        17: 'Percussive Organ',
        18: 'Rock Organ',
        19: 'Church Organ',
        20: 'Reed Organ',
        21: 'Accordion',
        22: 'Harmonica',
        23: 'Tango Accordion'
    },
    'Guitar': {
        24: 'Acoustic Guitar (nylon)',
        25: 'Acoustic Guitar (steel)',
        26: 'Electric Guitar (jazz)',
        27: 'Electric Guitar (clean)',
        28: 'Electric Guitar (muted)',
        29: 'Overdriven Guitar',
        30: 'Distortion Guitar',
        31: 'Guitar harmonics'
    },
    'Bass': {
        32: 'Acoustic Bass',
        33: 'Electric Bass (finger)',
        34: 'Electric Bass (pick)',
        35: 'Fretless Bass',
        36: 'Slap Bass 1',
        37: 'Slap Bass 2',
        38: 'Synth Bass 1',
        39: 'Synth Bass 2'
    },
    'Strings': {
        40: 'Violin',
        41: 'Viola',
        42: 'Cello',
        43: 'Contrabass',
        44: 'Tremolo Strings',
        45: 'Pizzicato Strings',
        46: 'Orchestral Harp',
        47: 'Timpani'
    },
    'Ensemble': {
        48: 'String Ensemble 1',
        49: 'String Ensemble 2',
        50: 'Synth Strings 1',
        51: 'Synth Strings 2',
        52: 'Choir Aahs',
        53: 'Voice Oohs',
        54: 'Synth Voice',
        55: 'Orchestra Hit'
    },
    'Brass': {
        56: 'Trumpet',
        57: 'Trombone',
        58: 'Tuba',
        59: 'Muted Trumpet',
        60: 'French Horn',
        61: 'Brass Section',
        62: 'Synth Brass 1',
        63: 'Synth Brass 2'
    },
    'Reed': {
        64: 'Soprano Sax',
        65: 'Alto Sax',
        66: 'Tenor Sax',
        67: 'Baritone Sax',
        68: 'Oboe',
        69: 'English Horn',
        70: 'Bassoon',
        71: 'Clarinet'
    },
    'Pipe': {
        72: 'Piccolo',
        73: 'Flute',
        74: 'Recorder',
        75: 'Pan Flute',
        76: 'Blown Bottle',
        77: 'Shakuhachi',
        78: 'Whistle',
        79: 'Ocarina'
    },
    'Synth Lead': {
        80: 'Lead 1 (square)',
        81: 'Lead 2 (sawtooth)',
        82: 'Lead 3 (calliope)',
        83: 'Lead 4 (chiff)',
        84: 'Lead 5 (charang)',
        85: 'Lead 6 (voice)',
        86: 'Lead 7 (fifths)',
        87: 'Lead 8 (bass + lead)'
    },
    'Synth Pad': {
        88: 'Pad 1 (new age)',
        89: 'Pad 2 (warm)',
        90: 'Pad 3 (polysynth)',
        91: 'Pad 4 (choir)',
        92: 'Pad 5 (bowed)',
        93: 'Pad 6 (metallic)',
        94: 'Pad 7 (halo)',
        95: 'Pad 8 (sweep)'
    },
    'Synth Effects': {
        96: 'FX 1 (rain)',
        97: 'FX 2 (soundtrack)',
        98: 'FX 3 (crystal)',
        99: 'FX 4 (atmosphere)',
        100: 'FX 5 (brightness)',
        101: 'FX 6 (goblins)',
        102: 'FX 7 (echoes)',
        103: 'FX 8 (sci-fi)'
    },
    'Ethnic': {
        104: 'Sitar',
        105: 'Banjo',
        106: 'Shamisen',
        107: 'Koto',
        108: 'Kalimba',
        109: 'Bag pipe',
        110: 'Fiddle',
        111: 'Shahnai'
    },
    'Percussive': {
        112: 'Tinkle Bell',
        113: 'Agogo',
        114: 'Steel Drums',
        115: 'Woodblock',
        116: 'Taiko Drum',
        117: 'Melodic Tom',
        118: 'Synth Drum'
    },
    'Sound Effects': {
        119: 'Reverse Cymbal',
        120: 'Guitar Fret Noise',
        121: 'Breath Noise',
        122: 'Seashore',
        123: 'Bird Tweet',
        124: 'Telephone Ring',
        125: 'Helicopter',
        126: 'Applause',
        127: 'Gunshot'
    }
}

if __name__ == '__main__':
    # print(get_note_name_by_midi_value(52))
    for num in range(23):
        print(num * 68.18181818181818)
