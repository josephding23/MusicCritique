import traceback
import mido

class UtilityBox():
    def __init__(self):
        pass

    def get_bpm_from_track(self, track):
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
        return 0

    def get_chord_arrangement(self, name):
        chord_dict = {
            'maj3': [0, 4, 7, 0],  # 大三和弦 根音-大三度-纯五度
            'min3': [0, 3, 7, 0],  # 小三和弦 根音-小三度-纯五度
            'aug3': [0, 4, 8, 0],  # 增三和弦 根音-大三度-增五度
            'dim3': [0, 3, 6, 0],  # 减三和弦 根音-小三度-减五度

            'M7': [0, 4, 7, 11],  # 大七和弦 根音-大三度-纯五度-大七度
            'Mm7': [0, 4, 7, 10],  # 属七和弦 根音-大三度-纯五度-小七度
            'm7': [0, 3, 7, 10],  # 小七和弦 根音-小三度-纯五度-小七度
            'mM7': [0, 3, 7, 11],  # 小大七和弦 根音-小三度-纯五度-大七度
            'aug7': [0, 4, 8, 10],  # 增七和弦 根音-大三度-增五度-小七度
            'augM7': [0, 4, 8, 11],  # 增大七和弦 根音-大三度-增五度-小七度
            'm7b5': [0, 3, 6, 10],  # 半减七和弦 根音-小三度-减五度-减七度
            'dim7': [0, 3, 6, 9]  # 减减七和弦 根音-小三度-减五度-减七度
        }

        chord = [0, 0, 0, 0]
        try:
            chord = chord_dict[name]
        except:
            print(traceback.format_exc())
        return chord

    def get_drum_dict(self):
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






