from mido import Message, MidiFile, MidiTrack, MetaMessage
from fractions import Fraction
import mido
import traceback
from midi_extended.UtilityBox import *

class TrackExtended(MidiTrack):
    def __init__(self, name='default', time_signature=None, bpm=None, key=None, instruments=None):
        MidiTrack.__init__(self)
        self.name = name
        self.time_signature = time_signature
        self.bpm = bpm
        self.key = key
        self.instruments = instruments

        if self.isInitiated():
            self.add_meta_info()

    def initiate_with_track(self, track):
        self.name = track.name
        self.bpm = get_bpm_from_track(track)
        self.key = get_key_from_track(track)
        self.time_signature = get_time_signature_from_track(track)
        self.instruments = get_instruments_from_track(track)

        return self.isInitiated()

    def __str__(self):
        return "Track: " + self.name + \
               " time signature: " + Fraction(self.time_signature).__str__() + \
               " initiated bpm: " + str(self.bpm) + \
               " key: " + self.key

    def isInitiated(self):
        return self.name != 'default' and self.time_signature != None and self.bpm != None and self.key != None and self.instruments != None

    def print_msgs(self):
        for msg in super():
            print(msg)

    def get_name(self):
        return self.name

    def get_time(self):
        return self.time_signature

    def get_bpm(self):
        return self.bpm

    def get_key(self):
        return self.key

    def get_instruments(self):
        return self.instruments

    def set_bpm(self, bpm):
        self.bpm = bpm
        tempo = mido.bpm2tempo(self.bpm)
        super().append(MetaMessage('set_tempo', tempo=tempo, time=0))

    def add_meta_info(self):
        tempo = mido.bpm2tempo(self.bpm)
        numerator = Fraction(self.time_signature).numerator
        denominator = Fraction(self.time_signature).denominator
        super().append(MetaMessage('time_signature', numerator=numerator, denominator=denominator))
        super().append(MetaMessage('set_tempo', tempo=tempo, time=0))
        super().append(MetaMessage('key_signature', key=self.key))
        for channel, program in self.instruments.items():
            super().append(Message('program_change', channel=int(channel), program=program, time=0))

    def add_chord(self, root, name, format, length, root_base=0, channel=3):
        bpm = self.bpm
        major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
        notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        notes_dict = {}
        for i, note in enumerate(notes):
            notes_dict[note] = 60 + sum(major_notes[0:i + 1])
        root_note = notes_dict[root] + root_base * 12
        chord = get_chord_arrangement(name)
        meta_time = 60 * 60 * 10 / bpm
        time = round(length / len(format) * meta_time)

        for dis in format:
            note = root_note + chord[dis]
            super().append(Message('note_on', note=note, velocity=56, time=0, channel=channel))
            super().append(Message('note_off', note=note, velocity=56, time=time, channel=channel))

    def add_bass(self, note, length, base_num=-1, velocity=0.7, channel=6, delay=0):
        bpm = self.bpm
        meta_time = 60 * 60 * 10 / bpm
        major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
        base_note = 60
        super().append(
            Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]), velocity=round(64 * velocity),
                    time=round(delay * meta_time), channel=channel))
        super().append(
            Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                    velocity=round(64 * velocity),
                    time=round(meta_time * length), channel=channel))

    def add_note(self, note, length, base_num=0, delay=0, velocity=1.0, channel=0, pitch_type=0, tremble_setting=None,
                 bend_setting=None):
        bpm = self.bpm
        meta_time = 60 * 60 * 10 / bpm
        major_notes = [0, 2, 2, 1, 2, 2, 2, 1]
        base_note = 60
        if pitch_type == 0:  # No Pitch Wheel Message
            super().append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(delay * meta_time), channel=channel))
            super().append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                 velocity=round(64 * velocity), time=round(meta_time * length), channel=channel))
        elif pitch_type == 1:  # Tremble
            try:
                pitch = tremble_setting['pitch']
                wheel_times = tremble_setting['wheel_times']
                super().append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                     velocity=round(64 * velocity),
                                     time=round(delay * meta_time), channel=channel))
                for i in range(wheel_times):
                    super().append(Message('pitchwheel', pitch=pitch, time=round(meta_time * length / (2 * wheel_times)),
                                         channel=channel))
                    super().append(Message('pitchwheel', pitch=0, time=0, channel=channel))
                    super().append(Message('pitchwheel', pitch=-pitch, time=round(meta_time * length / (2 * wheel_times)),
                                         channel=channel))
                super().append(Message('pitchwheel', pitch=0, time=0, channel=channel))
                super().append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                     velocity=round(64 * velocity), time=0, channel=channel))
            except:
                print(traceback.format_exc())
        elif pitch_type == 2:  # Bend
            try:
                pitch = bend_setting['pitch']
                PASDA = bend_setting['PASDA']  # Prepare-Attack-Sustain-Decay-Aftermath (Taken the notion of ADSR)
                prepare_rate = PASDA[0] / sum(PASDA)
                attack_rate = PASDA[1] / sum(PASDA)
                sustain_rate = PASDA[2] / sum(PASDA)
                decay_rate = PASDA[3] / sum(PASDA)
                aftermath_rate = PASDA[4] / sum(PASDA)
                super().append(Message('note_on', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                     velocity=round(64 * velocity), time=round(delay * meta_time), channel=channel))
                super().append(Message('aftertouch', time=round(meta_time * length * prepare_rate), channel=channel))
                super().append(
                    Message('pitchwheel', pitch=pitch, time=round(meta_time * length * attack_rate), channel=channel))
                super().append(Message('aftertouch', time=round(meta_time * length * sustain_rate), channel=channel))
                super().append(
                    Message('pitchwheel', pitch=0, time=round(meta_time * length * decay_rate), channel=channel))
                super().append(Message('note_off', note=base_note + base_num * 12 + sum(major_notes[0:note]),
                                     velocity=round(64 * velocity), time=round(meta_time * length * aftermath_rate),
                                     channel=channel))
            except:
                print(traceback.format_exc())

    def wait(self, time):
        bpm = self.bpm
        meta_time = 60 * 60 * 10 / bpm
        super().append(Message('note_off', time=round(meta_time * time)))

    def add_drum(self, name, time, delay=0, velocity=1):
        bpm = self.bpm
        meta_time = 60 * 60 * 10 / bpm
        drum_dict = get_drum_dict()
        try:
            note = drum_dict[name]
        except:
            print(traceback.format_exc())
            return
        super().append(Message('note_on', note=note, velocity=round(64 * velocity), time=delay, channel=9))
        super().append(
            Message('note_off', note=note, velocity=round(64 * velocity), time=round(meta_time * time), channel=9))