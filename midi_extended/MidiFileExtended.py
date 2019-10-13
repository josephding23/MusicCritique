from mido import Message, MidiFile, MidiTrack, MetaMessage
from midi_extended.Track import TrackExtended
from midi_extended.UtilityBox import UtilityBox
import traceback
import pypianoroll
import matplotlib.pyplot as plt
import numpy as np
from pypianoroll import Multitrack, Track
import os
import contextlib
with contextlib.redirect_stdout(None):
    import pygame

class MidiFileExtended(MidiFile):
    def __init__(self, path, mode='w', type=1, charset='utf-8'):
        self.path = path
        self.extended_tracks = []
        self.utility_box = UtilityBox()
        if mode == 'r':
            MidiFile.__init__(self, path, type=type, charset=charset)
        elif mode == 'w':
            MidiFile.__init__(self, type=type, charset=charset)

    def add_new_track(self, name, time_signature, bpm, key, instruments):
        new_track = TrackExtended(name, time_signature, bpm, key, instruments)
        super().add_track(name)
        self.tracks.append(new_track)
        self.extended_tracks.append(new_track)

    def get_extended_track(self, name):
        for track in self.extended_tracks:
            if type(track) == TrackExtended and type(track) != MidiTrack and track.name == name:
                return track

    def turn_track_into_numpy_matrix(self, track_name, path):
        track = self.get_track_by_name(track_name)
        time_per_unit = 60 * 60 * 10 / self.utility_box.get_bpm_from_track(track) / 4
        note_time_units = []
        length_units = []
        for msg in track:
            if msg.type == 'note_off':
                time = msg.time
                note = msg.note
                note_time_units.append((int(time/time_per_unit), note))
                length_units.append(time)
        piano_roll = np.zeros((sum(length_units), 128))
        times = [0]
        for i in range(len(note_time_units)-1):
            time_point = 0
            for j in range(i+1):
                time_point = time_point + note_time_units[j][0]
            times.append(time_point-1)

        notes = [nt[1] for nt in note_time_units]

        for i in range(len(times)-1):
            start = times[i]
            end = times[i+1]
            note = notes[i]
            for time in range(start, end):
                piano_roll[time][note] = 1
        # plt.scatter(times, notes)
        # plt.show()
        np.save(path, piano_roll)
        return piano_roll

    def generate_track_from_numpy_matrix(self, path, size, name, is_drum, save_fig=False, save_path=None, program=0):
        piano_roll = np.load(path)
        piano_roll.resize(size)
        track = Track(piano_roll, program, is_drum,  name)
        fig, ax = pypianoroll.track.plot_track(track)
        if save_fig:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_multiple_tracks_from_numpy_matrices(self, num, dir, files, size, names, are_drums,
                                                     tempo, downbeat, beat_resolution, programs=None, save_fig=False, save_path=None):
        tracks = []
        for i in range(num):
            path = dir + files[i]
            name = names[i]
            if programs == None:
                program = 0
            else:
                program = programs[i]
            is_drum = are_drums[i]
            piano_roll = np.load(path)
            piano_roll.resize(size)
            track = Track(piano_roll, program, is_drum, name)
            tracks.append(track)

        multitrack = Multitrack(tracks=tracks, tempo=tempo, downbeat=downbeat, beat_resolution=beat_resolution)
        multitrack.save(dir + 'multi.npz')

        fig, axs = pypianoroll.multitrack.plot_multitrack(multitrack, grid_linewidth=0.8, ytick='off')
        if save_fig:
            plt.savefig(save_path)
        else:
            plt.show()


    def get_track_by_name(self, name):
        tracks = []
        for track in self.tracks:
            if type(track) == MidiTrack and track.name == name:
                tracks.append(track)
        if len(tracks) == 0:
            return None

        max_length = 0
        return_track = tracks[0]
        for track in tracks:
            length = 0
            for msg in track:
                length = length + 1
            if max_length < length:
                max_length = length
                return_track = track
        return return_track

    def print_tracks_info(self):
        for track in self.tracks:
            print(track)

    def save_midi(self):
        super().save(self.path)

    def play_it(self):
        freq = 44100
        bitsize = -16
        channels = 2
        buffer = 1024
        pygame.mixer.init(freq, bitsize, channels, buffer)
        pygame.mixer.music.set_volume(1)
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(self.path)
        except:
            import traceback
            print(traceback.format_exc())
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30)