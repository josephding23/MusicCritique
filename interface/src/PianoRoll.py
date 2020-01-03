from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame.midi
from fractions import Fraction
import time

from midi_extended.UtilityBox import *
from interface.src.MusicSegment import MusicSegment, SegmentWindow
from interface.src.Key import Key

class PianoRoll(QWidget):

    def __init__(self, volume, octave, instr, mode, root_note, mode_display):
        super().__init__()
        self.setObjectName('PianoRoll')
        self.octave = octave
        self.instr = instr
        self.volume = volume
        self.mode = mode
        self.mode_pattern = get_mode_dict()[self.mode[0]][self.mode[1]]
        self.mode_distance = [sum(self.mode_pattern[:index+1]) for index in range(len(self.mode_pattern))]
        self.root_note = root_note
        self.mode_display = mode_display

        self.stable_notes = [self.root_note + distance for distance in [self.mode_distance[0], self.mode_distance[2], self.mode_distance[4],
                                    12 + self.mode_distance[0], 12 +self.mode_distance[2], 12 + self.mode_distance[4], 24]]
        self.unstable_notes = [self.root_note + distance for distance in [self.mode_distance[1], self.mode_distance[3], self.mode_distance[5], self.mode_distance[6],
                                      12 + self.mode_distance[1], 12 + self.mode_distance[3], 12 + self.mode_distance[5], 12 + self.mode_distance[6]]]

        self.is_recording = False
        self.music_segment = None

        pygame.midi.init()
        self.player = pygame.midi.Output(0)
        self.player.set_instrument(self.instr)

        self.white_notes_margin = [0, 2, 2, 1, 2, 2, 2]
        self.black_notes_margin = [1, 2, 3, 2, 2]

        self.white_notes_distance = [[0, 2, 4, 5, 7, 9, 11], [12, 14, 16, 17, 19, 21, 23, 24]]
        self.black_notes_distance = [[1, 3, 6, 8, 10], [13, 15, 18, 20, 22]]

        self.white_shortcuts = [
            [Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V, Qt.Key_B, Qt.Key_N, Qt.Key_M],
            [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T, Qt.Key_Y, Qt.Key_U, Qt.Key_I]
        ]
        self.black_shortcuts = [
            [Qt.Key_S, Qt.Key_D, Qt.Key_G, Qt.Key_H, Qt.Key_J],
            [Qt.Key_2, Qt.Key_3, Qt.Key_5, Qt.Key_6, Qt.Key_7]
        ]

        self.blackKeysList = [[], []]
        self.whiteKeysList = [[], []]

        self.pressed_time = 0
        self.release_time = 0

        self.start_note = 60 + (self.octave - 4) * 12
        self.initUI()

    def start_recording(self, music_segment):
        self.is_recording = True
        self.music_segment = music_segment
        for group in range(2):
            for black_key in self.blackKeysList[group]:
                black_key.start_recording(music_segment)
            for white_key in self.whiteKeysList[group]:
                white_key.start_recording(music_segment)

    def delete_player(self):
        try:
            self.player.close()
            pygame.midi.quit()
        except:
            pass

    def change_volume(self, volume):
        self.volume = volume

    def change_instrument(self, instr):
        self.instr = instr
        self.player.set_instrument(self.instr)

    def initUI(self):
        self.blackKeysBoxes = [QHBoxLayout(), QHBoxLayout()]
        self.whiteKeysBoxes = [QHBoxLayout(), QHBoxLayout()]
        self.octaveRollBoxes = [QVBoxLayout(), QVBoxLayout()]

        for group in range(2):
            for index in range(len(self.black_notes_distance[group])):
                note = self.start_note + self.black_notes_distance[group][index]
                if self.mode_display == True:
                    if note in self.stable_notes:
                        new_key = Key(self.player, note, self.volume, 'r')
                        self.blackKeysList[group].append(new_key)
                    elif note in self.unstable_notes:
                        new_key = Key(self.player, note, self.volume, 'g')
                        self.blackKeysList[group].append(new_key)
                    else:
                        new_key = Key(self.player, note, self.volume, 'b')
                        self.blackKeysList[group].append(new_key)
                else:
                    new_key = Key(self.player, note, self.volume, 'b')
                    self.blackKeysList[group].append(new_key)

            for index in range(len(self.white_notes_distance[group])):
                note = self.start_note + self.white_notes_distance[group][index]
                if self.mode_display == True:
                    if note in self.stable_notes:
                        new_key = Key(self.player, note, self.volume, 'r')
                        self.whiteKeysList[group].append(new_key)
                    elif note in self.unstable_notes:
                        new_key = Key(self.player, note, self.volume, 'g')
                        self.whiteKeysList[group].append(new_key)
                    else:
                        new_key = Key(self.player, note, self.volume, 'w')
                        self.whiteKeysList[group].append(new_key)
                else:
                    new_key = Key(self.player, note, self.volume, 'w')
                    self.whiteKeysList[group].append(new_key)

            for btn in self.blackKeysList[group]:
                self.blackKeysBoxes[group].addWidget(btn)
            self.blackKeysBoxes[group].setStretch(0, 2)
            self.blackKeysBoxes[group].setStretch(1, 4)
            self.blackKeysBoxes[group].setStretch(2, 2)
            self.blackKeysBoxes[group].setStretch(3, 2)
            self.blackKeysBoxes[group].setStretch(4, 2)

            for btn in self.whiteKeysList[group]:
                self.whiteKeysBoxes[group].addWidget(btn)
            self.whiteKeysBoxes[group].setSpacing(20)
            self.octaveRollBoxes[group].addLayout(self.blackKeysBoxes[group])
            self.octaveRollBoxes[group].addLayout(self.whiteKeysBoxes[group])
            self.octaveRollBoxes[group].setSpacing(2)

        self.whiteKeysBoxes[0].setContentsMargins(0, 0, 0, 0)
        self.whiteKeysBoxes[1].setContentsMargins(0, 0, 0, 0)
        self.whiteKeysBoxes[0].addSpacing(25)
        self.whiteKeysBoxes[1].addSpacing(25)
        self.blackKeysBoxes[0].setContentsMargins(50, 0, 55, 0)
        self.blackKeysBoxes[1].setContentsMargins(50, 0, 155, 0)

        self.wholeLayout = QHBoxLayout()
        self.wholeLayout.addLayout(self.octaveRollBoxes[0])
        self.wholeLayout.addLayout(self.octaveRollBoxes[1])
        self.wholeLayout.setStretch(0, 7)
        self.wholeLayout.setStretch(1, 8)
        self.wholeLayout.setSpacing(1)
        self.wholeLayout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.wholeLayout)
        self.setFixedSize(QSize(1500, 550))

    def keyPressEvent(self, e):
        for group in range(2):
            for index in range(len(self.white_shortcuts[group])):
                if e.key() == self.white_shortcuts[group][index]:
                    self.player.note_on(self.whiteKeysList[group][index].note, self.volume)
                    self.whiteKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
                    )

                    return
            for index in range(len(self.black_shortcuts[group])):
                if e.key() == self.black_shortcuts[group][index]:
                    self.player.note_on(self.blackKeysList[group][index].note, self.volume)
                    self.blackKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: #363636; color: white; border: 4px inset #9C9C9C;}'
                    )
                    return

    def keyReleaseEvent(self, e):
        for group in range(2):
            for index in range(len(self.white_shortcuts[group])):
                if e.key() == self.white_shortcuts[group][index]:
                    self.player.note_off(self.whiteKeysList[group][index].note, self.volume)
                    self.whiteKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: white; color: black; border: 4px outset #828282;}'
                        'QPushButton:hover{background-color: #B5B5B5; color: black; border: 4px outset #828282;}'
                        'QPushButton:pressed{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
                    )
                    if self.is_recording:
                        self.music_segment.add_note(self.whiteKeysList[group][index].note, self.music_segment.length_per_note)
                        if self.music_segment.window_on == True:
                            self.music_segment.replot()
                            '''
                            self.segment_window = SegmentWindow(self.music_segment)
                            self.segment_window.show()
                            self.segment_window.move(1200, 30)
                            '''
                    return
            for index in range(len(self.black_shortcuts[group])):
                if e.key() == self.black_shortcuts[group][index]:
                    self.player.note_off(self.blackKeysList[group][index].note, self.volume)
                    self.blackKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: black; color: white; border: 4px outset #9C9C9C;}'
                        'QPushButton:hover{background-color: #4F4F4F; color: white; border: 4px outset #9C9C9C;}'
                        'QPushButton:pressed{background-color: #363636; color: white; border: 4px inset #9C9C9C;}'
                    )
                    if self.is_recording:
                        self.music_segment.add_note(self.blackKeysList[group][index].note, self.music_segment.length_per_note)
                        if self.music_segment.window_on == True:
                            self.music_segment.replot()
                            # self.segment_window.show()
                            # self.segment_window.move(1200, 30)
                    return