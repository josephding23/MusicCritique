from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame.midi
from fractions import Fraction
import time

from midi_extended.UtilityBox import *
from interface.src.MusicSegment import MusicSegment, SegmentWindow

class Key(QWidget):

    def __init__(self, player, note, volume=63, color='w'):
        super().__init__()
        self.player = player
        self.note = note
        self.volume = volume
        self.note_name = get_note_name_by_midi_value(self.note)
        # print(self.note_name)
        self.color = color

        self.is_recording = False
        self.music_segment = None

        self.initUI()

    def change_volume(self, volume):
        self.volume = volume

    def set_note_signal_value(self):
        print('note ' + self.note)

    def initUI(self):
        self.keyBtn = QPushButton(self.note_name)
        self.note_font = QFont()
        self.note_font.setFamily('Times New Roman')
        self.note_font.setWeight(QFont.DemiBold)
        self.note_font.setPixelSize(14)
        self.keyBtn.setFont(self.note_font)
        self.keyBtn.setFixedHeight(250)
        self.keyBtn.setFixedWidth(50)
        self.keyBtn.setCursor(Qt.PointingHandCursor)
        self.keyBtn.pressed.connect(self.pressedKeyResponse)
        self.keyBtn.released.connect(self.releasedKeyResponse)
        self.keyLayout = QVBoxLayout()
        self.keyLayout.addWidget(self.keyBtn)
        self.setLayout(self.keyLayout)

        if self.color == 'w':
            self.setStyleSheet(
                'QPushButton{background-color: white; color: black; border: 4px outset #828282;}'
                'QPushButton:hover{background-color: #B5B5B5; color: black; border: 4px outset #828282;}'
                'QPushButton:pressed{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
            )
        elif self.color == 'b':
            self.setStyleSheet(
                'QPushButton{background-color: black; color: white; border: 4px outset #828282;}'
                'QPushButton:hover{background-color: #4F4F4F; color: white; border: 4px outset #828282;}'
                'QPushButton:pressed{background-color: #363636; color: white; border: 4px inset #828282;}'
            )
        elif self.color == 'r':
            self.setStyleSheet(
                'QPushButton{background-color: #FF4040; color: #FFE4E1; border: 4px outset #9C9C9C;}'
                'QPushButton:hover{background-color: #FA8072; color: #8B2323; border: 4px outset #9C9C9C;}'
                'QPushButton:pressed{background-color: #CD2626; color: #FFE4E1; border: 4px inset #9C9C9C;}'
            )
        elif self.color == 'g':
            self.setStyleSheet(
                'QPushButton{background-color: #008B45; color: #C0FF3E; border: 4px outset #9C9C9C;}'
                'QPushButton:hover{background-color: #9AFF9A; color: #6B8E23; border: 4px outset #9C9C9C;}'
                'QPushButton:pressed{background-color: #6E8B3D; color: #8FBC8F; border: 4px inset #9C9C9C;}'
            )
        # self.resize(3, 20)

    def pressedKeyResponse(self):
        # self.note_sig.emit()
        self.player.note_on(self.note, self.volume)

    def start_recording(self, music_segment):
        self.is_recording = True
        self.music_segment = music_segment

    def releasedKeyResponse(self):
        self.player.note_off(self.note, self.volume)

        if self.is_recording:
            self.music_segment.add_note(self.note, self.music_segment.length_per_note)
            if self.music_segment.window_on == True:
                self.music_segment.replot()
                '''
                self.segment_window = SegmentWindow(self.music_segment)
                self.segment_window.show()
                self.segment_window.move(1200, 30)
                '''



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


class Piano(QWidget):

    def __init__(self):
        super().__init__()
        self.volume = 63
        self.octave = 3
        self.control = 0

        self.octave_changed_time = 0

        self.control_lbl_style = 'QLable{text-align: center; color: #8B795E; font-style: Century Gothic; font-size: 16;}'
        self.option_lbl_font = QFont()
        self.option_lbl_font.setFamily('Century Gothic')
        self.option_lbl_font.setWeight(QFont.Bold)
        self.option_lbl_font.setStyle(QFont.StyleItalic)
        self.option_lbl_font.setPixelSize(16)

        self.combo_style = 'QComboBox{background-color: #FFF5EE; color: #8B5A2B; text-align: center;}'
        self.spin_style = 'QSpinBox{background-color: #FFF5EE; color: #8B5A2B;}'
        self.input_font = QFont()
        self.input_font.setFamily('Century Gothic')
        self.input_font.setWeight(QFont.DemiBold)
        self.input_font.setStyle(QFont.StyleNormal)
        self.input_font.setPixelSize(14)

        self.instr_type_list = get_instrument_types()
        self.instr_list = get_instrument_list()
        self.instr_type_index = 0
        self.instr_index = 0

        self.is_record_mode = False
        self.music_segment = MusicSegment
        self.record_btn_icon_size = QSize(45, 45)
        self.record_btn_size = QSize(60, 60)
        self.record_btn_style = \
            'QPushButton{background: #FFEFDB; border: 3px outset #8B8378; border-radius: 15px;}' \
            'QPushButton:hover{background: #CDC9C9; border: 3px outset #8B8378; border-radius: 15px;}' \
            'QPushButton:pressed{background: #8B8682; border: 3px inset #8B8378; border-radius: 15px;}'
        self.show_window_on = False

        self.metre = '1/4'
        self.metre_numerator = 1
        self.metre_denominator = 4
        self.metre_options = ['1/4', '2/4', '3/4', '4/4', '3/8', '6/8']
        self.metre_effects = {
            '1/4': (1, 4),
            '2/4': (2, 4),
            '3/4': (3, 4),
            '4/4': (4, 4),
            '3/8': (3, 8),
            '6/8': (6, 8)
        }

        self.length_per_note = 1/2
        self.length_per_note_options = ['1/16', '1/8', '1/4', '3/4', '1/2', '1', '3/2', '2', '3', '4']
        self.length_per_note_effects = [1/16, 1/8, 1/4, 3/4, 1/2, 1, 3/2, 2, 3, 4]
        self.length_per_note_index = 4

        self.beats_per_minute = 90

        self.root_notes_list = [60 + (self.octave - 4) * 12 + distance for distance in range(12)]
        self.root_note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        self.root_note_index = 0
        self.root_note = self.root_notes_list[self.root_note_index]

        self.mode_type_list = get_mode_types()
        self.mode_list = get_mode_name_list()
        self.mode_pattern_list = get_mode_pattern_list()
        self.mode_type = 'Heptatonic'
        self.mode_name = 'Ionian'
        self.mode_type_index = 0

        self.mode_display = False

        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)

        self.initUI()

    def initUI(self):
        self.record_start_btn = QPushButton()
        self.record_start_btn.setIcon(QIcon('./icon/clipboard_start.png'))
        self.record_start_btn.setToolTip('Start Recording')
        self.record_start_btn.setIconSize(self.record_btn_icon_size)
        self.record_start_btn.setFixedSize(self.record_btn_size)
        self.record_start_btn.setStyleSheet(self.record_btn_style)
        self.record_start_btn.setCursor(Qt.PointingHandCursor)
        self.record_start_btn.clicked.connect(self.recordStart)

        self.record_draw_btn = QPushButton()
        self.record_draw_btn.setIcon(QIcon('./icon/clipboard_see.png'))
        self.record_draw_btn.setToolTip('Draw Notes Plot')
        self.record_draw_btn.setIconSize(self.record_btn_icon_size)
        self.record_draw_btn.setFixedSize(self.record_btn_size)
        self.record_draw_btn.setStyleSheet(self.record_btn_style)
        self.record_draw_btn.setCursor(Qt.PointingHandCursor)
        self.record_draw_btn.clicked.connect(self.recordDraw)

        self.record_play_btn = QPushButton()
        self.record_play_btn.setIcon(QIcon('./icon/clipboard_play.png'))
        self.record_play_btn.setToolTip('Play Recorded Segment')
        self.record_play_btn.setIconSize(self.record_btn_icon_size)
        self.record_play_btn.setFixedSize(self.record_btn_size)
        self.record_play_btn.setStyleSheet(self.record_btn_style)
        self.record_play_btn.setCursor(Qt.PointingHandCursor)
        self.record_play_btn.clicked.connect(self.recordPlay)

        self.record_return_btn = QPushButton()
        self.record_return_btn.setIcon(QIcon('./icon/clipboard_return.png'))
        self.record_return_btn.setToolTip('Delete Last Note')
        self.record_return_btn.setIconSize(self.record_btn_icon_size)
        self.record_return_btn.setFixedSize(self.record_btn_size)
        self.record_return_btn.setStyleSheet(self.record_btn_style)
        self.record_return_btn.setCursor(Qt.PointingHandCursor)
        self.record_return_btn.clicked.connect(self.recordReturn)

        self.record_finish_btn = QPushButton()
        self.record_finish_btn.setIcon(QIcon('./icon/clipboard_finish.png'))
        self.record_finish_btn.setToolTip('Finish Recording')
        self.record_finish_btn.setIconSize(self.record_btn_icon_size)
        self.record_finish_btn.setFixedSize(self.record_btn_size)
        self.record_finish_btn.setStyleSheet(self.record_btn_style)
        self.record_finish_btn.setCursor(Qt.PointingHandCursor)
        self.record_finish_btn.clicked.connect(self.recordFinish)

        self.record_stop_btn = QPushButton()
        self.record_stop_btn.setIcon(QIcon('./icon/clipboard_stop.png'))
        self.record_stop_btn.setToolTip('Stop Recording')
        self.record_stop_btn.setIconSize(self.record_btn_icon_size)
        self.record_stop_btn.setFixedSize(self.record_btn_size)
        self.record_stop_btn.setStyleSheet(self.record_btn_style)
        self.record_stop_btn.setCursor(Qt.PointingHandCursor)
        self.record_stop_btn.clicked.connect(self.recordStop)

        self.record_btn_box = QHBoxLayout()
        self.record_btn_box.addWidget(self.record_start_btn)
        self.record_btn_box.addWidget(self.record_draw_btn)
        self.record_btn_box.addWidget(self.record_play_btn)
        self.record_btn_box.addWidget(self.record_return_btn)
        self.record_btn_box.addWidget(self.record_finish_btn)
        self.record_btn_box.addWidget(self.record_stop_btn)
        self.record_btn_box.setContentsMargins(5, 5, 5, 10)

        self.LPN_lbl = QLabel('LPN:')
        self.LPN_lbl.setAlignment(Qt.AlignCenter)
        self.LPN_lbl.setFont(self.option_lbl_font)
        self.LPN_ctrl = QComboBox()
        self.LPN_ctrl.addItems(self.length_per_note_options)
        self.LPN_ctrl.setCurrentIndex(self.length_per_note_index)
        self.LPN_ctrl.setFont(self.input_font)
        self.LPN_ctrl.setStyleSheet(self.combo_style)
        self.LPN_ctrl.currentIndexChanged.connect(self.LPNChanged)
        self.LPN_box = QHBoxLayout()
        self.LPN_box.addWidget(self.LPN_lbl)
        self.LPN_box.addWidget(self.LPN_ctrl)

        self.metre_lbl = QLabel('Metre:')
        self.metre_lbl.setAlignment(Qt.AlignCenter)
        self.metre_lbl.setFont(self.option_lbl_font)
        self.metre_ctrl = QComboBox()
        self.metre_ctrl.addItems(self.metre_options)
        self.metre_ctrl.setFont(self.input_font)
        self.metre_ctrl.setStyleSheet(self.combo_style)
        self.metre_ctrl.currentIndexChanged.connect(self.metreChanged)
        self.metre_box = QHBoxLayout()
        self.metre_box.addWidget(self.metre_lbl)
        self.metre_box.addWidget(self.metre_ctrl)

        self.BPM_lbl = QLabel('BPM:')
        self.BPM_lbl.setAlignment(Qt.AlignCenter)
        self.BPM_lbl.setFont(self.option_lbl_font)
        self.BPM_ctrl = QSpinBox()
        self.BPM_ctrl.setRange(40, 208)
        self.BPM_ctrl.setValue(self.beats_per_minute)
        self.BPM_ctrl.setAlignment(Qt.AlignCenter)
        self.BPM_ctrl.setStyleSheet(self.spin_style)
        self.BPM_ctrl.setFont(self.input_font)
        self.BPM_ctrl.setFocusPolicy(Qt.NoFocus)
        self.BPM_box = QHBoxLayout()
        self.BPM_box.addWidget(self.BPM_lbl)
        self.BPM_box.addWidget(self.BPM_ctrl)

        self.root_note_lbl = QLabel('Root Note:')
        self.root_note_lbl.setAlignment(Qt.AlignCenter)
        self.root_note_lbl.setFont(self.option_lbl_font)
        self.root_note_ctrl = QComboBox()
        self.root_note_ctrl.addItems(self.root_note_names)
        self.root_note_ctrl.setFont(self.input_font)
        self.root_note_ctrl.setStyleSheet(self.combo_style)
        self.root_note_ctrl.currentIndexChanged.connect(self.rootNoteChanged)
        self.root_note_box = QHBoxLayout()
        self.root_note_box.addWidget(self.root_note_lbl)
        self.root_note_box.addWidget(self.root_note_ctrl)

        self.mode_box = QHBoxLayout()
        self.mode_lbl = QLabel('Mode:')
        self.mode_lbl.setAlignment(Qt.AlignCenter)
        self.mode_lbl.setFont(self.option_lbl_font)
        self.mode_type_combo = QComboBox()
        self.mode_type_combo.setStyleSheet(self.combo_style)
        self.mode_type_combo.setFont(self.input_font)
        self.mode_type_combo.addItems(self.mode_type_list)
        self.mode_type_combo.setFocusPolicy(Qt.NoFocus)
        self.mode_type_combo.currentIndexChanged.connect(self.modeTypeChanged)
        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet(self.combo_style)
        self.mode_combo.setFont(self.input_font)
        self.mode_combo.addItems(self.mode_list[self.mode_type_index])
        self.mode_combo.setFocusPolicy(Qt.NoFocus)
        self.mode_combo.currentIndexChanged.connect(self.modeChanged)
        self.mode_box.addWidget(self.mode_lbl)
        self.mode_box.addWidget(self.mode_type_combo)
        self.mode_box.addWidget(self.mode_combo)

        self.record_option_box = QVBoxLayout()
        self.record_option_box_first = QHBoxLayout()
        self.record_option_box_second = QHBoxLayout()
        self.record_option_box_first.addLayout(self.LPN_box)
        self.record_option_box_first.addLayout(self.metre_box)
        self.record_option_box_first.addLayout(self.BPM_box)
        self.record_option_box_second.addLayout(self.root_note_box)
        self.record_option_box_second.addLayout(self.mode_box)
        self.record_option_box.addLayout(self.record_option_box_first)
        self.record_option_box.addLayout(self.record_option_box_second)
        self.record_option_box.setSpacing(20)
        # self.record_option_box.setSpacing(10)

        self.record_box = QVBoxLayout()
        self.record_box.addLayout(self.record_btn_box)
        self.record_box.addLayout(self.record_option_box)
        self.record_box.setStretch(0, 2)
        self.record_box.setStretch(1, 3)

        self.volume_box = QHBoxLayout()
        self.volume_lbl = QLabel('Volume:')
        self.volume_lbl.setAlignment(Qt.AlignCenter)
        self.volume_lbl.setFont(self.option_lbl_font)
        self.volume_ctrl = QSlider(Qt.Horizontal)
        self.volume_ctrl.setTickPosition(QSlider.TicksBothSides)
        self.volume_ctrl.setSingleStep(8)
        self.volume_ctrl.setTickInterval(16)
        self.volume_ctrl.setFixedWidth(300)
        self.volume_ctrl.setFixedHeight(15)
        self.volume_ctrl.setRange(0, 127)
        self.volume_ctrl.setStyleSheet(
            'QSlider:groove{ border-radius: 6px;}'
            'QSlider:handle{width: 15px; background-color: #8B7765; border-radius: 6px; border: 1px solid #8B4513;}'
            'QSlider:add-page{background-color: #EECFA1; border-radius: 6px;}'
            'QSlider:sub-page{background-color: #FFA07A; border-radius: 6px;}'
        )

        self.volume_ctrl.setValue(self.volume)
        self.volume_ctrl.setFocusPolicy(Qt.NoFocus)
        self.volume_ctrl.sliderReleased.connect(self.volumeChanged)
        self.mode_display_check = QCheckBox('Mode Display')
        self.mode_display_check.setFont(self.option_lbl_font)
        # self.mode_display_check.setStyleSheet()
        self.mode_display_check.stateChanged.connect(self.modeDisplayChanged)
        self.volume_box.addWidget(self.volume_lbl)
        self.volume_box.addWidget(self.volume_ctrl)
        self.volume_box.addWidget(self.mode_display_check)
        self.volume_box.setAlignment(Qt.AlignLeft)
        self.volume_box.setStretch(0, 1)
        self.volume_box.setStretch(1, 4)
        self.volume_box.setStretch(2, 1)

        self.octave_box = QHBoxLayout()
        self.octave_lbl = QLabel('Octave:')
        self.octave_lbl.setAlignment(Qt.AlignCenter)
        self.octave_lbl.setFont(self.option_lbl_font)
        self.octave_ctrl = QSpinBox()
        self.octave_ctrl.setStyleSheet(self.spin_style)
        self.octave_ctrl.setAlignment(Qt.AlignCenter)
        self.octave_ctrl.setFont(self.input_font)
        self.octave_ctrl.setRange(0, 8)
        self.octave_ctrl.setSingleStep(1)
        self.octave_ctrl.setValue(self.octave)
        self.octave_ctrl.setWrapping(True)
        self.octave_ctrl.setFocusPolicy(Qt.NoFocus)
        self.octave_ctrl.valueChanged.connect(self.octaveChanged)
        self.octave_box.addWidget(self.octave_lbl)
        self.octave_box.addWidget(self.octave_ctrl)
        octave_up = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        octave_up.activated.connect(self.octaveIncrease)
        octave_down = QShortcut(QKeySequence(Qt.Key_PageDown), self)
        octave_down.activated.connect(self.octaveDecrease)

        self.instr_box = QHBoxLayout()
        self.instr_lbl = QLabel('Instrument:')
        self.instr_lbl.setAlignment(Qt.AlignCenter)
        self.instr_lbl.setFont(self.option_lbl_font)
        self.instr_type_combo = QComboBox()
        self.instr_type_combo.setStyleSheet(self.combo_style)
        self.instr_type_combo.setFont(self.input_font)
        self.instr_type_combo.addItems(self.instr_type_list)
        self.instr_type_combo.setFocusPolicy(Qt.NoFocus)
        self.instr_type_combo.currentIndexChanged .connect(self.instrTypeChanged)
        self.instr_combo = QComboBox()
        self.instr_combo.setStyleSheet(self.combo_style)
        self.instr_combo.setFont(self.input_font)
        self.instr_combo.addItems(self.instr_list[self.instr_type_index])
        self.instr_combo.setFocusPolicy(Qt.NoFocus)
        self.instr_combo.currentIndexChanged .connect(self.instrChanged)
        self.instr_box.addWidget(self.instr_lbl)
        self.instr_box.addWidget(self.instr_type_combo)
        self.instr_box.addWidget(self.instr_combo)

        self.pianoroll_option_box = QVBoxLayout()
        self.pianoroll_option_box.addLayout(self.volume_box)
        self.pianoroll_option_box.addLayout(self.octave_box)
        self.pianoroll_option_box.addLayout(self.instr_box)
        # self.pianoroll_option_box.addLayout(self.mode_display_box)
        self.pianoroll_option_box.setStretch(0, 5)
        self.pianoroll_option_box.setStretch(1, 5)
        self.pianoroll_option_box.setStretch(2, 5)

        self.option_box = QHBoxLayout()
        self.option_box.addLayout(self.record_box)
        self.option_box.addLayout(self.pianoroll_option_box)
        self.option_box.setStretch(0, 9)
        self.option_box.setStretch(1, 6)

        self.optionField = QWidget()
        self.optionField.setObjectName('OptionField')
        self.optionField.setLayout(self.option_box)
        self.optionField.setStyleSheet('QWidget#OptionField{background-color: #CDC0B0; border: 5px ridge #8B795E;}')

        self.pianorollBox = QHBoxLayout()
        self.pianorollWindow = QMainWindow()
        self.pianorollWindow.setObjectName('PianoRollWindow')
        self.pianorollWindow.setStyleSheet('QMainWindow#PianoRollWindow{background-color: #FFF5EE;}')
        self.pianorollBox.addWidget(self.pianorollWindow)
        self.pianorollWindow.setCentralWidget(self.piano_roll)

        self.wholeLayout = QVBoxLayout()
        self.wholeLayout.addWidget(self.optionField)
        self.wholeLayout.addLayout(self.pianorollBox)

        self.setLayout(self.wholeLayout)
        self.piano_roll.setFocus()

    def recordStart(self):
        self.is_record_mode = True
        self.music_segment = MusicSegment(self.metre, self.beats_per_minute, self.length_per_note, self.root_note, (self.mode_type, self.mode_name))
        self.piano_roll.start_recording(self.music_segment)
        self.record_start_btn.setStyleSheet('QPushButton{background: #8B8682; border: 3px inset #8B8378; border-radius: 15px;}')

    def recordDraw(self):
        if self.is_record_mode == True and self.music_segment != None:
            if self.music_segment.window_on == False:
                self.music_segment.segment_window = SegmentWindow(self.music_segment)
                self.music_segment.segment_window.show()
                self.music_segment.segment_window.move(1510, 125)
                self.music_segment.window_on = True
        else:
            msg = QMessageBox()
            msg.setWindowIcon(QIcon('./icon/gramophone.png'))
            msg.setIcon(QMessageBox.Information)
            msg.setText('Please first start record mode')
            msg.setWindowTitle('Not in Record Mode')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            retval = msg.exec_()


    def recordPlay(self):
        if self.is_record_mode == True and self.music_segment != None:
            self.music_segment.play_music(self.piano_roll.player)
        else:
            msg = QMessageBox()
            msg.setWindowIcon(QIcon('./icon/gramophone.png'))
            msg.setIcon(QMessageBox.Information)
            msg.setText('Please first start record mode')
            msg.setWindowTitle('Not in Record Mode')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            retval = msg.exec_()

    def recordReturn(self):
        if self.is_record_mode == True and self.music_segment != None:
            self.music_segment.delete_last_msg()
            self.music_segment.replot()
        else:
            msg = QMessageBox()
            msg.setWindowIcon(QIcon('./icon/gramophone.png'))
            msg.setIcon(QMessageBox.Information)
            msg.setText('Please first start record mode')
            msg.setWindowTitle('Not in Record Mode')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            retval = msg.exec_()

    def recordFinish(self):
        self.is_record_mode = False
        self.record_start_btn.setStyleSheet(self.record_btn_style)
        # self.music_segment.turn_into_numpy_matrix()
        # self.music_segment.plot_canvas()
        # self.music_segment.play_music(self.piano_roll.player)

    def recordStop(self):
        self.is_record_mode = False
        self.record_start_btn.setStyleSheet(self.record_btn_style)

    def LPNChanged(self):
        self.length_per_note = self.length_per_note_effects[self.LPN_ctrl.currentIndex()]
        if self.is_record_mode:
            self.music_segment.length_per_note = self.length_per_note

    def metreChanged(self):
        self.metre = self.metre_ctrl.currentText()
        self.metre_numerator = self.metre_effects[self.metre][0]
        self.metre_denominator = self.metre_effects[self.metre][1]

        self.music_segment.metre = self.metre
        self.music_segment.metre_numerator = self.metre_numerator
        self.music_segment.metre_denominator = self.metre_denominator

        if self.is_record_mode:
            self.music_segment.metre = self.metre
            self.music_segment.metre_numerator = self.metre_numerator
            self.music_segment.metre_denominator = self.metre_denominator

    def BPMChanged(self):
        self.beats_per_minute = self.BPM_ctrl.value()
        if self.is_record_mode:
            self.music_segment.bpm = self.beats_per_minute

    def rootNoteChanged(self):
        self.root_note_index = self.root_note_ctrl.currentIndex()
        self.root_note = self.root_notes_list[self.root_note_index]
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

        if self.is_record_mode:
            self.music_segment.root_note = self.root_note
            self.piano_roll.start_recording(self.music_segment)

    def modeTypeChanged(self):
        self.mode_combo.clear()
        self.mode_type_index = self.mode_type_combo.currentIndex()
        self.mode_type = self.mode_type_list[self.mode_type_combo.currentIndex()]
        self.mode_combo.addItems(self.mode_list[self.mode_type_index])
        self.mode_combo.update()
        if self.is_record_mode:
            self.music_segment.mode = (self.mode_type, self.mode_name)
            self.piano_roll.start_recording(self.music_segment)
        # self.mode_name = self.mode_combo.currentText()
        # print((self.mode_type, self.mode_name))
        #  print((self.mode_type_combo.currentText(), self.mode_combo.currentText()))
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()


    def modeChanged(self):
        self.mode_name = self.mode_list[self.mode_type_index][self.mode_combo.currentIndex()]
        if self.is_record_mode:
            self.music_segment.mode = (self.mode_type, self.mode_name)
            self.piano_roll.start_recording(self.music_segment)
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()


    def volumeChanged(self):
        self.volume = self.volume_ctrl.value()
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

    def octaveChanged(self):
        self.octave = self.octave_ctrl.value()

        self.root_notes_list = [60 + (self.octave - 4) * 12 + distance for distance in range(12)]
        self.root_note_names = [get_note_name_by_midi_value(value) for value in self.root_notes_list]
        self.root_note = self.root_notes_list[self.root_note_index]

        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        if self.is_record_mode:
            self.piano_roll.start_recording(self.music_segment)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()


    def instrTypeChanged(self):
        self.instr_type_index = self.instr_type_combo.currentIndex()
        self.instr_combo.clear()
        self.instr_combo.addItems(self.instr_list[self.instr_type_index])
        self.instr_index = sum(get_instrument_margin()[:self.instr_type_index]) + self.instr_combo.currentIndex()
        self.instr_combo.update()
        self.piano_roll.change_instrument(self.instr_index)
        self.pianorollWindow.setFocus()

    def instrChanged(self):
        self.instr_index = sum(get_instrument_margin()[:self.instr_type_index]) + self.instr_combo.currentIndex()
        self.piano_roll.change_instrument(self.instr_index)
        self.pianorollWindow.setFocus()

    def modeDisplayChanged(self):
        self.mode_display = self.mode_display_check.isChecked()
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name), self.root_note, self.mode_display)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

    def octaveIncrease(self):
        if self.octave < 8:
            self.octave = self.octave + 1
            self.octave_ctrl.setValue(self.octave)

            self.root_notes_list = [60 + (self.octave - 4) * 12 + distance for distance in range(12)]
            self.root_note_names = [get_note_name_by_midi_value(value) for value in self.root_notes_list]
            self.root_note = self.root_notes_list[self.root_note_index]

            self.piano_roll.delete_player()
            self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name),
                                        self.root_note, self.mode_display)
            if self.is_record_mode:
                self.piano_roll.start_recording(self.music_segment)
            self.pianorollWindow.setCentralWidget(self.piano_roll)
            self.pianorollWindow.setFocus()

    def octaveDecrease(self):
        if self.octave > 0:
            self.octave = self.octave - 1
            self.octave_ctrl.setValue(self.octave)

            self.root_notes_list = [60 + (self.octave - 4) * 12 + distance for distance in range(12)]
            self.root_note_names = [get_note_name_by_midi_value(value) for value in self.root_notes_list]
            self.root_note = self.root_notes_list[self.root_note_index]

            self.piano_roll.delete_player()
            self.piano_roll = PianoRoll(self.volume, self.octave, self.instr_index, (self.mode_type, self.mode_name),
                                        self.root_note, self.mode_display)
            if self.is_record_mode:
                self.piano_roll.start_recording(self.music_segment)
            self.pianorollWindow.setCentralWidget(self.piano_roll)
            self.pianorollWindow.setFocus()


