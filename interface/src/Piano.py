from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame.midi
import time

from midi_extended.UtilityBox import *


class Key(QWidget):

    def __init__(self, player, note, color='w'):
        super().__init__()
        self.player = player
        self.note = note
        self.note_name = get_note_name_by_midi_value(self.note)
        # print(self.note_name)
        self.color = color
        self.initUI()

    def initUI(self):
        self.keyBtn = QPushButton(self.note_name)
        self.keyBtn.setFixedHeight(225)
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
                'QPushButton{background-color: black; color: white; border: 4px outset #9C9C9C;}'
                'QPushButton:hover{background-color: #4F4F4F; color: white; border: 4px outset #9C9C9C;}'
                'QPushButton:pressed{background-color: #363636; color: white; border: 4px inset #9C9C9C;}'
            )
        # self.resize(3, 20)

    def pressedKeyResponse(self):
        self.player.note_on(self.note, 127)

    def releasedKeyResponse(self):
        self.player.note_off(self.note, 127)


class PianoRoll(QWidget):

    def __init__(self, octave, instr=0):
        super().__init__()
        self.setObjectName('PianoRoll')
        pygame.midi.init()
        self.player = pygame.midi.Output(0)
        self.player.set_instrument(instr)

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

        self.octave = octave
        self.root_note = 60 + (self.octave - 4) * 12
        self.initUI()

    def delete_player(self):
        self.player.close()
        pygame.midi.quit()

    def initUI(self):
        self.blackKeysBoxes = [QHBoxLayout(), QHBoxLayout()]
        self.whiteKeysBoxes = [QHBoxLayout(), QHBoxLayout()]
        self.octaveRollBoxes = [QVBoxLayout(), QVBoxLayout()]

        for group in range(2):
            for index in range(len(self.black_notes_distance[group])):
                new_key = Key(self.player, self.root_note + self.black_notes_distance[group][index], 'b')

                self.blackKeysList[group].append(new_key)

            for index in range(len(self.white_notes_distance[group])):
                new_key = Key(self.player, self.root_note + self.white_notes_distance[group][index])
                # print(self.root_note + self.white_notes_distance[group][index])
                # shortcut = QShortcut(QKeySequence(self.white_shortcuts[index]), self)
                # shortcut.activated.connect(new_key.pressedKeyResponse)
                # self.whiteKeysBox.addWidget()
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

        self.whiteKeysBoxes[0].setContentsMargins(0, 0, 0, 0)
        self.whiteKeysBoxes[1].setContentsMargins(0, 0, 0, 0)
        self.whiteKeysBoxes[0].addSpacing(25)
        self.whiteKeysBoxes[1].addSpacing(25)
        self.blackKeysBoxes[0].setContentsMargins(50, 0, 50, 0)
        self.blackKeysBoxes[1].setContentsMargins(50, 0, 150, 0)

        self.wholeLayout = QHBoxLayout()
        self.wholeLayout.addLayout(self.octaveRollBoxes[0])
        self.wholeLayout.addLayout(self.octaveRollBoxes[1])
        self.wholeLayout.setStretch(0, 7)
        self.wholeLayout.setStretch(1, 8)
        self.wholeLayout.setSpacing(1)
        self.setLayout(self.wholeLayout)
        self.resize(1500, 500)

    def keyPressEvent(self, e):
        for group in range(2):
            for index in range(len(self.white_shortcuts[group])):
                if e.key() == self.white_shortcuts[group][index]:
                    self.player.note_on(self.whiteKeysList[group][index].note, 127)
                    self.whiteKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
                    )
                    return
            for index in range(len(self.black_shortcuts[group])):
                if e.key() == self.black_shortcuts[group][index]:
                    self.player.note_on(self.blackKeysList[group][index].note, 127)
                    self.blackKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: #363636; color: white; border: 4px inset #9C9C9C;}'
                    )
                    return

    def keyReleaseEvent(self, e):
        for group in range(2):
            for index in range(len(self.white_shortcuts[group])):
                if e.key() == self.white_shortcuts[group][index]:
                    self.player.note_off(self.whiteKeysList[group][index].note, 127)
                    self.whiteKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: white; color: black; border: 4px outset #828282;}'
                        'QPushButton:hover{background-color: #B5B5B5; color: black; border: 4px outset #828282;}'
                        'QPushButton:pressed{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
                    )
                    return
            for index in range(len(self.black_shortcuts[group])):
                if e.key() == self.black_shortcuts[group][index]:
                    self.player.note_off(self.blackKeysList[group][index].note, 127)
                    self.blackKeysList[group][index].setStyleSheet(
                        'QPushButton{background-color: black; color: white; border: 4px outset #9C9C9C;}'
                        'QPushButton:hover{background-color: #4F4F4F; color: white; border: 4px outset #9C9C9C;}'
                        'QPushButton:pressed{background-color: #363636; color: white; border: 4px inset #9C9C9C;}'
                    )
                    return

class Piano(QWidget):

    def __init__(self):
        super().__init__()
        self.octave = 4
        self.control = 0
        self.length_per_note = 2
        self.piano_roll = PianoRoll(self.octave)

        self.control_lbl_style = 'QLable{text-align: center; color: #8B795E; font-style: Century Gothic; font-size: 16;}'
        self.option_lbl_font = QFont()
        self.option_lbl_font.setFamily('Century Gothic')
        self.option_lbl_font.setWeight(QFont.Bold)
        self.option_lbl_font.setStyle(QFont.StyleItalic)
        self.option_lbl_font.setPixelSize(16)

        self.combo_style = 'QComboBox{background-color: #FFF5EE; color: #8B5A2B;}'
        self.spin_style = 'QSpinBox{background-color: #FFF5EE; color: #8B5A2B;}'
        self.input_font = QFont()
        self.input_font.setFamily('Century Gothic')
        self.input_font.setWeight(QFont.DemiBold)
        self.input_font.setStyle(QFont.StyleNormal)
        self.input_font.setPixelSize(12)

        self.instr_type_list = get_instrument_types()
        self.instr_list = get_instrument_list()
        self.instr_type_index = 0
        self.instr_index = 0

        self.is_record_mode = False
        self.length_per_note_options = ['1/8', '1/6', '1/4', '1/3', '1/2', '1', '2', '3', '4', '6', '8']
        self.initUI()

    def initUI(self):
        self.record_btn_icon_size = QSize(30, 30)
        self.record_btn_size = QSize(45, 45)
        self.record_btn = QPushButton()
        self.record_btn.setIcon(QIcon('./icon/record.png'))
        self.record_btn.setIconSize(self.record_btn_icon_size)
        self.record_btn.setFixedSize(self.record_btn_size)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon('./icon/stop.png'))
        self.stop_btn.setIconSize(self.record_btn_icon_size)
        self.stop_btn.setFixedSize(self.record_btn_size)

        self.record_btn_box = QHBoxLayout()
        self.record_btn_box.addWidget(self.record_btn)
        self.record_btn_box.addWidget(self.stop_btn)

        self.LPN_lbl = QLabel('LPN: ')
        self.LPN_lbl.setAlignment(Qt.AlignCenter)
        self.LPN_lbl.setFont(self.option_lbl_font)
        self.LPN_ctrl = QComboBox()
        self.LPN_ctrl.addItems(self.length_per_note_options)
        self.LPN_ctrl.setFont(self.input_font)
        self.LPN_ctrl.setStyleSheet(self.combo_style)
        self.LPN_box = QHBoxLayout()
        self.LPN_box.addWidget(self.LPN_lbl)
        self.LPN_box.addWidget(self.LPN_ctrl)

        self.record_option_box = QVBoxLayout()
        self.record_option_box.addLayout(self.record_btn_box)
        self.record_option_box.addLayout(self.LPN_box)

        self.octave_box = QHBoxLayout()
        self.octave_lbl = QLabel('Octave: ')
        self.octave_lbl.setAlignment(Qt.AlignCenter)
        self.octave_lbl.setFont(self.option_lbl_font)
        self.octave_ctrl = QSpinBox()
        self.octave_ctrl.setStyleSheet(self.spin_style)
        self.octave_ctrl.setAlignment(Qt.AlignCenter)
        self.octave_ctrl.setFont(self.input_font)
        self.octave_ctrl.setRange(0, 8)
        self.octave_ctrl.setValue(self.octave)
        self.octave_ctrl.setFocusPolicy(Qt.NoFocus)
        self.octave_ctrl.valueChanged.connect(self.octaveChanged)
        self.octave_box.addWidget(self.octave_lbl)
        self.octave_box.addWidget(self.octave_ctrl)
        octave_up = QShortcut(QKeySequence(Qt.Key_PageUp), self)
        octave_up.activated.connect(self.octaveIncrease)
        octave_down = QShortcut(QKeySequence(Qt.Key_PageDown), self)
        octave_down.activated.connect(self.octaveDecrease)

        self.instr_box = QHBoxLayout()
        self.instr_lbl = QLabel('Instrument: ')
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
        self.pianoroll_option_box.addLayout(self.octave_box)
        self.pianoroll_option_box.addLayout(self.instr_box)

        self.option_box = QHBoxLayout()
        self.option_box.addLayout(self.record_option_box)
        self.option_box.addLayout(self.pianoroll_option_box)

        self.optionField = QWidget()
        self.optionField.setObjectName('OptionField')
        self.optionField.setLayout(self.option_box)
        self.optionField.setStyleSheet('QWidget#OptionField{background-color: #CDC0B0; border: 5px ridge #8B795E;}')

        self.pianorollBox = QHBoxLayout()
        self.pianorollWindow = QMainWindow()
        self.pianorollWindow.setObjectName('PianoRollWindow')
        self.pianorollWindow.setStyleSheet('QMainWindow#PianoRollWindow{background-color: #FFF5EE;}')
        self.pianorollWindow.setFocus()
        self.pianorollBox.addWidget(self.pianorollWindow)
        self.pianorollWindow.setCentralWidget(self.piano_roll)

        self.wholeLayout = QVBoxLayout()
        self.wholeLayout.addWidget(self.optionField)
        self.wholeLayout.addLayout(self.pianorollBox)

        self.setLayout(self.wholeLayout)

    def octaveChanged(self):
        self.octave = self.octave_ctrl.value()
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.octave)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

    def instrTypeChanged(self):
        self.instr_type_index = self.instr_type_combo.currentIndex()
        self.instr_combo.clear()
        self.instr_combo.addItems(self.instr_list[self.instr_type_index])
        self.instr_index = self.instr_type_index * 8 + self.instr_combo.currentIndex()
        self.instr_combo.update()
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.octave, self.instr_index)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

    def instrChanged(self):
        self.instr_index = self.instr_type_index * 8 + self.instr_combo.currentIndex()
        self.piano_roll.delete_player()
        self.piano_roll = PianoRoll(self.octave, self.instr_index)
        self.pianorollWindow.setCentralWidget(self.piano_roll)
        self.pianorollWindow.setFocus()

    def octaveIncrease(self):
        if self.octave < 8:
            self.octave = self.octave + 1
            self.octave_ctrl.setValue(self.octave)
            self.piano_roll.delete_player()
            self.piano_roll = PianoRoll(self.octave)
            self.pianorollWindow.setCentralWidget(self.piano_roll)
            self.pianorollWindow.setFocus()

    def octaveDecrease(self):
        if self.octave > 0:
            self.octave = self.octave - 1
            self.octave_ctrl.setValue(self.octave)
            self.piano_roll.delete_player()
            self.piano_roll = PianoRoll(self.octave)
            self.pianorollWindow.setCentralWidget(self.piano_roll)
            self.pianorollWindow.setFocus()


