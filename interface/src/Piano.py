from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame.midi
from fractions import Fraction
import time

from midi_extended.UtilityBox import *
from interface.src.MusicSegment import MusicSegment, SegmentWindow
from interface.src.PianoRoll import PianoRoll

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
        self.record_start_btn.setIcon(QIcon('../icon/clipboard_start.png'))
        self.record_start_btn.setToolTip('Start Recording')
        self.record_start_btn.setIconSize(self.record_btn_icon_size)
        self.record_start_btn.setFixedSize(self.record_btn_size)
        self.record_start_btn.setStyleSheet(self.record_btn_style)
        self.record_start_btn.setCursor(Qt.PointingHandCursor)
        self.record_start_btn.clicked.connect(self.recordStart)

        self.record_draw_btn = QPushButton()
        self.record_draw_btn.setIcon(QIcon('../icon/clipboard_see.png'))
        self.record_draw_btn.setToolTip('Draw Notes Plot')
        self.record_draw_btn.setIconSize(self.record_btn_icon_size)
        self.record_draw_btn.setFixedSize(self.record_btn_size)
        self.record_draw_btn.setStyleSheet(self.record_btn_style)
        self.record_draw_btn.setCursor(Qt.PointingHandCursor)
        self.record_draw_btn.clicked.connect(self.recordDraw)

        self.record_play_btn = QPushButton()
        self.record_play_btn.setIcon(QIcon('../icon/clipboard_play.png'))
        self.record_play_btn.setToolTip('Play Recorded Segment')
        self.record_play_btn.setIconSize(self.record_btn_icon_size)
        self.record_play_btn.setFixedSize(self.record_btn_size)
        self.record_play_btn.setStyleSheet(self.record_btn_style)
        self.record_play_btn.setCursor(Qt.PointingHandCursor)
        self.record_play_btn.clicked.connect(self.recordPlay)

        self.record_return_btn = QPushButton()
        self.record_return_btn.setIcon(QIcon('../icon/clipboard_return.png'))
        self.record_return_btn.setToolTip('Delete Last Note')
        self.record_return_btn.setIconSize(self.record_btn_icon_size)
        self.record_return_btn.setFixedSize(self.record_btn_size)
        self.record_return_btn.setStyleSheet(self.record_btn_style)
        self.record_return_btn.setCursor(Qt.PointingHandCursor)
        self.record_return_btn.clicked.connect(self.recordReturn)

        self.record_finish_btn = QPushButton()
        self.record_finish_btn.setIcon(QIcon('../icon/clipboard_finish.png'))
        self.record_finish_btn.setToolTip('Finish Recording')
        self.record_finish_btn.setIconSize(self.record_btn_icon_size)
        self.record_finish_btn.setFixedSize(self.record_btn_size)
        self.record_finish_btn.setStyleSheet(self.record_btn_style)
        self.record_finish_btn.setCursor(Qt.PointingHandCursor)
        self.record_finish_btn.clicked.connect(self.recordFinish)

        self.record_stop_btn = QPushButton()
        self.record_stop_btn.setIcon(QIcon('../icon/clipboard_stop.png'))
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
            msg.setWindowIcon(QIcon('../icon/gramophone.png'))
            msg.setIcon(QMessageBox.Information)
            msg.setText('Please first start record mode')
            msg.setWindowTitle('Not in Record Mode')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            retval = msg.exec_()

    def recordFinish(self):
        if self.is_record_mode == True and self.music_segment != None:
            self.is_record_mode = False
            self.record_start_btn.setStyleSheet(self.record_btn_style)
            save_box = QMessageBox()
            save_box.setWindowIcon(QIcon('../icon/gramophone.png'))
            save_box.setIcon(QMessageBox.Question)
            save_box.setText('Save Music Segment?')
            save_box.setWindowTitle('Save Music Segment')
            save_box.setStandardButtons(QMessageBox.Save | QMessageBox.No)
            # save_box.buttonClicked.connect(self.saveBtnResponse)
            save_box.show()

            retval = save_box.exec_()
            if retval == QMessageBox.Save:
                print('save')
                file_name, _ = QFileDialog.getSaveFileName(self, r'Please choose where to save',
                                                             '.\..\data\music_segments', r'JSON Files(*.json)')
                self.music_segment.save(file_name)
            elif retval == QMessageBox.No:
                print('no')
            else:
                pass
        else:
            msg = QMessageBox()
            msg.setWindowIcon(QIcon('../icon/gramophone.png'))
            msg.setIcon(QMessageBox.Information)
            msg.setText('Please first start record mode')
            msg.setWindowTitle('Not in Record Mode')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            retval = msg.exec_()

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


