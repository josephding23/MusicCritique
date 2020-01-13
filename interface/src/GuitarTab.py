from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame.midi
import sys

from midi_extended.UtilityBox import *

class GuitarTab(QWidget):

    def __init__(self):
        super().__init__()
        self.setObjectName('GuitarTab')
        self.empty_notes = [52, 57, 62, 67, 71, 76]
        self.empty_margin = [5, 5, 5, 4, 5]
        self.major_keys = [0, 2, 4, 5, 7, 9, 11]
        self.total_fret_num = 21
        self.total_string_num = 6
        self.instr = 29

        pygame.midi.init()
        self.player = pygame.midi.Output(0)
        self.player.set_instrument(self.instr)

        self.initUI()

    def initUI(self):
        self.tab_layout = QGridLayout()
        self.tab_layout.setContentsMargins(10, 70, 0, 70)
        for string_num in range(self.total_string_num):
            for fret_num in range(self.total_fret_num+1):
                note = self.empty_notes[5-string_num] + fret_num
                fret_note = FretNode(self.player, note, show_name=note % 12 in self.major_keys)
                self.tab_layout.addWidget(fret_note, string_num, fret_num)

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap('../pic/guitar_tab.png')))

        self.setPalette(palette)
        self.setLayout(self.tab_layout)
        self.setFixedSize(1500, 550)


class FretNode(QWidget):

    def __init__(self, player, note, volume=63, color='w', show_name=False):
        super().__init__()
        self.player = player
        self.setObjectName('FretNode')
        self.note = note
        self.volume = volume
        self.note_name = get_note_name_by_midi_value(self.note)
        self.color = color
        self.show_name = show_name
        self.initUI()

    def initUI(self):
        self.keyBtn = QPushButton()
        if self.show_name:
            self.keyBtn.setText(self.note_name)
        self.note_font = QFont()
        self.note_font.setFamily('Times New Roman')
        self.note_font.setWeight(QFont.DemiBold)
        self.note_font.setPixelSize(14)
        self.keyBtn.setFont(self.note_font)
        self.keyBtn.setFixedHeight(20)
        self.keyBtn.setFixedWidth(32)
        self.keyBtn.setCursor(Qt.PointingHandCursor)
        self.keyBtn.pressed.connect(self.pressedKeyResponse)
        self.keyBtn.released.connect(self.releasedKeyResponse)

        self.setStyleSheet(
            'QPushButton{background-color: white; color: black; border: 4px outset #828282;}'
            'QPushButton:hover{background-color: #B5B5B5; color: black; border: 4px outset #828282;}'
            'QPushButton:pressed{background-color: #CFCFCF; color: black; border: 4px inset #828282;}'
        )

        self.keyLayout = QVBoxLayout()
        self.keyLayout.addWidget(self.keyBtn)
        self.setLayout(self.keyLayout)

    def pressedKeyResponse(self):
        # self.note_sig.emit()
        self.player.note_on(self.note, self.volume)

    def releasedKeyResponse(self):
        self.player.note_off(self.note, self.volume)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    guitar_tab = GuitarTab()
    guitar_tab.show()
    sys.exit(app.exec_())