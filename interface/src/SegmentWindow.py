import numpy as np
import time
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from midi_extended.UtilityBox import *

from midi_extended.UtilityBox import *


class SegmentCanvas(FigureCanvas):

    def __init__(self, width=4, height=7,  dpi=100):
        fig = Figure(figsize = (width, height), dpi=dpi, facecolor='#FFFAF0')
        FigureCanvas.__init__(self, fig)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlim(0, 8)
        self.axes.set_xticks(np.arange(0, 8, 1))
        self.axes.set_ylim(60, 84)
        self.axes.set_yticks(np.arange(60, 84, 12))

    def plot(self, msgs):
        try:
            final_time = msgs[-1][2] + msgs[-1][1]
            min_note = min([min([msg[0] for msg in msgs]) - 6, 60])
            max_note = max([max([msg[0] for msg in msgs]) + 6, 72])
        except:
            final_time = 0.5
            min_note = 60
            max_note = 72
        note_texts = [get_note_name_by_midi_value(note) for note in range(min_note, max_note+1)]
        self.axes.set_xlim(0, final_time)
        self.axes.set_xticks(np.arange(0, final_time, 0.5))
        self.axes.set_ylim(min_note, max_note)
        self.axes.set_yticks(np.arange(min_note, max_note, 1))
        self.axes.tick_params(axis='both', which='minor', labelsize=10)
        # print(note_texts)
        self.axes.set_yticklabels(note_texts)
        bar_width = 0.5
        for msg in msgs:
            note = msg[0]
            length = msg[1]
            time = msg[2]
            verts = [
                (time, note - bar_width/2),
                (time, note + bar_width/2),
                (time + length, note + bar_width/2),
                (time + length, note - bar_width/2),
                (time, note - bar_width/2)
            ]
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='#CD853F', lw=1)
            self.axes.add_patch(patch)
        # self.axes.plot(times, notes)


class SegmentWindow(QMainWindow):
    def __init__(self, music_segment):
        super().__init__()
        self.music_segment = music_segment
        self.initUI()

    def initUI(self):
        self.graphic_view = QGraphicsView()
        self.graphic_view.setObjectName('graphic_view')

        self.segment_canvas = SegmentCanvas()
        self.segment_canvas.plot(self.music_segment.msgs)
        graphic_scene = QGraphicsScene()
        graphic_scene.addWidget(self.segment_canvas)
        self.graphic_view.setScene(graphic_scene)
        self.graphic_view.show()
        self.setCentralWidget(self.graphic_view)
        self.graphic_view.setFixedSize(400, 700)

        self.setWindowIcon(QIcon('./icon/gramophone.png'))
        self.setWindowTitle('MusicCritique Notes Display')
        self.move(1505, 125)
        self.show()

    def closeEvent(self, e):
        self.music_segment.window_on = False

