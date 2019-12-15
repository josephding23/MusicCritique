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
import sys

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
        final_time = msgs[-1][2] + msgs[-1][1]
        min_note = min([msg[0] for msg in msgs]) - 6
        max_note = max([msg[0] for msg in msgs]) + 6
        note_texts = [get_note_name_by_midi_value(note) for note in range(min_note, max_note+1)]
        self.axes.set_xlim(0, final_time)
        self.axes.set_xticks(np.arange(0, final_time, 0.125))
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
            print(note, time, length)
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

    def closeEvent(self, e):
        self.music_segment.window_on = False


    def replot(self, music_segment):
        self.music_segment = music_segment
        self.segment_canvas = SegmentCanvas()
        self.segment_canvas.plot(self.music_segment.msgs)
        graphic_scene = QGraphicsScene()
        graphic_scene.addWidget(self.segment_canvas)
        self.graphic_view.setScene(graphic_scene)
        self.graphic_view.show()
        self.setCentralWidget(self.graphic_view)

class MusicSegment:
    def __init__(self, metre, bpm, length_per_note, total_length=128):
        self.metre_effects = {
            '1/4': (1, 4),
            '2/4': (2, 4),
            '3/4': (3, 4),
            '4/4': (4, 4),
            '3/8': (3, 8),
            '6/8': (6, 8)
        }

        self.metre = metre
        self.metre_numerator = self.metre_effects[self.metre][0]
        self.metre_denominator = self.metre_effects[self.metre][1]

        self.time_scale = 16

        self.bpm = bpm
        print(self.bpm)
        self.time_per_unit = ( 60 / self.bpm ) * 4
        print(self.time_per_unit)
        self.total_length = total_length
        self.length_per_note = length_per_note
        # self.matrix  = self.turn_into_numpy_matrix()
        self.msgs = []
        self.time_stamps = []
        # self.canvas = SegmentCanvas()

        self.segment_window = None
        self.window_on = False

    def add_note(self, note, raw_time):
        time = raw_time
        self.time_stamps.append(time)
        msg = (note, time, sum(self.time_stamps[:-1]))
        self.msgs.append(msg)
        # self.time_and_note.append((sum([msg[1] for msg in self.msgs]), note))

    def delete_last_msg(self):
        self.msgs.pop(len(self.msgs)-1)
        self.time_stamps.pop(len(self.time_stamps)-1)

    def replot(self):
        self.segment_window.segment_canvas = SegmentCanvas()
        self.segment_window.segment_canvas.plot(self.msgs)
        graphic_scene = QGraphicsScene()
        graphic_scene.addWidget(self.segment_window.segment_canvas)
        self.segment_window.graphic_view.setScene(graphic_scene)
        self.segment_window.graphic_view.show()
        self.segment_window.setCentralWidget(self.segment_window.graphic_view)

    def print_notes(self):
        for msg in self.msgs:
            print(msg)

    def play_music(self, player):
        for msg in self.msgs:
            player.note_on(msg[0], 127)
            time.sleep(msg[1] * self.time_per_unit)
            print(msg[1])
            player.note_off(msg[0], 127)

    def turn_into_numpy_matrix(self):
        # time_per_unit = 60 * 60 * 10 / self.bpm / 4
        notes = [msg[0] for msg in self.msgs]
        length_units = [round(msg[1] * self.metre_denominator) for msg in self.msgs]
        print(length_units)

        piano_roll = np.zeros((sum(length_units), 128))

        times = []
        for i in range(len(length_units)):
            time_point = 0
            time_point = sum(length_units[:i])
            times.append(time_point)

        for i in range(len(times) - 1):
            start = times[i]
            end = times[i + 1]
            note = notes[i]
            for time in range(start, end):
                piano_roll[time][note] = 1
        # plt.scatter(times, notes)
        # plt.show()
        # np.save(path, piano_roll)
        return piano_roll
