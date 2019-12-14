from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import os
from interface.src.Piano import Piano

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class Interface(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setCentralWidget(Piano())
        self.resize(1500, 600)
        self.setFixedSize(self.width(), self.height())
        self.setWindowIcon(QIcon('./icon/gramophone.png'))
        self.setWindowTitle('MusicCritique Interface')
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Interface()
    ex.setWindowOpacity(0.95)
    sys.exit(app.exec_())