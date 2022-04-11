from PyQt5 import QtCore
from PyQt5 import QtGui as qt
from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import itertools


class MultiTabNavTool(NavigationToolbar):
    # ====================================================================================================
    def __init__(self, canvases, tabs, parent=None):
        self.canvases = canvases
        self.tabs = tabs

        NavigationToolbar.__init__(self, canvases[0], parent)

    # ====================================================================================================
    def get_canvas(self):
        return self.canvases[self.tabs.currentIndex()]

    def set_canvas(self, canvas):
        self._canvas = canvas

    canvas = property(get_canvas, set_canvas)


class MplMultiTab(QtWidgets.QMainWindow):
    # ====================================================================================================
    def __init__(self, parent=None, figures=None, labels=None):
        qt.QMainWindow.__init__(self, parent)

        self.main_frame = qt.QWidget()
        self.tabWidget = qt.QTabWidget(self.main_frame)
        self.create_tabs(figures, labels)

        # Create the navigation toolbar, tied to the canvas
        self.mpl_toolbar = MultiTabNavTool(self.canvases, self.tabWidget, self.main_frame)

        self.vbox = vbox = qt.QVBoxLayout()
        vbox.addWidget(self.mpl_toolbar)
        vbox.addWidget(self.tabWidget)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    # ====================================================================================================
    def create_tabs(self, figures, labels):

        if labels is None:      labels = []
        figures = [
            Figure()] if figures is None else figures  # initialise with empty figure in first tab if no figures provided
        self.canvases = [self.add_tab(fig, lbl)
                         for (fig, lbl) in itertools.zip_longest(figures, labels)]

    # ====================================================================================================
    def add_tab(self, fig=None, name=None):
        '''dynamically add tabs with embedded matplotlib canvas with this function.'''

        # Create the mpl Figure and FigCanvas objects.
        if fig is None:
            fig = Figure()
            ax = fig.add_subplot(111)

        canvas = fig.canvas if fig.canvas else FigureCanvas(fig)
        canvas.setParent(self.tabWidget)
        canvas.setFocusPolicy(QtCore.Qt.ClickFocus)

        # self.tabs.append( tab )
        name = 'Tab %i' % (self.tabWidget.count() + 1) if name is None else name
        self.tabWidget.addTab(canvas, name)

        return canvas
