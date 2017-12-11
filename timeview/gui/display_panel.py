# stdlib
import logging
from typing import List, Optional
from operator import sub

# 3rd party
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Slot, Signal

# our modules
from .model import Panel, View
from .plot_area import DumbPlot
from .view_table import ViewTable

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

icon_color = QtGui.QColor('#00897B')


class Spacer(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)


class Handle(Spacer):

    def __init__(self, parent, label: str='test'):
        super().__init__()
        self.setParent(parent)
        self.setFixedWidth(30)
        self.label = QtWidgets.QLabel()
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
        self.setLayout(self.layout)
        self.label.setText(str(label))

    @Slot(name='update_label')
    def updateLabel(self):
        panel_obj = self.parent().panel
        if panel_obj is None:
            return
        index = self.parent().main_window.model.panels.index(panel_obj)
        self.label.setText(str(index + 1))


class TableSplitter(QtWidgets.QSplitter):
    position = Signal(list, name='position')

    def __init__(self, parent):
        super().__init__()
        self.setParent(parent)
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setHandleWidth(10)
        self.is_collapsed = False
        self.old_size = 0
        self.setStyleSheet("QSplitter::handle{background: darkgray;}")
        self.position.connect(self.parent().main_window.setSplitter)
        self.splitterMoved.connect(self.moveFinished)

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) \
            -> QtWidgets.QWidget.eventFilter:
        if event.type() == QtCore.QEvent.MouseButtonDblClick:
            self.is_collapsed = ~self.is_collapsed
            self.showOrHideChild()
        return QtWidgets.QWidget.eventFilter(self, source, event)

    def showOrHideChild(self):
        if self.is_collapsed:
            self.old_size = self.sizes()[1]
            self.setSizes([1, 0])
        else:
            self.setSizes([1, self.old_size])
        self.moveFinished()

    @Slot(int, int, name='moveFinished')
    def moveFinished(self):
        self.position.emit([1, self.sizes()[1]])

    @Slot(list, name='setSizes')
    def setSizes_(self, sizes: List[int]):
        self.setSizes(sizes)


class Frame(QtWidgets.QFrame):
    select_me = Signal(QtWidgets.QFrame, name='select_me')
    select_previous = Signal(name='select_previous')
    select_next = Signal(name='select_next')
    move_me = Signal(QtWidgets.QFrame, name='move_me')
    move_up = Signal(name='move_up')
    move_down = Signal(name='move_down')
    insert_here = Signal(QtWidgets.QFrame, name='insert_here')

    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParent(main_window)
        self.application = main_window.application
        self.setContentsMargins(0, 0, 0, 0)
        self.setFrameStyle(self.NoFrame)

        # Layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        # Focus
        self.installEventFilter(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Sizing
        self.n = self.application.config['panel_height']
        self.updateHeight()

        # Drag and Drop Related
        # TODO: only accept drag/drop of other Frames (through MIME data)
        self.setAcceptDrops(True)
        self.displayPanel: DisplayPanel = None
        self.dragStartPos = QtCore.QPoint()
        self.drag = None
        self.resetStyle()

        # Signals
        self.select_me.connect(self.parent().selectFrame,
                               QtCore.Qt.UniqueConnection)
        self.move_me.connect(self.parent().frameToMove)
        self.select_next.connect(self.parent().selectNext)
        self.select_previous.connect(self.parent().selectPrevious)
        self.insert_here.connect(self.parent().whereToInsert)

    def resetStyle(self):
        self.setStyleSheet("""
        Frame {
            border-width: 3px;
            border-color: transparent;
            border-style: solid;
        }
        """)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 400)  # TODO: read from config file?

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1200, 500)  # TODO: read from config file? (One or the other)

    def increaseSize(self, increment: int=50):
        self.n += increment
        self.updateHeight()

    def decreaseSize(self, increment: int=50):
        self.n -= increment
        if self.n < 100:  # TODO: from config?
            self.n = 100
        self.updateHeight()

    def updateHeight(self):
        self.setFixedHeight(self.n)
        self.application.config['panel_height'] = self.n

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.select_me.emit(self)
        return QtWidgets.QWidget.eventFilter(self, source, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragStartPos = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        if (sub(event.pos(), self.dragStartPos)).manhattanLength() < \
                QtWidgets.QApplication.startDragDistance():
            event.ignore()
            return
        self.move_me.emit(self)
        mime_data = QtCore.QMimeData()
        mime_data.setObjectName('frame')
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drop_action = drag.exec_(QtCore.Qt.MoveAction)

    def dropEvent(self, event: QtGui.QDropEvent):
        if event.mimeData().objectName() == 'frame':
            self.resetStyle()
            self.insert_here.emit(self)
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().objectName() == 'frame':
            event.accept()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self.resetStyle()
        event.accept()


class DisplayPanel(QtWidgets.QWidget):
    select_me = Signal(QtWidgets.QFrame, name='select_me')
    add_new_view = Signal(name='add_new_view')
    hideView = Signal(View, name='hideView')
    showView = Signal(View, int, name='showView')
    selectionChanged = Signal(View, name='selectionChanged')
    rendererChanged = Signal(View, name='updateView')
    changeColor = Signal(View, name='changeColor')
    plotViewObj = Signal(View, name='plotViewObj')
    viewMoved = Signal(int, name='viewMoved')

    def __init__(self,
                 frame: Frame):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)
        # Placeholder for item reference
        self.panel: Optional[Panel] = None
        self.setParent(frame)
        self.select_me.connect(frame.select_me)
        self.main_window = self.parent().parent()  # can't static type this because can't import viewer.py (circular)
        self.view_table = ViewTable(self, self.main_window.column_width_hint)

        # View Table
        self.view_table.installEventFilter(self)

        # Splitter
        self.pw = DumbPlot(self)
        self.table_splitter = TableSplitter(self)
        self.table_splitter.addWidget(self.pw)
        self.table_splitter.addWidget(self.view_table)
        self.table_splitter.setStretchFactor(0, 1)
        self.table_splitter.setStretchFactor(1, 0)
        self.table_splitter.setCollapsible(0, True)
        self.table_splitter.handle(1).installEventFilter(self.table_splitter)
        self.view_table.setMaximumWidth(self.view_table.viewportSizeHint().width())

        # Plot area
        self.hideView.connect(self.pw.hideView)
        self.showView.connect(self.pw.showView)
        self.selectionChanged.connect(self.pw.selectionChanged)
        self.rendererChanged.connect(self.pw.rendererChanged)
        self.changeColor.connect(self.pw.changeColor)
        self.plotViewObj.connect(self.pw.addView)
        self.viewMoved.connect(self.main_window.viewMoved)
        self.handle = Handle(self, label='')
        layout.addWidget(self.handle)
        layout.addWidget(self.table_splitter)
        self.add_new_view.connect(self.main_window.guiAddView)

    # def play(self):
    #     print('starting play()')
    #     track = self.getCurrentTrack()
    #     if not track:
    #         return
    #     import simpleaudio as sa
    #     # TODO: eventually write a whole QT audio player?
    #     try:
    #         play_obj = sa.play_buffer(track.value, 1, 2, track.fs)
    #         play_obj.wait_done()  # this will block
    #     except Exception:
    #         pass

    def setButtonEnableStatus(self):
        # TODO: reroute to main window method to enable/disable track items
        pass
    #     if self.panel.selected_view:
    #         self.view_control.enableButtonsNeedingView()
    #     else:
    #         self.view_control.disableButtonsNeedingView()

    def loadPanel(self, panel: Panel):
        assert isinstance(panel, Panel)
        self.panel = panel
        self.handle.updateLabel()
        self.view_table.loadPanel(panel)
        self.main_window.evalTrackMenu()

    @Slot(name='setSplitterPosition')
    def setSplitterPosition(self):
        current_sizes = self.table_splitter.sizes()
        if sum(current_sizes) == 0:
            return
        table_width = self.view_table.viewportSizeHint().width()
        self.table_splitter.setSizes([1, table_width])

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) \
            -> QtWidgets.QWidget.eventFilter:
        if event.type() == QtCore.QEvent.FocusIn:
            self.select_me.emit(self.parent())
            self.panel.select_me()
        return QtWidgets.QWidget.eventFilter(self, source, event)

    def createViewWithTrack(self,
                            track,
                            renderer_name: Optional[str]=None,
                            **kwargs):
        if 'renderer' in kwargs.keys():
            renderer_name = kwargs.pop('renderer')
        new_view = self.panel.new_view(track,
                                       renderer_name=renderer_name,
                                       **kwargs)
        self.view_table.addView(new_view)
        self.main_window.evalTrackMenu()
        self.main_window.resetEnabledProcessors()

    def removeViewFromChildren(self, view_to_remove):
        """this method removes the view from the child widgets"""
        self.pw.removeView(view_to_remove)
        self.view_table.delView(view_to_remove)

    def delViewFromModel(self, view_to_remove):
        del_index = self.panel.views.index(view_to_remove)
        self.panel.remove_view(pos=del_index)

    @Slot(View, int, name='moveView')
    def moveView(self, view_to_move: View, panel_index: int):
        assert(isinstance(view_to_move, View))
        self.removeViewFromChildren(view_to_move)
        destination_panel = self.determineDestination(panel_index)
        self.main_window.model.move_view_across_panel(view_to_move,
                                                      destination_panel)
        self.finishViewOperation(view_to_move, panel_index)

    @Slot(View, int, name='linkTrack')
    def linkTrack(self, view_to_link, panel_index):
        self.selectView(view_to_link)
        destination_panel = self.determineDestination(panel_index)
        self.main_window.model.link_track_across_panel(view_to_link,
                                                       destination_panel)
        self.finishViewOperation(view_to_link, panel_index)

    def copyView(self, view_to_copy, panel_index):
        self.selectView(view_to_copy)
        destination_panel = self.determineDestination(panel_index)
        self.main_window.model.copy_view_across_panel(view_to_copy,
                                                      destination_panel)
        self.finishViewOperation(view_to_copy, panel_index)

    def finishViewOperation(self, view, panel_index):
        self.viewMoved.emit(panel_index)
        view_range = view.renderer.vb.viewRange()
        self.main_window.model.panels[panel_index].views[-1].renderer.vb.setRange(xRange=view_range[0],
                                                                                  yRange=view_range[1],
                                                                                  padding=0)
        self.main_window.evalTrackMenu()

    def selectView(self, view):
        self.view_table.selectRow(self.panel.views.index(view))

    def determineDestination(self, panel_index: int) -> Panel:
        if 0 <= panel_index < len(self.main_window.model.panels):
            destination_panel = self.main_window.model.panels[panel_index]
        elif panel_index == -1:
            insert_index = len(self.main_window.model.panels)
            self.main_window.createNewPanel(pos=insert_index)
            destination_panel = self.main_window.model.panels[-1]
            self.main_window.applySync()
        else:
            raise IndexError
        return destination_panel
