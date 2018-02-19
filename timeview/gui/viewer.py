# stdlib
import logging
import sys
import json
import re
from pathlib import Path
from typing import Tuple, List, Optional, DefaultDict, Dict, Union
from collections import defaultdict
from timeit import default_timer as timer

# 3rd party
import numpy as np
# TODO remove pyqt5 dependency (see issue #41)
#  https://github.com/spyder-ide/qtpy/issues/127
from PyQt5 import QtHelp
from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtCore import Slot, Signal
import pyqtgraph as pg

# QtAwesome may give us difficulty in Windows 10 see:
# https://github.com/spyder-ide/spyder/issues/2490
import qtawesome as qta

# local
from .display_panel import DisplayPanel, Frame
from .dialogs import ProcessingDialog, About, HelpBrowser, InfoDialog, RenderDialog
from .view_table import ViewTable
from .model import Model, View, Panel
from .rendering import Partition
from ..dsp import processing, tracking
from ..manager.dataset_manager import ManagerWindow


CONFIG_PATH = Path(__file__).with_name('config.json')
ICON_PATH = Path(__file__).with_name('TimeView.icns')
ICON_COLOR = QtGui.QColor('#00F 897B')
MENU_ICON_COLOR = QtGui.QColor('#00897B')

logger = logging.getLogger()
if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARN)


class Group(QtCore.QObject):
    relay = Signal(name='relay')

    def __init__(self) -> None:
        super().__init__()
        self.views: List[View] = []

    def viewsExcludingSource(self, view_to_exclude):
        return [view for view in set(self.views)
                if view is not view_to_exclude]

    def join(self, view):
        self.views.append(view)
        self.relay.connect(view.renderer.reload)
        if isinstance(view.renderer, Partition):
            view.renderer.item.updatePartitionBoundaries\
                .connect(self.updatePartitionBoundaries)
            view.renderer.item.updatePartitionValuePosition\
                .connect(self.updatePartitionValuePosition)
            view.renderer.item.updatePartitionPosition\
                .connect(self.updatePartitionPosition)
            view.renderer.item.updatePartitionValue\
                .connect(self.updatePartitionValue)
            view.renderer.item.delete_segment\
                .connect(self.removeSegment)
            view.renderer.item.reload.connect(self.relay)

    @Slot(int, float, float, name='updatePartitionBoundaries')
    def updatePartitionBoundaries(self, index, min_bounds, max_bounds):
        view_to_exclude = self.sender().view
        for view in self.viewsExcludingSource(view_to_exclude):
            view.renderer.receivePartitionBoundaries(index, min_bounds,
                                                     max_bounds)

    @Slot(int, float, name='updatePartitionValuePosition')
    def updatePartitionValuePosition(self, index: int, mid_point: float):
        view_to_exclude = self.sender().view
        for view in self.viewsExcludingSource(view_to_exclude):
            view.renderer.receivePartitionValuePosition(index, mid_point)

    @Slot(int, name='updatePartitionPosition')
    def updatePartitionPosition(self, index: int):
        view_to_exclude = self.sender().view
        for view in self.viewsExcludingSource(view_to_exclude):
            view.renderer.receivePartitionPosition(index)

    @Slot(int, name='removeSegment')
    def removeSegment(self, index: int):
        view_to_exclude = self.sender().view
        for view in self.viewsExcludingSource(view_to_exclude):
            view.renderer.receiveRemoveSegment(index)

    @Slot(int, name='updatePartitionValue')
    def updatePartitionValue(self, index: int):
        view_to_exclude = self.sender().view
        for view in self.viewsExcludingSource(view_to_exclude):
            view.renderer.receivePartitionValue(index)


class ScrollArea(QtWidgets.QScrollArea):
    dragEnterSignal = Signal(name='dragEnterSignal')
    dragLeaveSignal = Signal(name='dragLeaveSignal')
    dropSignal = Signal(name='dropSignal')

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParent(parent)
        self.dropSignal.connect(self.parent().moveToEnd)
        self.setWidgetResizable(True)
        self.setAcceptDrops(True)
        self.setContentsMargins(0, 0, 0, 0)

    def dropEvent(self, event: QtGui.QDropEvent):
        self.dropSignal.emit()
        event.accept()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self.dragLeaveSignal.emit()
        event.accept()

    def sizeHint(self):
        return QtCore.QSize(1000, 810)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.type() == QtCore.QEvent.KeyPress:
            event.ignore()


class ScrollAreaWidgetContents(QtWidgets.QWidget):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParent(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.dragStartPos = QtCore.QPoint(0, 0)

    def swapWidgets(self, positions: Tuple[int, int]):
        assert len(positions) == 2
        if positions[0] == positions[1]:
            return
        frame_one = self.layout.takeAt(min(positions)).widget()
        frame_two = self.layout.takeAt(max(positions) - 1).widget()
        self.layout.insertWidget(min(positions), frame_two)
        self.layout.insertWidget(max(positions), frame_one)


class Viewer(QtWidgets.QMainWindow):
    queryAxesWidths = Signal(name='queryAxisWidths')
    queryColWidths = Signal(name='queryColWidths')
    setAxesWidth = Signal(float, name='setAxesWidth')
    setSplitter = Signal(list, name='setSplitter')
    moveSplitterPosition = Signal(name='moveSplitterPosition')
    setColWidths = Signal(list, name='setColWidths')
    refresh = Signal(name='refresh')
    cursorReadoutStatus = Signal(bool, name='cursor_readout_status')

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.manager = ManagerWindow("Dataset Manager", self)
        if ICON_PATH.exists():
            #pix_map_icon = QtGui.QPixmap(str(ICON_PATH), format="PNG")
            #self.setWindowIcon(QtGui.QIcon(pix_map_icon))
            #  this fixes a warning on OSX, but doesn't work at all on windows
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        else:
            logging.warning(f'cannot find icon at {ICON_PATH}')
        self.resize(QtWidgets.QDesktopWidget()
                             .availableGeometry(self).size() * 0.5)  # change this for video capture
        self.model: Model = Model()
        self.processor_action = {QtWidgets.QAction(f"{key}", self): processor()
                                 for key, processor
                                 in sorted(processing.get_processor_classes().items())}

        self.track_menu = None
        self.groups: DefaultDict[int, Group] = defaultdict(Group)
        self.setWindowTitle('TimeView')

        self.scrollArea = ScrollArea(self)
        self.scrollAreaWidgetContents = \
            ScrollAreaWidgetContents(self.scrollArea)
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.synchronized = True
        self.cursor_readout = False

        self.frames: List[Frame] = []
        self.selected_frame: Optional[Frame] = None
        self.moving_frame: Optional[Frame] = None
        self.from_index: Optional[int] = None
        self.to_index: Optional[int] = None

        self.axis_width = 10
        # for storing
        self.column_width_hint: List[int] = []
        self.all_column_widths: List[Dict[ViewTable, int]] = []

        self.reference_plot: Optional[pg.ViewBox] = None
        self.min_plot_width = self.width()

        self.help_window = self.createHelpWindow()
        self.createMenus()

        self.statusBar()
        self.status('Ready')
        self.guiAddPanel()
        self.evalTrackMenu()
        # At the moment, it makes no sense to start the application without at
        # least one panel.  In a later version, it may be possible to save/load
        # the entire set of panels and their views to save state.

    @Slot(name='guiAddPanel')
    def guiAddPanel(self, pos: Optional[int]=None):
        """
        when adding a panel through the gui, this method determines
        where the panel should go, and handles the associated frame selection
        """
        if pos is None:
            if not self.frames:
                pos = 0
            elif self.selected_frame:
                pos = self.frames.index(self.selected_frame) + 1
            else:
                pos = len(self.frames)
        self.createNewPanel(pos=pos)
        self.applySync()
        self.selectFrame(self.frames[pos])

    def createMenus(self):
        menu = self.menuBar()
        # to work around OSX bug that requires switching focus away from this
        # application, and the coming back to it, to make menu accessible this
        # is not necessary when started from the TimeView.app application icon
        if __debug__:  # I am beginning to think that I always want this
            menu.setNativeMenuBar(False)

        # File menu
        file_menu = menu.addMenu('&File')

        exit_action = QtWidgets.QAction('&Exit', self)
        exit_action.triggered.connect(QtWidgets.qApp.quit)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.setMenuRole(QtWidgets.QAction.QuitRole)
        exit_action.setStatusTip('Exit application')
        file_menu.addAction(exit_action)

        # Track Menu
        self.track_menu = menu.addMenu('&Track')

        new_partition_action = QtWidgets.QAction('&New Partition', self)
        new_partition_action.triggered.connect(self.newPartition)
        self.track_menu.addAction(new_partition_action)

        self.track_menu.addSeparator()

        open_action = QtWidgets.QAction("&Open…", self)
        open_action.triggered.connect(self.guiAddView)
        open_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.track_menu.addAction(open_action)

        save_action = QtWidgets.QAction("&Save…", self)
        save_action.triggered.connect(self.guiSaveView)
        save_action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.track_menu.addAction(save_action)

        revert_action = QtWidgets.QAction("&Revert…", self)
        revert_action.triggered.connect(self.guiRevertView)
        revert_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.track_menu.addAction(revert_action)

        self.track_menu.addSeparator()

        remove_action = QtWidgets.QAction("&Delete", self)
        remove_action.triggered.connect(self.guiDelView)
        remove_action.setShortcut(QtGui.QKeySequence("Ctrl+backspace"))
        self.track_menu.addAction(remove_action)

        self.track_menu.addSeparator()

        info_action = QtWidgets.QAction('&Info', self)
        info_action.triggered.connect(self.showInfoDialog)
        info_action.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        self.track_menu.addAction(info_action)

        options_action = QtWidgets.QAction("&Rendering Options", self)
        options_action.triggered.connect(self.showRenderDialog)
        options_action.setShortcut(QtGui.QKeySequence("Ctrl+V"))  # If using macOS menu, can't use this shortcut, QT will assign this to "preferences"
        self.track_menu.addAction(options_action)

        # Panel Menu
        panel_menu = menu.addMenu('&Panel')
        add_action = QtWidgets.QAction('&New Panel',
                                       self)
        add_action.triggered.connect(self.guiAddPanel)
        add_action.setShortcut(QtGui.QKeySequence.New)
        add_action.setStatusTip('Add Panel')
        panel_menu.addAction(add_action)
        remove_action = QtWidgets.QAction("&Close Panel",
                                          self)
        remove_action.triggered.connect(self.delItem)
        remove_action.setShortcut(QtGui.QKeySequence.Close)
        remove_action.setStatusTip('remove panel')
        panel_menu.addAction(remove_action)

        panel_menu.addSeparator()
        move_panel_up = QtWidgets.QAction("Move Up", self)
        move_panel_up.setShortcut(QtGui.QKeySequence("Ctrl+PgUp"))
        move_panel_up.triggered.connect(self.moveUp)
        panel_menu.addAction(move_panel_up)

        move_panel_down = QtWidgets.QAction("Move Down", self)
        move_panel_down.setShortcut(QtGui.QKeySequence("Ctrl+PgDown"))
        move_panel_down.triggered.connect(self.moveDown)
        panel_menu.addAction(move_panel_down)

        panel_menu.addSeparator()
        increase_size_action = QtWidgets.QAction('&Increase Height',
                                                 self)
        increase_size_action.triggered.connect(self.increaseSize)
        increase_size_action.setShortcuts([QtGui.QKeySequence.ZoomIn,
                                           QtGui.QKeySequence("Ctrl+=")])
        increase_size_action.setStatusTip("Increase the vertical size of the "
                                          "currently selected display panel")
        panel_menu.addAction(increase_size_action)

        decrease_size_action = QtWidgets.QAction('&Decrease Height',
                                                 self)
        decrease_size_action.triggered.connect(self.decreaseSize)
        decrease_size_action.setShortcut(QtGui.QKeySequence.ZoomOut)
        decrease_size_action.setStatusTip("Decrease the vertical size of the "
                                          "currently selected display panel")
        panel_menu.addAction(decrease_size_action)
        panel_menu.addSeparator()

        hide_cursor_info_action = QtWidgets.QAction('Show Cursor', self)
        hide_cursor_info_action.triggered.connect(self.toggleCursorReadout)
        hide_cursor_info_action.setCheckable(True)
        hide_cursor_info_action.setChecked(self.cursor_readout)
        hide_cursor_info_action.setStatusTip('Toggle the display of the '
                                             'readout of the cursor position '
                                             'in the plot area')
        panel_menu.addAction(hide_cursor_info_action)

        toggle_xaxis_label_action = QtWidgets.QAction("Show X-Axis Label", self)
        toggle_xaxis_label_action.triggered.connect(self.toggleXAxis)
        toggle_xaxis_label_action.setCheckable(True)
        toggle_xaxis_label_action.setChecked(self.application.config['show_x-axis_label'])
        panel_menu.addAction(toggle_xaxis_label_action)

        synchronize_action = QtWidgets.QAction('Synchronize', self)
        synchronize_action.triggered.connect(self.changeSync)
        synchronize_action.setCheckable(True)
        synchronize_action.setChecked(self.synchronized)
        panel_menu.addAction(synchronize_action)

        # Navigation Menu
        navigation = menu.addMenu('&Navigation')
        # these changes applies to current panel
        # (and thus globally if synchronization is on)
        shift_left_action = QtWidgets.QAction("Move &Left",
                                              self)
        shift_left_action.triggered.connect(self.shiftLeft)
        shift_left_action.setShortcut(QtGui.QKeySequence.MoveToPreviousChar)
        navigation.addAction(shift_left_action)

        move_right_action = QtWidgets.QAction("Move &Right",
                                              self)
        move_right_action.triggered.connect(self.shiftRight)
        move_right_action.setShortcut(QtGui.QKeySequence.MoveToNextChar)
        move_right_action.setStatusTip("Shift plot half a window to the right")
        navigation.addAction(move_right_action)

        goto_start_action = QtWidgets.QAction("Go to &Beginning",
                                              self)
        goto_start_action.triggered.connect(self.goToBeginning)
        goto_start_action.setShortcut(QtGui.QKeySequence("Ctrl+left"))
        navigation.addAction(goto_start_action)

        goto_end_action = QtWidgets.QAction("Go to &End",
                                            self)
        goto_end_action.triggered.connect(self.goToEnd)
        goto_end_action.setShortcut(QtGui.QKeySequence("Ctrl+Right"))
        navigation.addAction(goto_end_action)

        navigation.addSeparator()

        zoom_in_action = QtWidgets.QAction("Zoom &In",
                                           self)
        zoom_in_action.triggered.connect(self.zoomIn)
        zoom_in_action.setShortcut(QtGui.QKeySequence("Up"))
        navigation.addAction(zoom_in_action)
        zoom_out_action = QtWidgets.QAction("Zoom &Out",
                                            self)
        zoom_out_action.triggered.connect(self.zoomOut)  # no overlap
        zoom_out_action.setShortcut(QtGui.QKeySequence("Down"))
        navigation.addAction(zoom_out_action)

        zoom_to_match_action =\
            QtWidgets.QAction("Zoom to &1:1",
                              self)
        zoom_to_match_action.setShortcut(QtGui.QKeySequence
                                         .MoveToStartOfDocument)
        zoom_to_match_action.triggered.connect(self.zoomToMatch)
        navigation.addAction(zoom_to_match_action)

        zoom_fit_action = QtWidgets.QAction("Zoom to &Fit",
                                            self)
        zoom_fit_action.setShortcut(QtGui.QKeySequence.MoveToEndOfDocument)
        zoom_fit_action.triggered.connect(self.zoomFit)  # show all
        navigation.addAction(zoom_fit_action)

        # processing Menu
        processing_menu = menu.addMenu('&Processing')
        for processor_action, processor in self.processor_action.items():
            processor_action.triggered.connect(self.showProcessorDialog)
            processor_action.setEnabled(False)
            processing_menu.addAction(processor_action)

        # Window Menu
        window_menu = menu.addMenu('&Window')
        manager_action = QtWidgets.QAction('&Dataset Manager', self)
        manager_action.triggered.connect(self.manager.show)
        window_menu.addAction(manager_action)

        # Help Menu
        help_menu = menu.addMenu('&Help')
        help_action = QtWidgets.QAction("&TimeView Help",
                                        self)
        help_action.setShortcut(QtGui.QKeySequence.HelpContents)
        help_action.triggered.connect(self.help_window.show)
        help_action.setStatusTip('show help')
        help_menu.addAction(help_action)
        about_action = QtWidgets.QAction("&About",
                                         self)
        about_action.triggered.connect(self.showAbout)
        about_action.setMenuRole(QtWidgets.QAction.AboutRole)
        help_menu.addAction(about_action)

    def resetEnabledProcessors(self):
        for action, processor in self.processor_action.items():
            action.setEnabled(False)
            for track_name, track_type in processor.acquire.items():
                tracks = [view.track
                          for panel in self.model.panels
                          for view in panel.views
                          if isinstance(view.track, track_type)]
                if not tracks:
                    break
            else:
                action.setEnabled(True)

    def toggleCursorReadout(self):
        self.cursor_readout = not self.cursor_readout
        self.cursorReadoutStatus.emit(self.cursor_readout)

    def newPartition(self):
        new_track =\
            tracking.Partition(np.array([0.,
                                         48000.]).astype(tracking.TIME_TYPE),
                               np.array(['']).astype('U32'),
                               48000)
        self.selectedDisplayPanel.createViewWithTrack(new_track,
                                                      renderer='Partition (editable)')

    def getSelectedDisplayPanel(self) -> DisplayPanel:
        selected_index = self.model.panels.index(self.model.selected_panel)
        return self.frames[selected_index].displayPanel

    selectedDisplayPanel = property(getSelectedDisplayPanel)

    def getSelectedTrack(self) -> tracking.Track:
        panel = self.selectedPanel
        track = panel.selected_track()
        return track

    selectedTrack = property(getSelectedTrack)

    def getSelectedView(self) -> View:
        return self.selectedPanel.selected_view

    selectedView = property(getSelectedView)

    def viewRange(self, display_panel=None) -> Tuple[float, float]:
        if display_panel is None:
            display_panel = self.selectedDisplayPanel
            if display_panel is None:
                return 0., 1.
        view = display_panel.panel.selected_view
        if view is None:
            vb = display_panel.pw.main_vb
        else:
            vb = view.renderer.vb
        return vb.viewRange()[0]

    # TODO: when shifting by menu / keys, implement a *target* system,
    # where we are smoothly and exponentially scrolling to the desired target
    @Slot(name='pageRight')
    def pageRight(self):
        span = np.diff(self.viewRange())[0]
        self.translateBy(span)

    def translateBy(self, delta_x):
        view = self.selectedView
        if view is None:
            return
        x_min, x_max = view.renderer.vb.viewRange()[0]
        if x_min < 0 and delta_x < 0:
            return
        self.applySync()
        view.renderer.vb.translateBy(x=delta_x)
        self.selectedDisplayPanel.pw.alignViews()
        if self.synchronized:
            reference_view_range = self.reference_plot.viewRange()[0]
            for frame in self.frames:
                frame_view_range = frame.displayPanel.pw.main_vb.viewRange()[0]
                assert reference_view_range == frame_view_range

    def scaleBy(self, mag_x):
        view = self.selectedView
        if view is None:
            return
        self.applySync()
        center = view.renderer.vb.targetRect().center()
        padding = view.renderer.vb.suggestPadding(pg.ViewBox.XAxis)
        proposed_ranges = [dim * mag_x for dim in view.renderer.vb.viewRange()[0]]
        if proposed_ranges[0] < -padding:
            shift_right = abs(proposed_ranges[0]) - padding
            center.setX(center.x() + shift_right)
        view.renderer.vb.scaleBy(x=mag_x, center=center)
        self.selectedDisplayPanel.pw.alignViews()
        if self.synchronized:
            reference_view_range = self.reference_plot.viewRange()[0]
            assert all([reference_view_range == frame.displayPanel.pw.main_vb.viewRange()[0]
                        for frame in self.frames])

    def getSelectedPanel(self) -> Panel:
        return self.model.selected_panel

    def setSelectedPanel(self, panel: Panel):
        self.model.set_selected_panel(panel)

    selectedPanel = property(getSelectedPanel, setSelectedPanel)

    @Slot(name='pageLeft')
    def pageLeft(self):
        span = np.diff(self.viewRange())[0]
        self.translateBy(-span)

    @Slot(name='shiftRight')
    def shiftRight(self):
        span = np.diff(self.viewRange())[0]
        shift = span / 10
        self.translateBy(shift)

    @Slot(name='shiftLeft')
    def shiftLeft(self):
        vb = self.selectedPanel.selected_view.renderer.vb
        x_min, x_max = vb.viewRange()[0]
        padding = vb.suggestPadding(pg.ViewBox.XAxis)
        span = x_max - x_min
        shift = span / 10
        if x_min < 0:
            return
        elif x_min - shift < -padding:
            shift = max(x_min, padding)
        self.translateBy(-shift)

    @Slot(name='goToBeginning')
    def goToBeginning(self):
        x_min, x_max = self.viewRange()
        padding = self.selectedPanel.selected_view.renderer.vb.suggestPadding(1)
        self.translateBy(-x_min - padding)

    @Slot(name='goToEnd')
    def goToEnd(self):
        x_min, x_max = self.viewRange()
        view = self.selectedView
        if view is None:
            return
        track = view.track
        end_time = view.track.duration / view.track.fs
        self.translateBy(end_time - x_max)

    @Slot(name='zoomFit')
    def zoomFit(self):
        view = self.selectedView
        if view is None:
            return
        track = view.track
        max_t = track.duration / track.fs
        span = np.diff(view.renderer.vb.viewRange()[0])[0]
        self.scaleBy(max_t / span)
        self.goToBeginning()

    @Slot(name='zoomToMatch')
    def zoomToMatch(self):
        """
        where each pixel represents exactly one sample at the
        highest-available sampling-frequency
        :return:
        """
        view = self.selectedPanel.selected_view
        if view is None:
            return
        vb = view.renderer.vb
        pixels = vb.screenGeometry().width()
        mag_span = pixels / self.selectedTrack.fs
        span = np.diff(self.viewRange())[0]
        mag = mag_span / span
        self.scaleBy(mag)

    @Slot(name='zoomIn')
    def zoomIn(self):
        view = self.selectedPanel.selected_view
        if view is None:
            return
        vb = view.renderer.vb
        x_range = np.diff(vb.viewRange()[0])
        minXRange = vb.getState()['limits']['xRange'][0]
        zoom = 0.9
        if minXRange / x_range < zoom:
            pass
        elif zoom < minXRange / x_range <= 1.0:
            zoom = minXRange / x_range
        else:
            return
        self.scaleBy(zoom)

    @Slot(name='zoomOut')
    def zoomOut(self):
        self.scaleBy(1.1)

    @Slot(name='increaseSize')
    def increaseSize(self):
        self.selected_frame.increaseSize()

    @Slot(name='decreaseSize')
    def decreaseSize(self):
        self.selected_frame.decreaseSize()

    @Slot(name='showInfoDialog')
    def showInfoDialog(self):
        info_dialog = InfoDialog(str(self.selectedTrack))
        info_dialog.exec_()

    @Slot(name='showProcessorDialog')
    def showProcessorDialog(self):
        processor = self.processor_action[self.sender()]
        processing_dialog = ProcessingDialog(self, processor)
        processing_dialog.show()

    @Slot(tuple, name='finishedProcessing')
    def insert_processed_tracks(self, new_tracks: List[processing.Tracks]):
        for new_track in new_tracks:
            self.getSelectedDisplayPanel().createViewWithTrack(new_track)

    def status(self, msg: str, timeout: int=3000):
        self.statusBar().showMessage(msg, timeout)

    def joinGroup(self, view):
        group = self.groups[id(view.track)]
        group.join(view)

    def changeSync(self):
        self.synchronized = not self.synchronized
        self.reference_plot = self.selectedDisplayPanel.pw.main_vb
        self.applySync()

    def applySync(self):
        if self.synchronized:
            self.synchronize()
        else:
            self.desynchronize()

    def synchronize(self):
        self.reference_plot = self.selectedDisplayPanel.pw.main_vb
        assert isinstance(self.reference_plot, pg.ViewBox)
        x_min, x_max = self.reference_plot.viewRange()[0]
        for frame in self.frames:
            if frame.displayPanel.pw.main_vb is self.reference_plot:
                continue
            frame.displayPanel.pw.main_vb.setXLink(self.reference_plot)
            if frame.displayPanel.panel.selected_view:
                frame.displayPanel.panel.selected_view.renderer.vb.setXRange(x_min, x_max, padding=0)

    def desynchronize(self):
        self.reference_plot = None
        for frame in self.frames:
            frame.displayPanel.pw.main_vb.setXLink(frame.displayPanel.pw.main_vb)

    def toggleXAxis(self):
        self.application.config['show_x-axis_label'] = not self.application.config['show_x-axis_label']
        for frame in self.frames:
            frame.displayPanel.pw.axis_bottom.showLabel(self.application.config['show_x-axis_label'])

    def createNewPanel(self, pos=None):
        frame = Frame(main_window=self)
        w = DisplayPanel(frame=frame)
        w.pw.setAxesWidths(self.axis_width)
        self.queryAxesWidths.connect(w.pw.updateWidestAxis)
        self.setAxesWidth.connect(w.pw.setAxesWidths)
        self.moveSplitterPosition.connect(w.setSplitterPosition)
        self.setSplitter.connect(w.table_splitter.setSizes_)
        self.setColWidths.connect(w.view_table.setColumnWidths)
        self.queryColWidths.connect(w.view_table.calcColumnWidths)

        w.table_splitter.setSizes([1, w.view_table.viewportSizeHint().width()])
        frame.layout.addWidget(w)
        frame.displayPanel = w
        if pos is not None:
            insert_index = pos
        elif self.selected_frame:
            insert_index = self.frames.index(self.selected_frame) + 1
        else:
            insert_index = None
        panel = self.model.new_panel(pos=insert_index)
        w.loadPanel(panel)
        self.addFrame(frame, insert_index)
        self.applySync()

    def delItem(self):
        if self.selected_frame is None:
            logging.debug('no frame is selected for debug')
            return
        remove_index = self.frames.index(self.selected_frame)
        self.model.remove_panel(remove_index)
        self.removeFrame(self.selected_frame)
        if not self.frames:
            self.selected_frame = None
            self.reference_plot = None
            self.guiAddPanel()
            self.selectFrame(self.frames[-1])
        elif remove_index == len(self.frames):
            self.selectFrame(self.frames[-1])
        else:
            self.selectFrame(self.frames[remove_index])
        self.applySync()

    @Slot(int, name='viewMoved')
    def viewMoved(self, panel_index):
        view_to_add = self.model.panels[panel_index].views[-1]
        self.frames[panel_index].displayPanel.view_table.addView(view_to_add,
                                                                 setColor=False)

    def addFrame(self, frame: Frame, index=None):
        if not index:
            index = len(self.frames)
        self.frames.insert(index, frame)
        self.scrollAreaWidgetContents.layout.insertWidget(index, frame)
        self.updateFrames()

    def removeFrame(self, frame_to_remove: Frame):
        if frame_to_remove.displayPanel.pw.main_vb is self.reference_plot:
            self.reference_plot = None
        self.frames.remove(frame_to_remove)
        self.scrollAreaWidgetContents.layout.removeWidget(frame_to_remove)
        frame_to_remove.deleteLater()
        self.updateFrames()

    def updateFrames(self):
        self.scrollArea.updateGeometry()
        for panel, frame in zip(self.model.panels, self.frames):
            frame.displayPanel.handle.updateLabel()
            assert frame.displayPanel.panel == panel

    def swapFrames(self, positions: Tuple[int, int]):
        self.scrollAreaWidgetContents.swapWidgets(positions)
        self.frames[positions[0]], self.frames[positions[1]] = \
            self.frames[positions[1]], self.frames[positions[0]]
        self.model.panels[positions[0]], self.model.panels[positions[1]] =\
            self.model.panels[positions[1]], self.model.panels[positions[0]]
        self.updateFrames()

    @Slot(list, name='determineColumnWidths')
    def determineColumnWidths(self, widths: List[int]):
        if not self.all_column_widths:
            self.all_column_widths = [{self.sender(): width} for width in widths]
        else:
            for index, width in enumerate(widths):
                self.all_column_widths[index][self.sender()] = width

        self.column_width_hint = [max(column.values())
                                  for column in self.all_column_widths]
        self.setColWidths.emit(self.column_width_hint)
        self.moveSplitterPosition.emit()

    @Slot(name='moveUp')
    def moveUp(self):
        index = self.frames.index(self.selected_frame)
        if index == 0:
            return
        self.swapFrames((index, index - 1))

    @Slot(name='moveDown')
    def moveDown(self):
        index = self.frames.index(self.selected_frame)
        if index == len(self.frames) - 1:
            return
        self.swapFrames((index, index + 1))

    @Slot(name='selectNext')
    def selectNext(self):
        index = self.frames.index(self.selected_frame)
        if index == len(self.frames) - 1:
            return
        else:
            self.selectFrame(self.frames[index + 1])

    @Slot(name='selectPrevious')
    def selectPrevious(self):
        index = self.frames.index(self.selected_frame)
        if index == 0:
            return
        else:
            self.selectFrame(self.frames[index - 1])

    @Slot(QtWidgets.QFrame, name='selectFrame')
    def selectFrame(self, frame_to_select: Frame):
        assert isinstance(frame_to_select, Frame)
        assert frame_to_select in self.frames
        if self.selected_frame is not None:
            self.selected_frame.resetStyle()
        self.selected_frame = frame_to_select
        self.selected_frame.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.selected_frame.setStyleSheet("""
        Frame {
            border: 3px solid red;
        }
        """)
        index = self.frames.index(self.selected_frame)
        self.model.set_selected_panel(self.model.panels[index])
        if self.synchronized and self.reference_plot is None:
            self.reference_plot = self.selectedDisplayPanel.pw.main_vb
        self.evalTrackMenu()
        selected_frame_index = self.frames.index(frame_to_select)
        selected_panel_index = self.model.panels.index(self.selectedPanel)
        assert selected_frame_index == selected_panel_index

    @Slot(QtWidgets.QFrame, name='frameToMove')
    def frameToMove(self, frame_to_move: Frame):
        self.from_index = self.frames.index(frame_to_move)

    @Slot(QtWidgets.QFrame, name='whereToInsert')
    def whereToInsert(self, insert_here: Frame):
        self.to_index = self.frames.index(insert_here)
        if self.to_index == self.from_index:
            self.from_index = self.to_index = None
            return
        self.moveFrame()

    def moveFrame(self):
        if self.to_index is None or self.from_index is None:
            logging.debug('To and/or From index not set properly')
            return
        frame = self.frames[self.from_index]
        self.scrollAreaWidgetContents.layout.removeWidget(frame)
        self.frames.insert(self.to_index, self.frames.pop(self.from_index))
        self.model.move_panel(self.to_index, self.from_index)
        self.scrollAreaWidgetContents.layout.insertWidget(self.to_index,
                                                          frame)
        self.selectFrame(self.frames[self.to_index])
        self.updateFrames()
        # Resetting moving parameters
        self.from_index = self.to_index = None

    @Slot(name='moveToEnd')
    def moveToEnd(self):
        self.frameToMove(self.selected_frame)
        self.to_index = len(self.frames) - 1
        self.moveFrame()

    @Slot(name='checkAxesWidths')
    def checkAxesWidths(self):
        widths = [axis.preferredWidth()
                  for frame in self.frames
                  for axis in frame.displayPanel.pw.axes.values()]
        if not widths:
            return
        axis_width = max(widths)
        if axis_width != self.axis_width:
            self.axis_width = axis_width
            self.setAxesWidth.emit(self.axis_width)

    @staticmethod
    def showAbout():
        about_box = About()
        about_box.exec_()

    def createHelpWindow(self):
        # http://www.walletfox.com/course/qhelpengineexample.php
        help_path = (Path(__file__).parent / 'TimeView.qhc').resolve()
        assert help_path.exists()
        help_engine = QtHelp.QHelpEngine(str(help_path))
        help_engine.setupData()

        tab_widget = QtWidgets.QTabWidget()
        tab_widget.setMaximumWidth(400)
        tab_widget.addTab(help_engine.contentWidget(), "Contents")
        tab_widget.addTab(help_engine.indexWidget(), "Index")

        text_viewer = HelpBrowser(help_engine)
        url = "qthelp://org.sphinx.timeview.1.0/doc/index.html"
        text_viewer.setSource(QtCore.QUrl(url))

        help_engine.contentWidget()\
                   .linkActivated['QUrl'].connect(text_viewer.setSource)
        help_engine.indexWidget()\
                   .linkActivated['QUrl', str].connect(text_viewer.setSource)

        horiz_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        horiz_splitter.insertWidget(0, tab_widget)
        horiz_splitter.insertWidget(1, text_viewer)

        help_window = QtWidgets.QDockWidget('Help', self)
        help_window.setWidget(horiz_splitter)
        help_window.hide()
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, help_window)
        return help_window

    @Slot(name='guiAddView')
    def guiAddView(self,
                   file_name: Union[str, List, Path, None]=None,
                   renderer: Optional[str]=None,
                   **kwargs):
        if file_name is None:
            file_name, _ =\
                QtWidgets.QFileDialog.getOpenFileNames(self,
                                                      "Add Track to Panel",
                                                      self.application.config['working_directory'],
                                                      "Track and EDF Files (*.wav *.lab *.tmv *.edf);;\
                                                      All Files (*)",
                                                      options=QtWidgets.QFileDialog.Options())
        if isinstance(file_name, str):
            # panel_index = self.model.panels.index(self.selectedPanel)
            self.application.add_view_from_file(Path(file_name))
            self.application.config['working_directory'] = str(Path(file_name).parent)
        elif isinstance(file_name, List):
            if len(file_name):
                for f in file_name:
                    #self.guiAddPanel()  # it's difficult to guess what the user really wants
                    self.application.add_view_from_file(Path(f))
                self.application.config['working_directory'] = str(Path(f).parent)
        else:
            raise Exception

    @Slot(name='guiSaveView')
    def guiSaveView(self):
        """identifies the selected view and removes it"""
        view = self.selectedView
        if view is None:
            return
        track = view.track
        file_name, _ = \
            QtWidgets.QFileDialog.getSaveFileName(self,
                                                  "Save Track",
                                                  str(Path(self.application.config['working_directory']) / track.path.name),
                                                  f"Files (*{track.default_suffix})")
        if file_name:
            track.write(file_name)
            track.path = Path(file_name)
            # TODO: update view table to show you name
            self.application.config['working_directory'] = str(Path(file_name).parent)

    @Slot(name='guiRevertView')
    def guiRevertView(self):
        """identifies the selected view and reverts to contents on disk"""
        view = self.selectedView
        if view is None:
            return
        reply = QtWidgets.QMessageBox.question(self,
                                               'Message',
                                               'Are you sure you want to revert to the contents on disk? All changes since loading will be lost',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.Cancel)
        if reply == QtWidgets.QMessageBox.Yes:
            track_path = str(view.track.path)
            view.track = view.track.read(view.track.path)
            new_track = view.track.read(view.track.path)
            view.track = new_track
            view.renderer.track = new_track  # TODO: change to render property that pulls the track from the view
            view.renderer.reload()


    @Slot(name='guiDelView')
    def guiDelView(self):
        """identifies the selected view and removes it"""
        if self.selectedView is None:
            return
        view_to_remove = self.selectedView
        self.selectedDisplayPanel.removeViewFromChildren(view_to_remove)
        self.selectedDisplayPanel.delViewFromModel(view_to_remove)
        self.evalTrackMenu()

    def showRenderDialog(self):
        renderer = self.selectedView.renderer
        if not renderer.parameters:
            logger.info('No parameters to modify for given renderer')
            return
        render_dialog = RenderDialog(self, renderer)
        render_dialog.exec_()
        if not render_dialog.result():  # undo changes if cancel is pressed
            return

    def setTrackMenuStatus(self, enabled):
        ignore_actions = ["New Partition", "Open"] # TODO: hate this...
        for action in self.track_menu.actions():
            if any([ignore_str in action.text() for ignore_str in ignore_actions]):
                continue
            else:
                action.setEnabled(enabled)

    def evalTrackMenu(self):
        self.setTrackMenuStatus(bool(self.selectedPanel.views))


class TimeView(object):  # Application - here's still the best place for it methinks
    def __init__(self):
        start = timer()
        sys.argv[0] = 'TimeView'  # to override Application menu on OSX
        QtCore.qInstallMessageHandler(self._log_handler)
        QtWidgets.QApplication.setDesktopSettingsAware(False)
        self.qtapp = qtapp = QtWidgets.QApplication(sys.argv)
        qtapp.setStyle("fusion")
        qtapp.setApplicationName("TimeView")
        qtapp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        if hasattr(QtWidgets.QStyleFactory, 'AA_UseHighDpiPixmaps'):
            qtapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

        self.config = {'working_directory': str(Path.home()),
                       'panel_height': 300,
                       'show_x-axis_label': True}
        try:
            with open(CONFIG_PATH) as file:
                self.config.update(json.load(file))
        except IOError:
            logging.debug('cannot find saved configuration, using default configuration')

        self.viewer = Viewer(self)
        # audio player here?

        if not __debug__:
            sys.excepthook = self._excepthook

        finish = timer()
        logging.debug(f'complete startup time is {finish-start:{0}.{3}} seconds')

    @staticmethod
    def _log_handler(msg_type, msg_log_context, msg_string):
        if msg_type == 1:
            if re.match("QGridLayoutEngine::addItem: Cell \\(\\d+, \\d+\\) already taken", msg_string):
                return
            logger.warning(msg_string)
        elif msg_type == 2:
            logger.critical(msg_string)
        elif msg_type == 3:
            logger.error(msg_string)
        elif msg_type == 4:
            logger.info(msg_string)
        elif msg_type == 0:
            logger.debug(msg_string)
        else:
            logger.warning(f'received unknown message type from qt system with contents {msg_string}')

    def _excepthook(self, exc_type, exc_value, exc_traceback):
        logging.exception('Uncaught Exception', exc_info=(exc_type, exc_value, exc_traceback))
        from .dialogs import Bug
        bug_box = Bug(self.qtapp, exc_type, exc_value, exc_traceback)
        bug_box.exec_()

    def start(self):
        self.viewer.show()
        self._exit(self.qtapp.exec_())

    def _exit(self, status):
        with open(CONFIG_PATH, 'w') as file:
            json.dump(self.config, file)
        del self.viewer
        del self.qtapp
        sys.exit(status)

    def add_view(self,
                 track_obj: tracking.Track,
                 panel_index: int=None,
                 renderer_name: Optional[str]=None,
                 *args,
                 **kwargs):
        if isinstance(panel_index, int) and \
                        panel_index >= len(self.viewer.frames):
            for pos in range(len(self.viewer.frames), panel_index + 1):
                self.viewer.guiAddPanel()
                self.viewer.selectFrame(self.viewer.frames[pos])
        self.viewer.selectedDisplayPanel.createViewWithTrack(track_obj,
                                                             renderer_name,
                                                             **kwargs)

    def add_view_from_file(self, file: Path, panel_index: int=None):
        if file.suffix == '.edf':
            import pyedflib
            with pyedflib.EdfReader(str(file)) as f:
                labels = f.getSignalLabels()
                for label in labels:
                    index = labels.index(label)
                    wav = tracking.Wave(f.readSignal(index), f.getSampleFrequency(index))
                    wav.label = label
                    wav.path = file.with_name(file.stem + '-' + label + '.wav')
                    wav.min = f.getPhysicalMinimum(index)
                    wav.max = f.getPhysicalMaximum(index)
                    wav.unit = f.getPhysicalDimension(index)
                    self.add_view(wav, panel_index=panel_index, y_min=wav.min, y_max=wav.max)
        else:
            try:
                track_obj = tracking.Track.read(file)
            except Exception as e:
                logging.exception(e)
            else:
                self.add_view(track_obj, panel_index=panel_index)
