import logging
from typing import Dict, Optional
from math import ceil
import sys

from qtpy import QtGui, QtWidgets, QtCore
from qtpy.QtCore import Slot, Signal
import pyqtgraph as pg

from .model import View
from . import rendering

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

icon_color = QtGui.QColor('#00897B')


class DumbPlot(pg.GraphicsView):
    maxWidthChanged = Signal(name='maxWidthChanged')

    def __init__(self, display_panel):
        super().__init__()
        self.display_panel = display_panel
        self.main_window = self.display_panel.main_window
        # Layout
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.layout = pg.GraphicsLayout()
        self.layout.layout.setColumnFixedWidth(0,
                                               self.main_window.axis_width)

        # Variables
        self.axes: Optional[Dict[View, pg.AxisItem]] = {}
        self.vbs: Optional[Dict[View, pg.ViewBox]] = {}

        self.main_plot = pg.PlotItem(enableMenu=False)
        self.main_plot.hideButtons()
        self.main_plot.hideAxis('left')
        self.main_plot.hideAxis('bottom')
        self.main_vb: pg.ViewBox = self.main_plot.getViewBox()
        self.main_vb.sigXRangeChanged.connect(self.zoomChanged)
        self.main_vb.setXRange(0, 1)
        self.main_vb.setYRange(0, 1)
        self.main_vb.setMouseEnabled(False, False)
        self.main_plot.setZValue(-sys.maxsize - 1)
        self.axis_bottom = pg.AxisItem('bottom', parent=self.main_vb)
        self.axis_bottom.setLabel('time (s)')
        self.axis_bottom.showLabel(self.main_window.application.config['show_x-axis_label'])
        # self.axis_bottom.setFixedHeight(self.axis_bottom.height())
        self.label = pg.LabelItem(justify='left', color=[255, 255, 255, 0])

        # Connections
        self.maxWidthChanged.connect(self.main_window.checkAxesWidths)
        self.main_vb.sigResized.connect(self.updateViews)
        self.main_window.cursorReadoutStatus.connect(self.setCursorReadout)

        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved,
                                    rateLimit=60,
                                    slot=self.mouseMoved)
        self.buildLayout()
        self.setCursorReadout(self.main_window.cursor_readout)

    def buildLayout(self):
        self.setCentralWidget(self.layout)
        self.layout.addItem(self.main_plot, row=0, col=1)
        self.layout.addItem(self.label, row=0, col=1)
        self.layout.addItem(self.axis_bottom, row=1, col=1)
        self.axis_bottom.linkToView(self.main_vb)
        self.layout.scene().sigMouseClicked.connect(self.onClick,
                                                    QtCore.Qt.UniqueConnection)
        self.layout.layout.setRowStretchFactor(0, 1)
        self.layout.layout.setColumnStretchFactor(1, 1)
        self.layout.update()
        self.axis_bottom.hide()


    def wheelEvent(self, event: QtGui.QWheelEvent):
        super().wheelEvent(event)
        event.accept()

    def mouseMoved(self, event):
        if self.selected_view() not in self.vbs.keys():
            return
        vb = self.vbs[self.selected_view()]
        pos = event[0]  # using signal proxy turns original arguments into a tuple
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mousePoint = vb.mapSceneToView(pos)
        time = f"{mousePoint.x():12.8}"
        sample = int(ceil(mousePoint.x() * self.selected_view().track.fs))
        y = f"{mousePoint.y():+12.8}"  # always show sign
        self.label.setText(
            "<span style=color: white> "  # TODO: is this doing anything?
            f"t = {time}<br /> "
            f"x = {sample}<br />"
            f"y = {y}"
            f"</span>")

    @Slot(bool, name='Set Cursor Readout')
    def setCursorReadout(self, enabled):
        if enabled:
            self.label.show()
        else:
            self.label.hide()
        self.layout.update()

    def onClick(self, event):
        if event.double():
            items = self.layout.scene().items(event.scenePos())
            obj = items[0]

            # double clicked on infinite-line implies remove
            if isinstance(obj, pg.InfiniteLine):
                obj.parent.removeSegment(obj, modify_track=True)
                event.accept()
                return True
            # double click on view box implies add partition
            elif isinstance(obj, pg.ViewBox):
                partitions = [view.renderer for view in self.vbs.keys()
                              if isinstance(view.renderer,
                                            rendering.PartitionEdit)]
                # no partitions found, exiting
                if not partitions:
                    event.ignore()
                    return False
                x = self.main_vb.mapFromItemToView(self.main_vb,
                                                   event.pos()).x()
                # determine which partition view to add new partition to
                selected_view = self.selected_view()
                if len(partitions) == 1:
                    partitions[0].insertSegment(x)
                    event.accept()
                    return True
                elif isinstance(selected_view.renderer,
                                rendering.PartitionEdit):
                    selected_view.renderer.insertSegment(x)
                    event.accept()
                    return True
                else:
                    logger.info("Multiple Partition Views detected, "
                                "none of which are selected.\n"
                                "select the view you wish to insert the "
                                "partition into")
                    event.ignore()
                    return False

        event.ignore()

    @Slot(View, name='rendererChanged')
    def rendererChanged(self, view: View):
        self.removeView(view)
        self.addView(view, forceRangeReset=False)

    @Slot(View, name='addView')
    def addView(self, view: View, forceRangeReset=None):
        logging.debug(f'Adding {view.renderer.name}')
        if forceRangeReset is not None:
            rangeReset = forceRangeReset
        else:
            if len(self.main_window.model.panels) == 1:
                rangeReset = not(bool(self.vbs))
            else:
                rangeReset = False
        ax, vb = view.renderer.render(self)
        self.main_window.joinGroup(view)
        self.axes[view] = ax
        self.vbs[view] = vb
        self.updateWidestAxis()
        self.updateViews()
        if view.show is False:
            self.hideView(view)
        self.axis_bottom.show()
        # this fixes the bottom axis mirroring on macOS
        old_size = self.size()
        self.resize(QtCore.QSize(old_size.width()+1, old_size.height()))
        self.resize(old_size)

        self.layout.update()
        if isinstance(view.renderer, rendering.Spectrogram):
            view.renderer.generatePlotData()
        if rangeReset:
            self.main_window.zoomFit()

    # TODO: problem, this method is called after a view_to_remove.renderer has
    # already changed in view_table.changeRenderer
    # this should be refactored, so this method here can call
    # view_to_remove.renderer.prepareToDelete() if such a method exists
    def removeView(self, view_to_remove: View):
        axis_to_remove = self.axes.pop(view_to_remove)
        vb_to_remove = self.vbs.pop(view_to_remove)
        assert isinstance(vb_to_remove, pg.ViewBox)
        assert isinstance(axis_to_remove, pg.AxisItem)
        self.layout.removeItem(vb_to_remove)
        del vb_to_remove
        if axis_to_remove in self.layout.childItems():
            view_to_remove.is_selected()
            self.layout.removeItem(axis_to_remove)
        if not self.axes:
            self.layout.layout.setColumnFixedWidth(0,
                                                   self.main_window.axis_width)
        self.updateViews()
        self.updateWidestAxis()
        if not self.vbs:
            self.axis_bottom.hide()
        self.layout.update()

    def hideView(self, view_to_hide: View):
        self.vbs[view_to_hide].setXLink(None)
        self.vbs[view_to_hide].hide()
        axis = self.axes[view_to_hide]
        width = axis.width()
        axis.showLabel(show=False)
        axis.setStyle(showValues=False)
        axis.setWidth(w=width)
        self.main_vb.setFixedWidth(self.vbs[view_to_hide].width())

    def showView(self, view_to_show: View):
        self.axes[view_to_show].showLabel(show=True)
        if not isinstance(view_to_show.renderer, rendering.Partition):
            self.axes[view_to_show].setStyle(showValues=True)
        self.vbs[view_to_show].setXLink(self.main_vb)
        self.vbs[view_to_show].show()
        self.updateViews()

    @Slot(View, name='changeColor')
    def changeColor(self, view_object: View):
        view_object.renderer.changePen()

    @Slot(name='Align Views')
    def alignViews(self):
        x_min, x_max = self.selected_view().renderer.vb.viewRange()[0]
        for view, vb in self.vbs.items():
            if view.is_selected():
                continue
            vb.setXRange(x_min, x_max, padding=0)
        self.axis_bottom.setRange(x_min, x_max)

    @Slot(name='updateViews')
    def updateViews(self):
        if self.selected_view() is None \
                or not self.main_vb.width() \
                or not self.main_vb.height():
            return
        track = self.selected_view().track
        # termining max zoom
        minXRange = self.main_vb.screenGeometry().width() / track.fs
        x_min, x_max = self.main_vb.viewRange()[0]
        for view, view_box in self.vbs.items():
            view_box.blockSignals(True)
            if view_box.geometry() != self.main_vb.sceneBoundingRect():
                view_box.setGeometry(self.main_vb.sceneBoundingRect())
            view_box.setLimits(minXRange=minXRange) # applying max zoom
            view_box.setXRange(x_min, x_max, padding=0)
            view_box.blockSignals(False)
        self.axis_bottom.setRange(x_min, x_max)

    def zoomChanged(self):
        if self.main_vb.geometry():
            try:
                pixel_width = self.main_vb.viewPixelSize()[0]
                self.main_vb.setLimits(xMin=-pixel_width)
                for vb in self.vbs.values():
                    vb.setLimits(xMin=-pixel_width)
            except Exception as e:
                logger.exception('Why is this happening?')
                # has to do with the viewbox geometry not being rendered
                # and i'm asking for the pixel width
                # my if condition was wrong (now fixed)

    @Slot(View, name='selectionChanged')
    def selectionChanged(self, selected_view: View):
        assert selected_view is self.selected_view()
        self.blockViewBoxSignals()
        old_axis = self.layout.getItem(0, 0)
        if old_axis in self.axes.values():
            self.layout.removeItem(old_axis)
        self.layout.addItem(self.vbs[self.selected_view()],
                            row=0,
                            col=1)
        self.layout.addItem(self.axes[self.selected_view()],
                            row=0,
                            col=0)
        self.unblockViewBoxSignals()

    def updateWidestAxis(self):
        self.maxWidthChanged.emit()

    def selected_view(self) -> View:
        return self.display_panel.panel.selected_view

    @Slot(float, name='setAxesWidths')
    def setAxesWidths(self, width: float):
        if not self.axes or width == 0:
            return
        for axis in self.axes.values():
            if axis.width() != width:
                axis.blockSignals(True)
                axis.setWidth(w=width)
                axis.blockSignals(False)
        self.layout.update()

    def blockViewBoxSignals(self):
        self.main_vb.blockSignals(True)
        for vb in self.vbs.values():
            vb.blockSignals(True)

    def unblockViewBoxSignals(self):
        self.main_vb.blockSignals(False)
        for vb in self.vbs.values():
            vb.blockSignals(False)
