import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Union, Tuple, Optional, Type, Dict
from math import floor, ceil
from timeit import default_timer as timer

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Slot, Signal

from ..dsp import tracking, dsp, processing
from .plot_objects import InfiniteLinePlot

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class InvalidDataError(Exception):
    pass


class InvalidParameterError(Exception):
    pass


class LabelEventFilter(QtCore.QObject):
    select_next = Signal(QtWidgets.QGraphicsTextItem)
    select_previous = Signal(QtWidgets.QGraphicsTextItem)

    # be careful if changing this code, it's not intuitive, but I saw no other
    # way to get the desired behavior
    def eventFilter(self,
                    obj: Union[QtWidgets.QGraphicsTextItem],
                    event):
        if isinstance(obj, QtWidgets.QGraphicsTextItem):
            if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
                obj.setTextInteractionFlags(QtCore.Qt.TextEditable)
                obj.setFocus()
                event.accept()

            elif event.type() == QtCore.QEvent.KeyPress:
                if QtGui.QGuiApplication.keyboardModifiers() == \
                        QtCore.Qt.NoModifier:
                    if event.key() == QtCore.Qt.Key_Return:
                        obj.clearFocus()
                        return True

                    elif event.key() == QtCore.Qt.Key_Tab:
                        self.select_next.emit(obj)
                        return True

                elif event.key() == QtCore.Qt.Key_Backtab:
                    self.select_previous.emit(obj)
                    return True

            elif event.type() == QtCore.QEvent.FocusIn:
                obj.setTextInteractionFlags(QtCore.Qt.TextEditorInteraction)
                event.accept()

            elif event.type() == QtCore.QEvent.FocusOut:
                obj.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                event.accept()
            event.ignore()
            return False
        event.ignore()
        return False


class Renderer(metaclass=ABCMeta):  # MixIn
    accepts = tracking.Track
    z_value: int = 0
    name = 'metaclass'

    def __init__(self, *args, **parameters):
        self.track: Optional[processing.Tracks] = None
        self.view = None
        self.item: Union[pg.PlotItem,
                         pg.PlotCurveItem,
                         pg.ImageItem,
                         InfiniteLinePlot,
                         None] = None
        self.ax: Optional[pg.AxisItem] = None
        self.vb: Optional[pg.ViewBox] = None
        self.segments: List[pg.InfiniteLine] = []
        self.names: List[pg.TextItem] = []
        self.filter: Optional[QtCore.QObject] = None
        self.pen: Optional[QtGui.QPen] = None
        self.plot_area: pg.GraphicsView = None
        self.parameters = parameters
        if 'y_min' not in self.parameters:
            self.parameters['y_min'] = 0
        if 'y_max' not in self.parameters:
            self.parameters['y_max'] = 1

    def __str__(self) -> str:
        return self.name

    def set_track(self, track: accepts):
        self.track = track

    def set_view(self, view, **kwargs):
        self.view = view
        self.set_track(view.track)
        self.parameters['y_min'], self.parameters['y_max'] =\
            self.getDefaultYRange()
        self.parameters = {**self.parameters, **kwargs}

    def set_parameters(self, parameters: Dict[str, str]) -> None:
        old_parameters = self.parameters
        if __debug__:
            for name, value in parameters.items():
                logging.debug(f'Received parameter {name} of value {value}')
        try:
            for key, value in parameters.items():
                if isinstance(self.parameters[key], np.ndarray):
                    self.parameters[key] = np.fromstring(value.
                                                         rstrip(')]').
                                                         lstrip('[('), sep=' ')
                elif value.isdigit():
                    self.parameters[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    self.parameters[key] = float(value)
                else:
                    self.parameters[key] = type(self.parameters[key])(value)
        except Exception as e:
            raise InvalidParameterError(e)

        remaining_parameters = self.allRendererParameterProcessing(parameters)
        self.perRendererParameterProcessing(remaining_parameters)

    def allRendererParameterProcessing(self,
                                       parameters: Dict) -> Dict:
        if 'y_min' in parameters or 'y_max' in parameters:
            if self.check_y_limits():
                parameters.pop('y_min', None)
                parameters.pop('y_max', None)
                self.setLimits()
        return parameters

    def check_y_limits(self):
        y_min = self.parameters['y_min']
        y_max = self.parameters['y_max']

        if y_min >= y_max:
            logger.warning('y-min value set greater or equal to y-max value')
            return False
        return True

    def get_parameters(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.parameters.items()}

    def strColor(self) -> str:
        q_color = QtGui.QColor.fromRgb(self.view.color[0],
                                       self.view.color[1],
                                       self.view.color[2])
        return f'#{pg.colorStr(q_color)[:6]}'

    def setAxisLabel(self):
        self.ax.setLabel(self.track.label, color=self.strColor(), units=self.track.unit)

    def configNewAxis(self):
        assert isinstance(self.ax, pg.AxisItem)
        assert isinstance(self.vb, pg.ViewBox)
        self.ax.setZValue(self.z_value)
        axis_width = self.plot_area.main_window.axis_width
        self.setAxisLabel()
        if isinstance(self, Partition):
            self.ax.setStyle(showValues=False)
        self.ax.linkToView(self.vb)
        if self.ax.preferredWidth() <= axis_width:
            self.ax.setWidth(w=axis_width)
        old_axis = self.plot_area.layout.getItem(0, 0)
        if isinstance(old_axis, pg.AxisItem):
            if old_axis.width() > self.ax.width():
                axis_width = old_axis.width()
                self.ax.setWidth(w=axis_width)
            self.plot_area.layout.removeItem(old_axis)
        self.ax.update()
        self.plot_area.layout.addItem(self.ax,
                                      row=0,
                                      col=0)
        self.ax.geometryChanged.connect(self.plot_area.maxWidthChanged)

    def configNewViewBox(self):
        assert isinstance(self.vb, pg.ViewBox)
        self.setLimits()
        self.vb.setZValue(self.z_value)
        self.vb.setXLink(self.plot_area.main_vb)
        self.plot_area.layout.addItem(self.vb,
                                      row=0,
                                      col=1)

    def render(self,
               plot_area) -> Tuple[pg.AxisItem, pg.ViewBox]:
        """generates pg.AxisItem and pg.ViewBox"""
        self.plot_area = plot_area
        self.generateBlankPlotItems()
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.setMenuEnabled(False)
        return self.ax, self.vb

    @abstractmethod
    def reload(self):
        """clears current plot items, and reloads the track"""
        
    @abstractmethod
    def perRendererParameterProcessing(self, parameters):
        """depending on what the parameters changed call different methods"""

    @abstractmethod
    def generateBlankPlotItems(self):
        """creates plot items"""

    @abstractmethod
    def getDefaultYRange(self) -> Tuple[Union[int, float], Union[int, float]]:
        """returns the default y-bounds of this renderer"""

    def changePen(self):
        """changes the color/colormap of the plot"""
        self.setPen()
        self.setAxisLabel()
        self.item.setPen(self.pen)

    def setPen(self):
        self.pen = pg.mkPen(self.view.color)

    def setLimits(self):
        assert isinstance(self.vb, pg.ViewBox)
        self.check_y_limits()
        self.vb.setYRange(self.parameters['y_min'],
                          self.parameters['y_max'])
        self.vb.setLimits(yMin=self.parameters['y_min'],
                          yMax=self.parameters['y_max'])


def get_renderer_classes(accepts: Optional[tracking.Track] = None) \
        -> List[Type[Renderer]]:
    def all_subclasses(c: Type[Renderer]):
        return c.__subclasses__() + [a for b in c.__subclasses__()
                                     for a in all_subclasses(b)]

    if accepts is None:
        return [obj for obj in all_subclasses(Renderer)
                if obj.accepts is not None]
    else:
        return [obj for obj in all_subclasses(Renderer)
                if obj.accepts == accepts]


# first renderer will be the default for that track type
# | | |
# v v v


class Waveform(Renderer):
    name = 'Waveform'
    accepts = tracking.Wave
    z_value = 10

    def getDefaultYRange(self) -> Tuple[float, float]:
        if self.track.min and self.track.max:
            return self.track.min, self.track.max
        else:
            return {np.dtype('int16'): (-32768, 32768),
                    np.dtype('float'): (-1, 1)}[self.track.value.dtype]

    def reload(self):
        # TODO: waveform needs some kind of update scheme
        pass
    
    def perRendererParameterProcessing(self, parameters):
        # TODO: look at parameters and modify things accordingly
        pass

    def generateBlankPlotItems(self):
        self.item = pg.PlotCurveItem()
        self.item.setZValue(self.z_value)
        self.vb = pg.ViewBox()
        self.vb.addItem(self.item, ignoreBounds=True)
        self.ax = pg.AxisItem('left')
        self.configNewAxis()
        self.configNewViewBox()
        self.vb.setMouseEnabled(x=True, y=False)
        self.vb.sigXRangeChanged.connect(self.generatePlotData,
                                         QtCore.Qt.DirectConnection)

    def generatePlotData(self):
        # don't bother computing if there is no screen geometry
        if not self.vb.width():
            return
        # x_min, x_max = self.plot_area.main_vb.viewRange()[0]
        x_min, x_max = self.vb.viewRange()[0]
        start = max([0, int(floor(x_min * self.track.fs))])
        assert start >= 0
        if start > self.track.duration:
            return
        stop = min([self.track.duration, int(ceil(x_max * self.track.fs)) + 1])
        ds = int(round((stop - start) / self.vb.screenGeometry().width())) + 1
        if ds <= 0:
            logger.exception('ds should be > 0')
            return

        if ds == 1:
            visible = self.track.value[start:stop]
        else:
            samples = 1 + ((stop - start) // ds)
            visible = np.empty(samples * 2, dtype=self.track.value.dtype)
            source_pointer = start
            target_pointer = 0

            chunk_size = int(round((1e6 // ds) * ds))
            # assert isinstance(source_pointer, int)
            # assert isinstance(chunk_size, int)
            while source_pointer < stop - 1:
                chunk = self.track.value[
                    source_pointer:min([stop, source_pointer + chunk_size])]
                source_pointer += len(chunk)
                chunk =\
                    chunk[:(len(chunk) // ds) * ds].reshape(len(chunk) // ds,
                                                            ds)
                chunk_max = chunk.max(axis=1)
                chunk_min = chunk.min(axis=1)
                chunk_len = chunk.shape[0]
                visible[target_pointer:target_pointer + chunk_len * 2:2] =\
                    chunk_min
                visible[1 + target_pointer:1 + target_pointer + chunk_len * 2:2] =\
                    chunk_max
                target_pointer += chunk_len * 2
            visible = visible[:target_pointer]
        self.item.setData(x=np.linspace(start,
                                        stop,
                                        num=len(visible),
                                        endpoint=True) / self.track.fs,
                          y=visible,
                          pen=self.view.color)


class Spectrogram(Renderer):
    name = 'Spectrogram'
    accepts = tracking.Wave
    z_value = -100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = {'frame_size': 0.01,
                           'normalized': 0,
                           'y_min': 0,
                           'y_max': 1}
        for parameter, value in kwargs.items():
            if parameter in self.parameters.keys():
                self.parameters[parameter] = value
        # TODO: fft_size: 'auto'
        # TODO: frame_rate: 'auto'
        self.vmin = self.vmax = None

    def setAxisLabel(self):
        self.ax.setLabel('frequency', color=self.strColor(), units='Hz')

    def reload(self):
        # TODO: make some kind of reload method for the spectrogram
        pass

    def set_track(self, track: accepts):
        logging.info('setting spectrogram track')
        super().set_track(track)
        #self.set_parameters(self.parameters)  # TODO: discuss this
        self.compute_initial_levels()

    def compute_initial_levels(self):
        if self.parameters['normalized']:
            self.vmin = 0
            self.vmax = 1
        else:
            half = self.parameters['frame_size'] * self.track.fs // 2
            centers = np.round(np.linspace(half, self.track.duration - half, 1000)).astype(np.int)
            X, f = dsp.spectrogram_centered(self.track,
                                            self.parameters['frame_size'],
                                            centers,
                                            NFFT=256,
                                            normalized=self.parameters['normalized'])
            self.vmin = np.min(X)
            self.vmax = np.max(X)
            # need to have serious thinking here

    def set_parameters(self, parameters: Dict[str, str]):
        logging.info('setting spectrogram parameters')
        Renderer.set_parameters(self, parameters)
        self.compute_initial_levels()
        self.generatePlotData()

    def perRendererParameterProcessing(self, parameters):
        # # TODO: look at parameters and modify things accordingly
        # for parameter, value in parameters.items():
        #     if parameter == 'frame_size':
        #         self.generatePlotData()
        # if 'y_min' in parameters or 'y_max' in parameters:
        #     self.setLimits()
        self.setLimits()
        self.generatePlotData()

    def getDefaultYRange(self) -> Tuple[float, float]:
        return 0, self.track.fs / 2

    def generateBlankPlotItems(self):
        self.item = pg.ImageItem()
        self.item.setZValue(self.z_value)
        self.applyColor(self.view.color)
        self.vb = pg.ViewBox(parent=self.plot_area.main_vb)
        self.vb.addItem(self.item, ignoreBounds=True)
        self.ax = pg.AxisItem('left')
        self.configNewAxis()
        self.configNewViewBox()
        self.plot_area.main_vb.sigXRangeChanged.connect(self.generatePlotData)
        self.plot_area.main_vb.sigResized.connect(self.generatePlotData)

    def prepareForDeletion(self):
        self.plot_area.main_vb.sigXRangeChanged.disconnect(self.generatePlotData)
        self.plot_area.main_vb.sigResized.disconnect(self.generatePlotData)

    def generatePlotData(self):
        start = timer()
        if not self.vb or not self.vb.width():
            return
        screen_geometry = self.vb.screenGeometry()
        if screen_geometry is None:
            return
        fs = self.track.fs
        t_min, t_max = self.vb.viewRange()[0]

        # determine frame_rate and NFFT automatically
        NFFT = 2 ** max(dsp.nextpow2(screen_geometry.height() * 2), int(np.ceil(np.log2(self.parameters['frame_size'] * fs))))
        centers = np.round(np.linspace(t_min, t_max, screen_geometry.width(),
                                       endpoint=True) * fs).astype(np.int)
        # this computes regions that are sometimes grossly out of range...
        if 0:  # enable this to see when it is called
            print(f'track: {str(self.track.path).split("/")[-1]}, '
                  f'view range: {t_min:{0}.{4}}:{t_max:{0}.{4}},'
                  f'width: {screen_geometry.width()}')
        X, f = dsp.spectrogram_centered(self.track,
                                        self.parameters['frame_size'],
                                        centers,
                                        NFFT=NFFT,
                                        normalized=self.parameters['normalized'])
        if X.shape[0]:
            # TODO: how about calculating this after setting the render params?
            top = np.searchsorted(f, self.parameters['y_max'])
            bottom = np.searchsorted(f, self.parameters['y_min'])
            self.item.setImage(image=np.fliplr(-X[:, bottom:top]), levels=[-self.vmax, -self.vmin])

            rect = self.vb.viewRect()
            rect.setBottom(f[bottom])
            rect.setTop(f[top-1])
            self.item.setRect(rect)
            # print(f'Spectrogram Shape {X.shape}')
            # print(f'Screen Geometry: {screen_geometry.width()} x {screen_geometry.height()}')
            # print(f'Height: {screen_geometry.height()} \t NFFT: {NFFT}')
            # print(f"computation took {timer() - start:{0}.{4}} seconds")

    def changePen(self):
        self.applyColor(self.view.color)
        self.setAxisLabel()

    def applyColor(self, color):
        pos = np.array([1., 0.])
        color = np.array([[0, 0, 0], color], dtype=np.ubyte)
        c_map = pg.ColorMap(pos, color)
        lut = c_map.getLookupTable(start=0.0, stop=1.0, nPts=256, alpha=True)
        self.item.setLookupTable(lut)


# TODO: Alex will implement Corellogram, once unified (with plot_objects)
# #  and resampled Spectrogram is available
# class Correlogram(Renderer):
#     name = 'Correlogram'
#     accepts = tracking.Wave
#     z_value = -101
#
#     def getDefaultYRange(self) -> Tuple[float, float]:
#         return 0, self.track.fs / 2
#
#     def render(self):
#         # will call dsp.correlogram()
#         raise NotImplementedError
#
#     def changePen(self):
#         self.applyColor(self.view.color)
#         self.setAxisLabel()


class TimeValue(Renderer):
    name = 'Time-Value (read-only)'
    accepts = tracking.TimeValue
    z_value = 11

    def getDefaultYRange(self) -> Tuple[float, float]:
        return self.track.min, self.track.max

    def reload(self):
        pass
    
    def perRendererParameterProcessing(self, parameters):
        # TODO: look at parameters and modify things accordingly
        if 'y_min' in parameters or 'y_max' in parameters:
            self.setLimits()
        pass

    # TODO: finish me: editing time-only, value-only,
    # time+value, insertion, deletion
    def generateBlankPlotItems(self):
        self.item = pg.PlotCurveItem(self.track.time / self.track.fs,
                                     self.track.value,
                                     pen=self.view.color, connect='finite')
        self.ax = pg.AxisItem('left')
        self.vb = pg.ViewBox()
        self.ax.linkToView(self.vb)
        self.configNewAxis()
        self.configNewViewBox()
        self.vb.addItem(self.item, ignoreBounds=True)
        self.vb.setMouseEnabled(x=True, y=False)


class Partition(Renderer):
    accepts = None  # "abstract" rendering class
    vertical_placement = 0.1
    z_value = 100
    
    def perRendererParameterProcessing(self, parameters):
        if 'y_min' in parameters or 'y_max' in parameters:
            self.setLimits()
        # TODO: look at parameters and modify things accordingly
        pass

    def getDefaultYRange(self) -> Tuple[float, float]:
        return 0, 1

    def reload(self):
        """method to reload track"""
        self.vb.blockSignals(True)
        self.clearViewBox()
        self.segments = []
        self.names = []
        self.createSegments()
        self.vb.blockSignals(False)

    def clearViewBox(self):
        for n, item in enumerate(self.vb.addedItems[:]):
            try:
                self.vb.addedItems.remove(item)
            except AttributeError:
                try:
                    self.plot_area.removeItem(item)
                except AttributeError:
                    pass
                pass

        for ch in self.vb.childGroup.childItems():
            ch.setParentItem(None)

    def generateBlankPlotItems(self):
        kwargs = {'view': self.view}
        self.filter = LabelEventFilter()
        self.setPen()
        self.item = InfiniteLinePlot(**kwargs)
        self.item.setZValue(self.z_value)
        self.item.disableAutoRange()
        self.ax = self.item.getAxis('left')
        self.vb = self.item.getViewBox()
        self.configNewAxis()
        self.configNewViewBox()
        self.vb.setMouseEnabled(x=True, y=False)
        self.createSegments()

    def createSegments(self):
        assert len(self.track.value) == len(self.track.time) - 1
        self.createLines()
        self.createNames()
        self.positionLabels()
        # print(f'items added to viewbox {len(self.vb.addedItems)}')

    def createLines(self):
        assert isinstance(self.vb, pg.ViewBox)
        times = self.track.time / self.track.fs
        for time in times:
            line = self.genLine(time)
            line.sigDragged.connect(self.movePartition)
            line.sigPositionChangeFinished.connect(self.updateAdjacentBounds)
            self.segments.append(line)
            self.vb.addItem(line)
        self.segments[0].setMovable(False)
        for index, segment in enumerate(self.segments[1:-1], start=1):
            self.updateBounds(segment, index)
        self.vb.sigXRangeChanged.connect(self.refreshBounds)

    def genName(self, value) -> pg.TextItem:
        name = pg.TextItem(str(value),
                           anchor=(0.5, self.vertical_placement),
                           color=self.view.color,
                           border=pg.mkPen(0.4, width=1),
                           fill=None)
        name.textItem.document().contentsChanged.connect(self.nameChanged)
        name.textItem.setParent(name)
        if isinstance(self, PartitionEdit):
            name.textItem.installEventFilter(self.filter)
        name.textItem.setTabChangesFocus(True)
        name.textItem.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        return name

    def createNames(self):
        # filter has to be class attribute to prevent garbage collection
        self.filter.select_next.connect(self.select_next)
        self.filter.select_previous.connect(self.select_previous)
        self.names = [self.genName(value)
                      for value in self.track.value]
        for name in self.names:
            self.vb.addItem(name)

    def positionLabels(self):
        for index, label in enumerate(self.names):
            self.calcPartitionNamePlacement(label,
                                            index=index)

    def nameChanged(self):
        obj: pg.TextItem = self.item.sender().parent().parent().parentItem()
        assert isinstance(obj, pg.TextItem)
        index = self.names.index(obj)
        self.track.value[index] = obj.textItem.document().toPlainText()
        self.item.updatePartitionValue.emit(index)
        self.calcPartitionNamePlacement(obj, index)

    def select_next(self, source: QtWidgets.QGraphicsTextItem):
        text_item: pg.TextItem = source.parentItem()
        index = self.names.index(text_item)
        new_index = (index + 1) % len(self.names)
        source.clearFocus()
        self.focusNew(new_index)

    def focusNew(self, index: int):
        self.names[index].textItem.setFocus()

    def select_previous(self, source):
        index = self.names.index(source.parentItem())
        source.clearFocus()
        self.focusNew(index - 1)

    def calcPartitionNamePlacement(self,
                                   label: pg.TextItem,
                                   index: int,
                                   emit_signal=False):
        start_point = self.segments[index].getXPos()
        end_point = self.segments[index + 1].getXPos()
        mid_point = start_point + ((end_point - start_point) / 2)
        label.setPos(mid_point, self.vertical_placement)
        label.updateTextPos()
        if emit_signal:
            self.item.updatePartitionValuePosition.emit(index, mid_point)

    def removeSegment(self,
                      line: pg.InfiniteLine,
                      modify_track=True):
        index = self.segments.index(line)
        # would be silly to remove beginning or end line
        if index == 0 or index == len(self.segments) - 1:
            return
        label = self.names[index]

        self.segments.remove(line)
        self.names.remove(label)

        self.vb.removeItem(line)
        self.vb.removeItem(label)

        if modify_track:
            self.track.delete_merge_left(index)
        self.calcPartitionNamePlacement(self.names[index - 1],
                                        index - 1,
                                        emit_signal=False)
        self.updateBounds(self.segments[index - 1], index - 1)
        self.updateBounds(self.segments[index], index)
        if modify_track:
            self.item.delete_segment.emit(index)

        line.deleteLater()
        label.deleteLater()

    def receivePartitionValuePosition(self, index: int, mid_point: float):
        self.names[index].setPos(mid_point, self.vertical_placement)
        self.names[index].updateTextPos()

    def receiveRemoveSegment(self, _):
        # TODO: on remove segment, we may not have to reload from scratch?
        self.reload()

    def receivePartitionPosition(self, index):
        self.segments[index].setPos(self.track.time[index] / self.track.fs)

    def receivePartitionBoundaries(self,
                                   index: int,
                                   x_min: float,
                                   x_max: float):
        self.segments[index].setBounds((x_min, x_max))

    def receivePartitionValue(self, index):
        self.names[index].textItem.document().contentsChanged.disconnect()
        self.names[index].setText(str(self.track.value[index]))
        self.names[index].textItem.document().contentsChanged.\
            connect(self.nameChanged)

    def movePartition(self, line: pg.InfiniteLine):
        index = self.segments.index(line)
        self.track.time[index] = int(round(line.getXPos() * self.track.fs))
        self.calcPartitionNamePlacement(self.names[index - 1],
                                        index=index - 1,
                                        emit_signal=True)
        if index < len(self.segments) - 1:
            self.calcPartitionNamePlacement(self.names[index],
                                            index=index,
                                            emit_signal=True)
        self.item.updatePartitionPosition.emit(index)

    @Slot(pg.ViewBox, Tuple, name='refreshBounds')
    def refreshBounds(self, _, xrange: Tuple[float, float]):
        # determine the lines in the view
        x_min, x_max = xrange
        for index, line in enumerate(self.segments[1:-1], start=1):
            if line.getXPos() < x_min:
                continue
            elif x_min <= line.getXPos() <= x_max:
                self.updateBounds(line, index)
            else:
                break
        self.positionLabels()

    def updateBounds(self, line: pg.InfiniteLine, index: int):
        if index == 0 or index == len(self.segments) - 1:
            return
        last_partition = self.segments[index - 1].getXPos()
        next_partition = self.segments[index + 1].getXPos()
        cushion = self.calcCushion()
        min_bounds = last_partition + cushion
        max_bounds = next_partition - cushion
        line.setBounds((min_bounds, max_bounds))
        self.item.updatePartitionBoundaries.emit(index, min_bounds, max_bounds)

    def calcCushion(self) -> float:
        if self.vb.width():
            cushion = max([self.vb.viewPixelSize()[0], 1 / self.track.fs])
        else:
            cushion = 1 / self.track.fs
        return cushion

    def positionChangeFinished(self, line: pg.InfiniteLine):
        # called on infiniteLine.sigPositionChangeFinished
        index = self.segments.index(line)
        self.updateAdjacentBounds(index)

    def updateAdjacentBounds(self, index: int):
        self.updateBounds(self.segments[index - 1], index - 1)
        if index < len(self.segments) - 1:
            self.updateBounds(self.segments[index + 1], index + 1)

    def changePen(self):
        """changes the color/colormap of the plot"""
        pen = pg.mkPen(self.view.color)
        for name in self.names:
            name.setColor(self.view.color)
        for line in self.segments:
            line.setPen(pen)
        self.setAxisLabel()

    def genLine(self, time, movable=False):
        line = pg.InfiniteLine(time,
                               angle=90,
                               pen=self.pen,
                               movable=movable)
        line.parent = self
        return line


class PartitionRO(Partition):
    accepts = tracking.Partition
    name = 'Partition (read-only)'

    def createLines(self):
        times = self.track.time / self.track.fs
        self.segments = []
        for time in times:
            line = self.genLine(time, movable=False)
            self.segments.append(line)
            self.vb.addItem(line)

    def createNames(self):
        self.names = [self.genName(value)
                      for value in self.track.value]
        for name in self.names:
            name.textItem.setParent(name)
            self.vb.addItem(name)


class PartitionEdit(Partition):
    accepts = tracking.Partition
    name = 'Partition (editable)'

    def createLines(self):
        self.segments = []
        times = self.track.time / self.track.fs
        for time in times:
            line = self.genLine(time, movable=True)
            line.sigDragged.connect(self.movePartition)
            line.sigPositionChangeFinished.connect(self.positionChangeFinished)
            self.segments.append(line)
            self.vb.addItem(line)
        self.segments[0].setMovable(False)

        for index, segment in enumerate(self.segments[1:-1], start=1):
            self.updateBounds(segment, index)

    def insertSegment(self, x_pos: float):
        # how to check if correct view is selected
        new_value = ' '
        time = np.array([x_pos * self.track.fs]).astype(int)[0]
        index = np.searchsorted(self.track.time,
                                np.array([time])).astype(int)[0]
        dist_left = time - self.track.time[index - 1]
        if index == len(self.track.time):
            dist_right = 1
            new_name = self.genName(new_value)
            self.names.append(new_name)
            self.vb.addItem(new_name)
            item_appended = True
        else:
            item_appended = False
            dist_right = self.track.time[index] - time
        assert dist_left > 0
        assert dist_right > 0

        if dist_right > dist_left:
            value = self.track.value[index - 1]
            self.track.insert(time, value)
            self.track.value[index - 1] = new_value
        else:
            self.track.insert(time, new_value)
        self.item.reload.emit()

        if dist_right > dist_left or item_appended:
            self.names[index - 1].textItem.setFocus()
        else:
            self.names[index].textItem.setFocus()


class Event(Renderer):
    accepts = tracking.Event
    name = 'Event'
    z_value = 12

    def getDefaultYRange(self) -> Tuple[float, float]:
        return 0, 1

    def reload(self):
        pass

    def perRendererParameterProcessing(self, parameters):
        # TODO: look at parameters and modify things accordingly
        pass

    def render(self) -> Tuple[pg.AxisItem, pg.ViewBox]:
        raise NotImplementedError
        # TODO: implement me: adding, deleting, moving of events


def main():
    # example
    print(get_renderer_classes())
    track = tracking.Track.read(Path(__file__).parents[2].resolve() /
                                '/dat/speech.wav')
    print([o for o in get_renderer_classes(type(track))])
    # build dictionary of per-track availability of renderers
    available_renderers = {t.__name__:
                           {r.name: r for r in get_renderer_classes(t)}
                           for t in tracking.get_track_classes()}
    print(available_renderers)
    print(f"default wave renderer: {next(iter(available_renderers['Wave']))}")


if __name__ == '__main__':
    main()
