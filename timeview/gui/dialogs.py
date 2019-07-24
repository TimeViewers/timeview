import logging
from typing import Dict, Tuple, Union, Optional
import time

from qtpy import QtWidgets, QtCore, QtGui, QtHelp
from qtpy.QtCore import Slot, Signal

from .rendering import Renderer
from ..dsp.tracking import Track
from ..dsp import processing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ProcessingError(Exception):
    pass


class RenderDialog(QtWidgets.QDialog):

    def __init__(self, display_panel, renderer: Renderer):
        super().__init__()
        self.renderer = renderer
        self.panel = display_panel
        self.setWindowTitle(f'Render Parameter Entry for {renderer.name}')
        self.button_box =\
            QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                       QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.checkValues)
        self.button_box.rejected.connect(self.reject)

        self.parameters: Dict[str, str] = {}
        self.parameter_layout: Dict[str, QtWidgets.QLineEdit] = {}

        main_layout = QtWidgets.QVBoxLayout()
        self.formGroupBox = QtWidgets.QGroupBox("Parameters")
        self.createParameterLayout()

        main_layout.addWidget(self.formGroupBox)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        self.setWindowTitle(f'Parameter Entry for {renderer.name} Renderer')

    def createParameterLayout(self):
        layout = QtWidgets.QFormLayout()
        for parameter, value in self.renderer.get_parameters().items():
            self.parameter_layout[parameter] = QtWidgets.QLineEdit()
            self.parameter_layout[parameter].setText(value)
            layout.addRow(QtWidgets.QLabel(parameter),
                          self.parameter_layout[parameter])
        self.formGroupBox.setLayout(layout)

    def checkValues(self):
        for parameter, entry in self.parameter_layout.items():
            user_input = entry.text().rstrip(']) ').lstrip('([ ')
            logging.debug(f'For entry: {parameter} \t'
                          f'user entered: {entry.text()} \t'
                          f'saving as: {user_input}')
            self.parameters[parameter] = user_input
        try:
            # TODO: only pass the parameters that have been changed!
            self.renderer.set_parameters(self.parameters)
        except Exception as e:  # TODO: what kind of exception?
            logging.exception("Invalid Parameter Error")
            logging.exception(e)
            raise  # TODO: handle Invalid Parameter Exception
        self.accept()


class ProcessingDialog(QtWidgets.QDialog):
    relay_generated_tracks = Signal(tuple, name='relay_tracks')

    def __init__(self,
                 parent,
                 processor: processing.Processor):
        super().__init__(parent)
        self.setModal(False)
        self.parent = parent
        self.processor = processor
        self.abort_process = False
        self.setWindowTitle(f'Track and Parameter Entry for {processor.name}')
        self.buttonBox = QtWidgets.QDialogButtonBox()

        self.rejectButton = QtWidgets.QPushButton('Cancel')
        self.acceptButton = QtWidgets.QPushButton('Process')
        self.buttonBox.addButton(self.acceptButton,
                                 QtWidgets.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(self.rejectButton,
                                 QtWidgets.QDialogButtonBox.RejectRole)
        self.buttonBox.accepted.connect(self.preAccept)
        self.buttonBox.rejected.connect(self.reject)
        self.acceptButton.setEnabled(False)

        self.relay_generated_tracks.connect(self.parent.insert_processed_tracks)

        # {'wave': {test-mwm.wav: tracking.Wave()}}
        self.tracks: Dict[str, Dict[str, Track]] = {}
        self.parameters: Dict[str, str] = {}

        self.trackGroupBox = QtWidgets.QGroupBox("Track Selection")
        self.parameterGroupBox = QtWidgets.QGroupBox('Parameter Selection')

        self.track_layout: Dict[QtWidgets.QComboBox, str] = {}
        self.parameter_layout: Dict[str, QtWidgets.QLineEdit] = {}

        self.process_bar = QtWidgets.QProgressBar()
        self.process_bar.setRange(0, 100)
        self.process_bar.hide()

        self.createLayout()
        self.checkCurrentSelections()

        self.processor_thread = None

    def checkCurrentSelections(self):
        current_selected_track = self.parent.getSelectedTrack()
        for track_type in self.processor.acquire.values():
            for combo_box in self.track_layout.keys():
                if combo_box.count() == 1:
                    combo_box.setCurrentIndex(0)
                elif isinstance(current_selected_track, track_type):
                    combo_box.setCurrentText(current_selected_track.path.name)

    def createLayout(self):
        self.createTrackLayout()
        self.createParameterLayout()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.trackGroupBox)
        main_layout.addWidget(self.parameterGroupBox)
        main_layout.addWidget(self.buttonBox, alignment=QtCore.Qt.AlignHCenter)
        main_layout.addWidget(self.process_bar)
        self.setLayout(main_layout)

    def createTrackLayout(self):
        layout = QtWidgets.QFormLayout()
        for track_name, track_type in self.processor.acquire.items():
            combo_box = QtWidgets.QComboBox()

            tracks = [view.track
                      for panel in self.parent.model.panels
                      for view in panel.views
                      if isinstance(view.track, track_type)]
            track_name_dict: Dict[str, Track] = {track.path.name: track
                                                 for track in tracks}
            self.tracks[track_name] = track_name_dict
            combo_box.addItems(track_name_dict.keys())
            combo_box.setCurrentIndex(-1)
            combo_box.currentIndexChanged.connect(self.setData)
            self.track_layout[combo_box] = track_name
            layout.addRow(QtWidgets.QLabel(track_name),
                          combo_box)
        self.trackGroupBox.setLayout(layout)

    @Slot(name='setData')
    def setData(self):
        data: Optional[Dict[str, processing.Tracks]] = {}
        for combo_box, track_type in self.track_layout.items():
            track_name = combo_box.currentText()
            if track_name == '':
                logging.debug(f'{track_type} value entered is empty')
                return
            track = self.tracks[track_type][track_name]
            data[track_type] = track

        try:
            self.processor.set_data(data)
        except processing.InvalidDataError:
            logging.exception(f'Invalid data being passed to '
                              f'{self.processor}.set_data()')
            raise
        parameters = self.processor.get_parameters()
        for parameter, value in parameters.items():
            self.parameter_layout[parameter].setText(value)
            self.parameter_layout[parameter].setReadOnly(False)
        self.acceptButton.setEnabled(True)

    def createParameterLayout(self):
        layout = QtWidgets.QFormLayout()
        for parameter, value in self.processor.get_parameters().items():
            self.parameter_layout[parameter] = QtWidgets.QLineEdit()
            self.parameter_layout[parameter].setText(value)
            self.parameter_layout[parameter].setReadOnly(True)
            layout.addRow(QtWidgets.QLabel(parameter),
                          self.parameter_layout[parameter])
        self.parameterGroupBox.setLayout(layout)

    def preAccept(self):
        parameters = {parameter: line_edit.text()
                      for parameter, line_edit
                      in self.parameter_layout.items()}
        try:
            self.processor.set_parameters(parameters)
        except processing.InvalidParameterError:
            logging.exception(f'Invalid Parameter entered')
            raise
        self.startThread()

    def startThread(self):
        self.abort_process = False
        self.acceptButton.setEnabled(False)
        self.process_bar.show()
        self.process_bar.setValue(0)
        self.processor_thread = ProcessorThread(self.processor,
                                                self.processor_finished,
                                                self.update_process_bar)
        self.parent.application.qtapp.aboutToQuit.connect(self.processor_thread.quit)
        self.buttonBox.rejected.disconnect()
        self.buttonBox.rejected.connect(self.abort)
        self.processor_thread.start()

    @Slot(int, name='update_process_bar')
    def update_process_bar(self, value: int):
        if self.abort_process:
            return
        self.process_bar.setValue(value)

    @Slot(name='processor_terminated')
    def abort(self):
        self.abort_process = True
        logging.debug('Sending quit signal to processor thread')
        self.processor_thread.quit()
        self.processor_thread.abort = True
        self.process_bar.reset()
        # self.rejectButton.setText('Terminate!')
        # self.buttonBox.rejected.disconnect()
        # self.buttonBox.rejected.connect(self.processor_thread.quit)
        # while self.processor_thread.isRunning():
        #     logger.info('Waiting for thread to quit...')
        #     time.sleep(.1)
        # logging.info('Processor thread quit without being terminated')
        # self.rejectButton.setText('Cancel')
        self.acceptButton.setEnabled(True)
        self.buttonBox.rejected.disconnect()
        self.buttonBox.rejected.connect(self.reject)

    @Slot(tuple, name='processor_finished')
    def processor_finished(self, new_tracks):
        logger.info('processor finished')
        logger.info(f'abort status: {self.abort_process}')
        if self.abort_process:
            self.reject()
            self.close()
        else:
            self.relay_generated_tracks.emit(new_tracks)
            self.accept()


class About(QtWidgets.QMessageBox):
    def __init__(self):
        super().__init__()
        # You cannot easily resize a QMessageBox
        self.setText('© Copyright 2009-2017, TimeView Developers')
        self.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.setDefaultButton(QtWidgets.QMessageBox.Ok)
        self.setWindowTitle('About Timeview')


class Bug(QtWidgets.QDialog):
    def __init__(self, app, exc_type, exc_value, exc_traceback):
        super().__init__()
        self.app = app
        import traceback
        self.setWindowTitle('Bug Report')
        traceback_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
        self.traceback = ''.join([element.rstrip() + '\n' for element in traceback_list])
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        copyButton = QtWidgets.QPushButton('Copy To Clipboard')
        copyButton.pressed.connect(self.copyToClipboard)
        self.buttonBox.addButton(copyButton, QtWidgets.QDialogButtonBox.ApplyRole)

        main_layout = QtWidgets.QVBoxLayout()
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setLineWrapMode(0)
        self.textEdit.setText(self.traceback.replace('\n', '\r'))
        self.textEdit.setReadOnly(True)

        main_layout.addWidget(self.textEdit)
        main_layout.addWidget(self.buttonBox)
        self.setFixedWidth(self.textEdit.width() +
                           main_layout.getContentsMargins()[0] +
                           main_layout.getContentsMargins()[2])
        self.setLayout(main_layout)

    def copyToClipboard(self):
        text = "```\r" + self.textEdit.toPlainText() + "```"
        cb = self.app.clipboard()
        cb.setText(text)


class HelpBrowser(QtWidgets.QTextBrowser):
    def __init__(self, help_engine: QtHelp.QHelpEngine, parent: QtWidgets.QWidget=None):
        super().__init__(parent)
        self.help_engine = help_engine

    def loadResource(self, typ: int, name: QtCore.QUrl):
        if name.scheme() == 'qthelp':
            return QtCore.QVariant(self.help_engine.fileData(name))
        else:
            return super().loadResource(typ, name)


class InfoDialog(QtWidgets.QMessageBox):
    def __init__(self, info):
        super().__init__()
        info_string = info
        self.setText(info_string)
        self.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.setDefaultButton(QtWidgets.QMessageBox.Ok)
        self.setWindowTitle('Track info')


class ProgressTracker(QtCore.QObject):
    progress = Signal(int)

    def update(self, value):
        self.progress.emit(value)


# look at example here:
# https://stackoverflow.com/questions/25108321/how-to-return-value-from-function-running-by-qthread-and-queue
class ProcessorThread(QtCore.QThread):
    finished = Signal(tuple, name='finished')

    def __init__(self,
                 processor: processing.Processor,
                 callback,
                 update_process_bar):
        super().__init__()
        self.finished.connect(callback)
        self.processor = processor
        self.abort = False
        self.progressTracker = ProgressTracker()
        self.progressTracker.progress.connect(update_process_bar)

    def __del__(self):
        self.wait()

    def process(self) -> Tuple[Union[processing.Tracks]]:
        try:
            new_track_list = self.processor.process(progressTracker=self.progressTracker)
        except Exception as e:
            logging.exception("Processing Error")
            logging.exception(e)
            self.exit()
            # TODO: how do we want to handle processing errors?
            raise ProcessingError
        else:
            self.progressTracker.update(100)
            return new_track_list

    def run(self):
        while not self.abort:
            new_track_list = self.process()
            if self.abort:
                self.quit()
            self.finished.emit(new_track_list)
            self.abort = True
        self.quit()
