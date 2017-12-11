#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manager"""


# TODO: integrate with timeview GUI

# STL
import sys
import logging
from pathlib import Path

# 3rd party
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, StatementError
from qtpy import QtCore, uic
from qtpy.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QMessageBox
import qtawesome as qta

# local
from .dataset_manager_model import Dataset, File, Model


logger = logging.getLogger(__name__)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# logger.addHandler(ch)

ENGINE_PATH = 'sqlite:///' + str(Path(__file__).with_name('dataset.db'))

class TableModel(QtCore.QAbstractTableModel):
    # see also for alchemical model gist
    # https://gist.github.com/harvimt/4699169

    def __init__(self, model, parent):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.model = model

        # need to store parent QWidget, self.parent() is private / has different functionality
        self.widget = parent
        self.qry = []  # list of query result rows
        self.refresh()

    # def index(self, row, column, parent=QtCore.QModelIndex()):
        # return self.createIndex(row, column)

    # def parent(self, child):
        # return 0

    def rowCount(self, _parent=None):
        return len(self.qry)

    def columnCount(self, _parent=None):
        return len(self.tbl.columns)

    def headerData(self, col, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.tbl.columns[col]
            elif orientation == QtCore.Qt.Vertical:
                # self.qry[section].id
                # nobody wants to know ids presumably
                return col + 1

    def data(self, q_index, role=QtCore.Qt.DisplayRole):
        if q_index.isValid() and role == QtCore.Qt.DisplayRole:
            return self.qry[q_index.row()][q_index.column()]

    def flags(self, q_index):
        defaults = QtCore.QAbstractTableModel.flags(self, q_index)
        return defaults | QtCore.Qt.ItemIsEditable

    def setData(self, q_index, value, _role):
        # print('setdata')
        try:
            self.qry[q_index.row()][q_index.column()] = value
            self.model.session.commit()
        except (IntegrityError, StatementError) as e:
            self.model.session.rollback()
            self.widget.statusBar().showMessage(str(e))
            return False
        else:
            # TODO: correctly used?
            self.dataChanged.emit(q_index, q_index)
            self.widget.statusBar().showMessage('Updated.')
            return True

    def change_layout(self):
        self.layoutAboutToBeChanged.emit()
        self.refresh()
        self.layoutChanged.emit()


class TableModelDataset(TableModel):
    def __init__(self, *args, **kwargs):
        self.tbl = Dataset
        TableModel.__init__(self, *args, **kwargs)

    def refresh(self):
        self.qry = self.model.get_dataset()


class TableModelFile(TableModel):
    def __init__(self, *args, **kwargs):
        self.dataset_id = None
        self.tbl = File
        TableModel.__init__(self, *args, **kwargs)

    def flags(self, q_index):
        defaults = QtCore.QAbstractTableModel.flags(self, q_index)
        return defaults

    def refresh(self):
        self.qry = self.model.get_file(None, dataset_id=self.dataset_id)




class ManagerWindow(QMainWindow):
    def __init__(self, title, parent=None):
        super(ManagerWindow, self).__init__(parent)
        # TODO: for performance reasons, perhaps should precompile using pyuic when UI is near finalized
        uic.loadUi(Path(__file__).with_name("main.ui"), self)

        self.viewer = parent
        # self.setCentralWidget(self.viewer)

        # model
        self.model = Model(create_engine(ENGINE_PATH, echo=False))
        self.tableModelDataset = TableModelDataset(self.model, self)
        self.tableModelFile = TableModelFile(self.model, self)
        self.tableViewDataset.setModel(self.tableModelDataset)
        self.tableViewFile.setModel(self.tableModelFile)

        # GUI
        # update file query
        self.tableViewDataset.clicked.connect(self.clicked_dataset)
        self.tableViewFile.doubleClicked.connect(self.double_clicked_file)

        # Icons
        self.addDatasetButton.setIcon(qta.icon('fa.plus'))
        self.delDatasetButton.setIcon(qta.icon('fa.minus'))
        self.addFileButton.setIcon(qta.icon('fa.plus'))
        self.delFileButton.setIcon(qta.icon('fa.minus'))

        # first selection
        self.tableViewDataset.selectRow(0)
        try:
            self.tableModelFile.dataset_id = int(self.tableModelDataset.qry[0].id)
        except IndexError:
            pass
        else:
            self.tableModelFile.change_layout()

        self.addDatasetButton.clicked.connect(self.add_dataset)
        self.delDatasetButton.clicked.connect(self.del_dataset)
        self.addFileButton.   clicked.connect(self.add_file)
        self.delFileButton.   clicked.connect(self.del_file)

        # Status bar
        # TODO: timed statusBar (goes empty after a while)
        self.statusBar().showMessage("Ready.")

        # Window Title
        self.setWindowTitle(title)


    def clicked_dataset(self, q_index):  # refresh table of files
        self.tableModelFile.dataset_id = int(self.tableModelDataset.qry[q_index.row()].id)
        self.tableModelFile.change_layout()

    def double_clicked_file(self, q_index):
        # print('dblclick')
        row = q_index.row()
        # col = q_index.column()
        # if File.columns[col] == 'path':
        file = self.tableModelFile.qry[row].path
        self.viewer.application.add_view_from_file(Path(file))
        # self.parent().application.add_Panel
        #
        # # example setup
        # wav_file = Path(__file__).resolve().parents[0] / 'dat' / 'speech.wav'
        # wav_obj = Track.read(wav_file)
        #
        # app = TimeView()
        # # panel 0 exists already at this point
        # app.add_view(0, wav_obj)
        # app.add_view(0, lab_obj)
        # app.add_panel()
        # app.add_view(1, wav_obj, renderer='Spectrogram')  # linked
        # app.add_view(1, lab_obj)  # linked
        # app.start()
        ##########
        #self.viewer.show()

    def add_dataset(self, _e):
        while True:
            name, ok = QInputDialog.getText(self,
                                            'New Dataset',
                                            'Enter the name of the new dataset:')
            if ok:
                try:
                    # TODO: how to set defaults?
                    self.model.add_dataset(Dataset(name=name)) #parameter=0))
                except IntegrityError:
                    self.model.session.rollback()
                    QMessageBox.information(self,
                                            "Cannot proceed",
                                            "This dataset name already exists, \
                                            please select a different one (or cancel).",
                                            defaultButton=QMessageBox.Ok)
                else:
                    self.tableViewDataset.model().change_layout()
                    self.statusBar().showMessage('Added dataset.')
                    # self.tableViewDataset.selectRow(len(self.model.dataset) - 1)
                    break
            else:  # cancel
                self.statusBar().showMessage('Cancelled.')
                return  # exit loop
        if len(self.model.get_dataset()) == 1:  # first entry after being empty
            self.tableViewDataset.selectRow(0)

    def del_dataset(self, _e):
        sm = self.tableViewDataset.selectionModel()
        if sm.hasSelection():
            q_index = sm.selectedRows()
            if len(q_index):
                if QMessageBox.question(self,
                                        "Are you sure?",
                                        "You are about to delete a dataset. \
                                        This will also delete the list of files \
                                        associated with this dataset.",
                                        buttons=QMessageBox.Ok | QMessageBox.Cancel,
                                        defaultButton=QMessageBox.Cancel) == QMessageBox.Ok:
                    dataset_id = self.tableViewDataset.model().qry[q_index[0].row()].id
                    self.model.del_dataset(dataset_id)
                    self.tableViewDataset.model().change_layout()
                    self.tableViewFile.   model().change_layout()
                    # because files associated with that dataset are deleted also
                    self.statusBar().showMessage('Deleted dataset.')
                else:
                    self.statusBar().showMessage('Cancelled.')

    def add_file(self, _e):
        sm = self.tableViewDataset.selectionModel()
        if sm.hasSelection():
            q_index = sm.selectedRows()
            if len(q_index):
                dataset_qry = self.tableViewDataset.model().qry
                # dataset_id = dataset_qry[q_index[0].row()].id
                while True:
                    paths = QFileDialog.getOpenFileNames(self,
                                                         "Select one or more files",
                                                         '',
                                                         "All Files (*)")[0]
                    if len(paths):
                        dataset_id = dataset_qry[q_index[0].row()].id
                        files = [File(path=path) for path in paths]
                        try:
                            self.model.add_files(dataset_id, files)
                        except IntegrityError as e:  # this should not be happening
                            self.model.session.rollback()
                            QMessageBox.information(self,
                                                    "Integrity Error",
                                                    e,
                                                    defaultButton=QMessageBox.Ok)
                        else:
                            self.tableViewFile.model().change_layout()
                            self.statusBar().showMessage('Added file(s).')
                            break
                    else:  # cancel
                        self.statusBar().showMessage('Cancelled.')
                        return  # exit loop

    def del_file(self, _e):
        sm = self.tableViewFile.selectionModel()
        if sm.hasSelection():
            q_index = sm.selectedRows()
            if len(q_index):
                [self.model.del_file(self.tableViewFile.model().qry[qi.row()].id) for qi in q_index]
                self.tableViewFile.model().change_layout()
                self.statusBar().showMessage('Deleted file(s).')
            # else: # cancel
                # self.statusBar().showMessage('No file(s) selected.')

    # def show_error(self, exc):
        # QMessageBox.information(self, "Error", str(exc), defaultButton=QMessageBox.Ok)

    # def closeEvent(self, _e):
        # pass
