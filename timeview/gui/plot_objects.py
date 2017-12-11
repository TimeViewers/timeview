import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Signal

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class InfiniteLinePlot(pg.PlotItem):
    updatePartitionPosition = Signal(int, name='updatePartitionPosition')
    updatePartitionValuePosition = Signal(int, float, name='updatePartitionValuePosition')
    updatePartitionValue = Signal(int, name='updatePartitionValue')
    updatePartitionBoundaries = Signal(int, float, float, name='updatePartitionBoundaries')
    delete_segment = Signal(int, name='delete_segment')
    reload = Signal(name='reload')

    def __init__(self, **kwargs):
        super().__init__()
        self.setMenuEnabled(False)
        if 'view' in kwargs.keys():
            self.view = kwargs['view']