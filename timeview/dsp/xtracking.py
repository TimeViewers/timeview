from typing import Dict, Union

import numpy as np
import xarray as xr


class Track(object):  # Deriving from DataArray is not well-supported for version 0.9.6, see also issue #1097
    def __init__(self,
                 data: np.ndarray,
                 fs: int,
                 coords: Dict[str, np.ndarray]=None,
                 dims=None,
                 name: str=None,
                 attrs: dict=None):
        if attrs is None:
            attrs = {'fs': fs}
        else:
            attrs['fs'] = fs
        self.xar = xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=attrs)
        # store everything in attrs in the hope that someday we can derive

    def get_coords(self):
        return self.xar.coords
    coords = property(get_coords)

    def get_time(self) -> np.array:
        return self.xar.coords['time'].values  # this exists even if coords were not specified
    time = property(get_time)

    def get_realtime(self):
        return self.get_time() / self.xar.attrs['fs']

    realtime = property(get_realtime)

    def __str__(self):
        return self.xar.__str__()

    def __repr__(self):
        return self.xar.__repr__()


class Signal(Track):
    def __init__(self,
                 data: np.ndarray,
                 fs: int,
                 coords=None,
                 dims=None,
                 name: str=None,
                 attrs: dict=None):
        assert data.ndim == 2
        if coords is None:
            if dims is None:
                dims = ('time', 'amplitude')  # for default Wave
            else:
                assert dims[0] == 'time'
        else:
            if isinstance(coords, dict):
                assert 'time' in coords
            else:
                assert coords[0][0] == 'time'
        # if coords are not specified, coordinate variables are auto-generated if accessed & attrs are not possible
        Track.__init__(self, data, fs, coords=coords, dims=dims, name=name, attrs=attrs)


class Event(Track):
    def __init__(self,
                 data: np.ndarray,
                 time: np.ndarray,
                 fs: int,
                 name: str=None,
                 attrs: dict=None):
        if data is None:
            data = np.ones((len(time), 1), dtype=bool)
        assert data.ndim == 2
        assert data.shape[1] == 1
        assert time.ndim == 1
        assert np.all(time >= 0)
        assert np.all(np.diff(time) > 0)  # monotonically increasing
        assert len(time) == len(data)  # format is time-position and data-label associated with it
        Track.__init__(self, data, fs, coords={'time': time}, dims=('time', 'label'), name=name, attrs=attrs)


class Segmentation(Track):
    def __init__(self,
                 data: np.ndarray,
                 time: np.ndarray,
                 fs: int,
                 name: str=None,
                 attrs: dict=None):
        assert data.ndim == 2
        assert data.shape[1] == 1
        assert time.ndim == 1
        assert np.all(time >= 0)
        assert np.all(np.diff(time) > 0)  # monotonically increasing
        if len(time) == len(data) + 1:
            data = np.r_[data, np.atleast_2d(data[-1])]  # repeat last row
            # in practice, the last datum may be ignored
        elif len(time) != len(data):  # format is time-position and data-label associated with time region on RIGHT
            raise ValueError
        assert len(time) == len(data)  # format is time-position and data-label associated with it
        Track.__init__(self, data, fs, coords={'time': time}, dims=('time', 'label'), name=name, attrs=attrs)




dat = np.arange(10).reshape(-1, 1) + 10  # should be in format time x value = N x 1
wav = Signal(dat, 8000,
             name='Fun Wave',
             attrs={'unit': 'int16',
                    'min': -32768,  # could exceed extremes of data
                    'max': 32767,
                    'path': 'here'}  # path to file
             )

frm = Signal(np.random.randn(10, 4), 8000,
             name='frames')  # frames, example of 2D signal

tmv = Signal(dat, 8000,
             coords=(('time', np.arange(len(dat)) * 2 + 1),
                     ('frequency', np.array([0]), {'unit': 'Hz'})),  # a bit awkward here if we want to specify a unit
             attrs={'min': 0,  # could exceed extremes of data
                    'max': 300,
                    'path': 'here'}  # path to file
             )  # time-value

img = Signal(np.random.randn(10, 4), 8000,
             coords=(('time', np.arange(10) + 1),
                     ('frequency', np.linspace(0, 8000, 4), {'unit': 'Hz'})),
             name='spectrogram')  # image


dat = np.array(list("abcd")).reshape(-1, 1)
seg = Segmentation(dat, np.arange(len(dat) + 1) * 10, 8000)

evt = Event(None, np.arange(10), 8000)
print(evt)

