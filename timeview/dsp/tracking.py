"""Tracks
Each track has a fs and a duration. There are 4 kinds of tracks:

1 Event - times
2 Wave - values
3 TimeValue - values at times, duration
4 Partition - values between times

All track intervals are of the type [), and duration points to the next unoccupied sample == length
"""

import abc
import codecs
import copy
from collections import Iterable
import json
import logging
import os
import unittest
from pathlib import Path
from typing import List, Tuple

import numpy
from scipy.io.wavfile import read as wav_read, write as wav_write

from logging.config import fileConfig
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)


TIME_TYPE = numpy.int64


def convert_dtype(source, target_dtype):
    """
    return a link (if unchanged) or copy of signal in the specified dtype (often changes bit-depth as well)
    """
    assert isinstance(source, numpy.ndarray)
    source_dtype = source.dtype
    assert source_dtype in (numpy.int16, numpy.int32, numpy.float32, numpy.float64), 'source must be a supported type'
    assert target_dtype in (numpy.int16, numpy.int32, numpy.float32, numpy.float64), 'target must be a supported type'
    if source_dtype == target_dtype:
        return source
    else:  # conversion
        if source_dtype == numpy.int16:
            if target_dtype == numpy.int32:
                return source.astype(target_dtype) << 16
            else:  # target_dtype == numpy.float32 / numpy.float64:
                return source.astype(target_dtype) / (1 << 15)
        elif source_dtype == numpy.int32:
            if target_dtype == numpy.int16:
                return (source >> 16).astype(target_dtype)  # lossy
            else:  # target_dtype == numpy.float32 / numpy.float64:
                return source.astype(target_dtype) / (1 << 31)
        else:  # source_dtype == numpy.float32 / numpy.float64
            M = numpy.max(numpy.abs(source))
            limit = 1-1e-16
            if M > limit:
                factor = limit / M
                logger.warning(f'maximum float waveform value {M} is beyond [-{limit}, {limit}], applying scaling of {factor}')
                source *= factor
            if target_dtype == numpy.float32 or target_dtype == numpy.float64:
                return source.astype(target_dtype)
            else:
                if target_dtype == numpy.int16:
                    return (source * (1 << 15)).astype(target_dtype)  # dither?
                else:  # target_dtype == numpy.int32
                    return (source * (1 << 31)).astype(target_dtype)  # dither?


class Error(Exception):
    pass


class LabreadError(Error):
    pass


class MultiChannelError(Error):
    pass


class Track(metaclass=abc.ABCMeta):
    default_suffix = '.trk'

    def __init__(self, path):
        self._fs = 0
        self.type = None
        self.min = None
        self.max = None
        self.unit = None
        self.label = None
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(self.default_suffix)

    def get_time(self):
        raise NotImplementedError

    def set_time(self, time):
        raise NotImplementedError

    time = property(get_time, set_time)

    def get_value(self):
        raise NotImplementedError

    def set_value(self, value):
        raise NotImplementedError
    value = property(get_value, set_value)

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")
    fs = property(get_fs, set_fs, doc="sampling frequency")

    @abc.abstractmethod
    def get_duration(self):
        raise NotImplementedError

    def set_duration(self, duration):
        raise NotImplementedError

    duration = property(get_duration, set_duration)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    @classmethod
    def read(cls, path, *args, **kwargs):
        """Loads object from name, adding default extension if missing."""
        E = []
        suffix = Path(path).suffix
        if suffix == '.wav':
            return Wave.wav_read(path)
        elif suffix == '.tmv':
            return TimeValue.read_tmv(path)  # for now, handle nans
        elif suffix == '.lab':
            return Partition.read(path)
        else:
            raise Exception(f"I don't know how to read files with suffix {suffix}")

    def write(self, name, *args, **kwargs):
        """Saves object to name, adding default extension if missing."""
        raise NotImplementedError

    def resample(self, fs):
        """resample self to a certain fs"""
        raise NotImplementedError

    def select(self, a, b):
        """
        return a selection of the track from a to b. a and b are in fs units.
        Times are new objects, but values are views - idea is to make a read-only section, not a copy
        """
        raise NotImplementedError

    def insert(self, a, t):
        raise NotImplementedError

    def remove(self, a, b):
        raise NotImplementedError

    def copy(self, a, b):
        raise NotImplementedError

    def cut(self, a, b):
        t = self.copy(a, b)
        self.remove(a, b)
        return t


def get_track_classes() -> List[Track]:
    def all_subclasses(c):
        return c.__subclasses__() + [a for b in c.__subclasses__() for a in all_subclasses(b)]
    return [obj for obj in all_subclasses(Track)]


# TODO: class NamedEvent(_Track)
#  there hasn't been a need for it yet, but may be useful in the future
#  wonder if I can extend Event itself with optional values...
# class NamedEvent(_Track):
#  def __init__(self, time, value, fs, duration)


class Event(Track):
    def __init__(self, time, fs, duration, path=None):
        super().__init__(path)
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (numpy.diff(time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"
        assert isinstance(fs, int)
        assert fs > 0
        # assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (len(time) and duration <= time[-1]), "duration is not > times"
        self._fs = fs
        self._time = time
        self._duration = TIME_TYPE(duration)

    def get_time(self):
        assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"  # in case the user messed with .time[index] directly
        return self._time

    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (numpy.diff(time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"        
        # assert (numpy.diff(time.astype(numpy.float)) >= 0).all(), "times must be strictly monotonically increasing"
        assert not (len(time) and self._duration <= time[-1]), "duration is not > times"
        self._time = time

    def get_value(self):
        raise Exception("No values exist for Events")

    def set_value(self, value):
        raise Exception("can't set values for Events")



    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (len(self._time) and duration <= self._time[-1]), "duration is not > times"
        self._duration = duration

    time = property(get_time, set_time)
    value = property(get_value, set_value)
    duration = property(get_duration, set_duration, doc="duration of track")

    def __len__(self):
        return len(self._time)

    def __str__(self):
        return "%s, fs=%i, duration=%i." % (self._time, self.fs, self.duration)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self.fs == other.fs, "sampling frequencies must match"
        time = numpy.concatenate((self.time, (other.time + self.duration).astype(other.time.dtype)))
        duration = self.duration + other.duration
        return type(self)(time, self.fs, duration)

    def __eq__(self, other):
        if (self._fs == other._fs) and \
           (self._duration == other._duration) and \
           (len(self._time) == len(other._time)) and \
           (self._time == other._time).all():
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def crossfade(self, event, length):
        """append wave to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(event), "Cannot add Track objects of different types"
        assert self.fs == event.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert event.duration >= length
        assert self.duration >= length
        # cut left, cut right, concatenate
        a = self.select(0, self.duration - length // 2)
        b = event.select(length - length // 2, event.duration)
        return a + b

    def resample(self, fs):
        if fs != self._fs:
            factor = fs / self._fs
            # need to use numpy.round for consistency - it's different from the built-in round
            duration = int(numpy.ceil(factor * self._duration))
            if len(self._time):
                time = numpy.round(factor * self._time).astype(TIME_TYPE)
                if (numpy.diff(time) == 0).any():
                    logger.warning("new fs causes times to fold onto themselves due to lack in precision, eliminating duplicates")
                    time = numpy.unique(time)
                if duration <= time[-1]:  # try to fix this situation
                    if len(time) > 1:
                        if time[-2] == time[-1] - 1:  # is the penultimate point far enough away?
                            raise Exception('cannot adjust last time point to be smaller than the duration of the track')
                    logger.warning("new fs causes last time point to be == duration, retarding last time point by one sample")
                    time[-1] -= 1
            else:
                time = self._time
            return type(self)(time, fs, duration)
        else:
            return self

    def select(self, a, b):
        assert a >= 0
        assert b > a
        assert b <= self._duration
        ai = self.time.searchsorted(a)
        bi = self.time.searchsorted(b)
        time = self._time[ai:bi] - a
        return type(self)(time, self.fs, b - a)

    default_suffix = '.evt'

    @classmethod
    def read(cls, name, fs, duration, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        if name == os.path.splitext(name)[0]:
            ext = cls.default_suffix
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()
        if ext == '.pml' or ext == cls.default_suffix:
            self = cls.read_pml(name, fs)
        elif ext == '.pp':
            self = cls.read_PointProcess(name, fs)
        else:
            raise ValueError("file '{}' has unknown format".format(name))
        if duration:
            self.set_duration(duration)
        return self

    @classmethod
    def read_PointProcess(cls, name, fs=48000):
        raise NotImplementedError
        # from pysig import praat
        # p = praat.PointProcessFromFile(name)
        # time = [e.time for e in p.pointsP]
        # time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # if (numpy.diff(time) <= 0).any():
        #     logger.error('events are too close (for fs=%i) in file: %s, merging events' % (fs, name))
        #     time = numpy.unique(time)
        # return Event(time, fs, int(time[-1] + 1))

    @classmethod
    def read_pml(cls, name, fs=48000):
        with open(name, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.warning('pmlread(): empty file')
            return Event(numpy.empty(0, dtype=TIME_TYPE), fs, 0)
        time = numpy.zeros(len(lines), TIME_TYPE)
        for i, line in enumerate(lines):
            token = line.split(" ")
            t1 = token[0]
            t2 = token[1]
            time[i] = numpy.round(float(t2) * fs)
        if (numpy.diff(time) <= 0).any():
            logger.error('events are too close (for fs=%i) in file: %s, merging events' % (fs, name))
            time = numpy.unique(time)
        # we cannot truly know the duration, so we are giving it the minimum duration
        return Event(time, fs, int(time[-1] + 1))

    pmlread = read_pml
    @classmethod
    def read_pm(cls, name, fs, _duration):
        # suited for loading .pm files (pitch mark) that exist in CMU-ARCTIC
        with open(name, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.warning('pmread(): empty file')
            return Event(numpy.empty(0, dtype=TIME_TYPE), fs, 0)
        time = numpy.zeros(len(lines), TIME_TYPE)-1
        for i, line in enumerate(lines):
            token = line.split(" ")
            t1 = token[0]
            try:
                time[i] = numpy.round(float(t1) * fs)
            except:
                continue
            else:
                t2 = token[1]

        time = time[time!=-1]
        if (numpy.diff(time) <= 0).any():
            logger.error('events are too close (for fs=%i) in file: %s, merging events' % (fs, name))
            time = numpy.unique(time)
        # if int(time[-1] + 1) >= duration:
            # time = time[:-1]
        # we cannot truly know the duration, so we are giving it the minimum duration
        return Event(time, fs, int(time[-1] + 1))

    def write(self, name, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        self.write_pml(name)

    def write_pml(self, name):
        f = open(name, 'w')
        t1 = 0.
        for t in self.time:
            t2 = t / self.fs
            f.write('%f %f .\n' % (t1, t2))
            t1 = t2
        f.close()

    pmlwrite = write_pml

    # def __getitem__(self, index):
        # return self._time[index]

    # def __setitem__(self, index, value):
        # self._time[index] = value

    def get(self, t):
        if t in self._time:
            return True
        else:
            return False

    def draw_pg(self, **kwargs):
        raise NotImplementedError

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        if 0:
            # from matplotlib import pylab
            # pylab.plot(X, Y, 'rx-')
            # for x, y in zip(self.time, time):
            #     pylab.plot([x, x, 0], [0, y, y])
            # pylab.show()
            raise NotImplementedError
        # may have to remove some collapsed items
        assert len(numpy.where(numpy.diff(time) == 0)[0]) == 0
        self._time = time  # [index]
        self.duration = Y[-1]  # changed this from _duration

    #  TODO: NEEDS TESTING!
    def insert(self, a, t):  #
        assert isinstance(t, type(self))
        index = numpy.where((self.time >= a))[0][0]  # first index of the "right side"
        self._time = numpy.hstack((self._time[:index], t._time + a, self._time[index:] + t.duration))
        self._duration += t.duration


class Wave(Track):
    """monaural waveform"""
    default_suffix = '.wav'

    def __init__(self, value: numpy.ndarray, fs, duration=None, offset=0, path=None):
        super().__init__(path)
        assert isinstance(value, numpy.ndarray)
        assert 1 <= value.ndim, "only a single channel is supported"
        assert isinstance(fs, int)
        assert fs > 0
        self._value = value
        self._fs = fs
        self._offset = offset  # this is required to support heterogenous fs in multitracks
        self.type = 'Wave'
        self.label = f'amplitude-{value.dtype}'
        if not duration:
            duration = len(self._value)
        assert len(self._value) <= duration < len(self._value) + 1, \
            "Cannot set duration of a wave to other than a number in [length, length+1) - where length = len(self.value)"
        self._duration = duration


    def get_offset(self):
        return self._offset
    def set_offset(self, offset):
        assert 0.0 <= offset < 1.0
        self._offset = offset
    offset = property(get_offset, set_offset)

    def get_time(self):
        return numpy.arange(len(self._value)) if self._offset == 0 else numpy.arange(len(self._value)) + self._offset

    def set_time(self, time):
        raise Exception("can't set times for Wave")
    time = property(get_time, set_time)

    def get_value(self):
        return self._value
    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
        assert 1 == value.ndim, 'only a single channel is supported'
        self._value = value
        if not (len(self._value) <= self._duration < len(self._value) + 1):
            self._duration = len(self._value)
    value = property(get_value, set_value)

    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert len(self._value) <= duration < len(self._value) + 1, "Cannot set duration of a wave to other than a number in [length, length+1) - where length = len(self.value)"
        self._duration = duration
    duration = property(get_duration, set_duration)

    #def get_channels(self):
        #return 1 if self._value.ndim == 1 else self._value.shape[1] # by convention
    ##def set_channels(self, *args):
    ##    raise Exception("Cannot change wave channels - create new wave instead")
    #channels = property(get_channels)  #, set_channels)

    def get_dtype(self):
        return self._value.dtype
    #def set_dtype(self, *args):
    #    raise Exception("Cannot change wave dtype - create new wave instead")
    dtype = property(get_dtype)

    def get_bitdepth(self):
        dtype = self._value.dtype
        if dtype == numpy.int16:
            return 16
        elif dtype == numpy.float32 or dtype == numpy.int32:
            return 32
        elif dtype == numpy.float64:
            return 64
        else:
            raise Exception('unknown dtype = %s' % dtype)
    bitdepth = property(get_bitdepth)

    def __eq__(self, other):
        if (self._fs == other._fs) and \
           (self._duration == other._duration) and \
           (len(self._value) == len(other._value)) and \
           (self._value == other._value).all():
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"value={self.value}\nmin={self.value.min()}\nmax={self.value.max()}\n" \
               f"dtype={self.dtype}\nfs={self.fs}\nduration={self.duration}"
               #f"%s\nfs=%i\nduration=%i" % (self.value, self.dtype, self.fs, self.duration)

    def __len__(self):
        return len(self._value)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self._fs == other._fs, "sampling frequencies must match"
        assert self.value.dtype == other.value.dtype, "dtypes must match"
        value = numpy.concatenate((self._value, other._value))
        return type(self)(value, self.fs)

    def resample(self, fs):
        """resample to a certain fs"""
        assert isinstance(fs, int)
        if fs != self._fs:
            from pysig import multirate  # do this here, because multirate loading an external lib is problematic for the iOS port
            #import fractions
            #return type(self)(multirate.resample(self._value, fractions.Fraction(fs, self._fs)), fs)
            return type(self)(multirate.resample(self._value, self._fs, fs), fs)
        else:
            return self

    def convert_dtype(self, target_dtype):
        """returns a new wave with the waveform in the specified target_dtype"""
        return type(self)(convert_dtype(self._value, target_dtype), self._fs, path=self.path)  # TODO: take care of setting new min and max

    def select(self, a, b):
        assert a >= 0
        assert a < b  # or a <= b?
        assert b <= self.duration
        # TODO: modify this for float a and b
        return type(self)(self._value[a:b], self._fs)

    @classmethod
    def read(cls, name, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += cls.default_suffix
        return cls.read_wav(name)

    @classmethod
    def read_wav(cls, path, channel=None, mmap=False):
        """load waveform from file"""
        fs, value = wav_read(path, mmap=mmap)
        if value.ndim == 1:
            if channel is not None and channel != 0:
                raise MultiChannelError('cannot select channel {} from monaural file {}'.format(channel, path))
        if value.ndim == 2:
            if channel is None:
                raise MultiChannelError('must select channel when loading file {} with {} channels'.format(path, value.shape[1]))
            try:
                value = value[:, channel]
            except IndexError:
                raise MultiChannelError('cannot select channel {} from file {} with {} channels'.format(channel, path, value.shape[1]))
        wav = Wave(value, fs, path=path)
        if value.dtype == numpy.int16:
            wav.min = -32767
            wav.max = 32768
        else:
            raise NotImplementedError
        return wav
    wav_read = read_wav

    def write(self, name, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        self.write_wav(name)

    def write_wav(self, name):
        """save waveform to file
        The bits-per-sample will be determined by the data-type (mostly)"""
        wav_write(name, self._fs, self._value)

    wav_write = write_wav

    def __getitem__(self, index):
        return Wave(self._value[index], self.fs)

    #def __setitem__(self, index, value):
        #self._value[index] = value

        #def __add__(self, other):
        #"""wave1 + wave2"""
        #if self.fs != other.fs:
            #raise Exception("sampling frequency of waves must match")
        #return type(self)(numpy.concatenate((self.va, other.va)), self.fs)  # return correct (child) class

    #def delete(self, a, b, fade = 0):
        #pass

    #def cut(self, a, b, fade = 0):
        #wave = self.copy(a, b)
        #self.delete(a, b, fade)
        #return wave

    #def insert(self, wave, a, fade = 0):
        #"""insert wave into self at time a"""
        #if self.fs != wave.fs:
            #raise Exception("sampling frequency of waves must match")
        #if fade:
            #n = round(fade * self.fs)
            #if n*2 > len(wave.signal):
                #raise Exception("fade inverval is too large")
            #up = numpy.linspace(0, 1, n)
            #down = numpy.linspace(1, 0, n)
            #p = wave.signal.copy()
            #p[:n] *= up
            #p[-n:] *= down
            #l = self.signal[:a+n]
            #l[-n:] *= down
            #r = self.signal[a-n:]
            #r[:n] *= up
        #else:
            #self.signal = numpy.concatenate((self.signal[:a], wave.signal, self.signal[a:]))

    def crossfade(self, wave, length):
        """append wave to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(wave), "Cannot add Track objects of different types"
        assert self.fs == wave.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert wave.duration >= length
        assert self.duration >= length
        ramp = numpy.linspace(1, 0, length + 2)[1:-1]  # don't include 0 and 1
        value = self.value.copy()
        value[-length:] = value[-length:] * ramp + wave.value[:length] * (1 - ramp)  # TODO: think about dtypes here
        value = numpy.concatenate((value, wave.value[length:]))
        return type(self)(value, self.fs)

    # TODO: Test / fix me!
    def time_warp(self, x, y):
        raise NotImplementedError
        logger.warning('time_warping wave, most of the time this is not what is desired')
        time = numpy.arange(len(self._value))
        #time = index / self._fs
        time = numpy.round(numpy.interp(time, x, y)).astype(numpy.int)
        #index = int(time * self.fs)
        self._value = self._value[time]

    def draw_waveform_mpl(self, **kwargs):
        raise NotImplementedError
        # from matplotlib import pyplot as pp
        # if self.dtype == numpy.int16:
        #     limit = 2 ** 15
        # elif self.dtype == numpy.int32:
        #     limit = 2 ** 31
        # elif self.dtype == numpy.float32 or self.dtype == numpy.float64:
        #     limit = 1.0
        # else:
        #     raise
        # h = pp.plot(numpy.arange(self.duration) / self._fs, self._value / limit, **kwargs)
        # pp.axis([0, (self.duration-1) / self._fs, -1, 1])
        # #pp.yticks(numpy.arange(-1,1,0.2), numpy.arange(-1,1,0.2) * limit)
        # pp.yticks([0], [0])
        # pp.xlabel('time (s)')
        # pp.ylabel('value')
        # return h


    def draw_waveform_pg(self, **kwargs):
        import pyqtgraph as pg
        if self.dtype == numpy.int16:
            limit = 2 ** 15
        elif self.dtype == numpy.int32:
            limit = 2 ** 31
        elif self.dtype == numpy.float32 or self.dtype == numpy.float64:
            limit = 1.0
        else:
            raise
        h = pg.PlotItem(x=numpy.arange(self.duration) / self._fs, y=self._value / limit, **kwargs)
        h.setLabels(bottom='time (s)', left='value')
        return h

    draw_waveform = draw_waveform_mpl

    # TODO: stand-alone implementation, please, perhaps not here
    def specgram(self):
        f0 = 150  # average human fundamental frequency (Hz)
        t0 = self._fs / f0  # period length (samples)
        n = 2 ** int(numpy.ceil(numpy.log2(t0)))
        #frm = speech.frame(self._value, n // 2, n)
        #x = frm.value * windows.hanning(n)
        # TODO: avoid matplotlib code, write yourself!
        # from matplotlib import pyplot as pp
        # Pxx, freqs, bins, im = pp.specgram(self.value, NFFT=512, Fs=self.fs, cmap=pp.cm.gist_heat) # noverlap=256+128+64
        # return Pxx, bins, freqs
        raise NotImplementedError


    def draw_spectrogram_pg(self, **kwargs):
        H, _x, _y = self.specgram()
        import pyqtgraph as pg
        #from pyqtgraph import QtCore
        # TODO Why 10 * ? for dB scale we need 20 *
        H = 10 * numpy.log10(H) # log-magnitude
        imi = pg.ImageItem()
        imi.setImage(-H.T, autoLevels=True)
        #imi.setRect(QtCore.QRect(0, 0, 3., 11000)) # this has an integer bug in it, reported!
        imi.scale(self.duration / self.fs / imi.width(),
                  self.fs / 2 / imi.height())
        imv = pg.ViewBox()
        imv.addItem(imi)
        imp = pg.PlotItem(viewBox=imv)
        imp.setLabels(bottom='time (s)',
                      left='frquency (Hz)')
        return imp, imi

    def draw_pg(self):
        raise NotImplementedError

    def play(self):
        # from pysig.audio import audio
        # TODO is simpleaudio going to be a dependency?
        import simpleaudio as sa
        audio = (self._value / max(abs(self._value)) * 32767).astype(numpy.int16)
        obj = sa.play_buffer(audio, 1, 2, self._fs)
        obj.wait_done()

    @classmethod
    def record(cls, channels, fs, duration):
        from pysig.audio import audio
        value = audio.record(numpy.int16, channels, fs, duration)
        return Wave(value, fs)


class TimeValue(Track):
    def __init__(self, time, value, fs, duration, path=None):
        super().__init__(path)
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (numpy.diff(time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"
        assert isinstance(value, numpy.ndarray)
        assert isinstance(fs, int)
        assert fs > 0
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert len(time) == len(value), "length of time and value must match"
        assert not (len(time) and duration <= time[-1]), "duration is not > times"
        self._time = time
        self._value : numpy.ndarray = value
        self._fs = fs
        self._duration = duration
        self.min = numpy.nanmin(value)
        self.max = numpy.nanmax(value)
        self.unit = ''
        self.label = ''
        self.path = path

    def get_time(self):
        assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"  # in case the user messed with .time[index] directly
        return self._time
    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert not (len(time) and self._duration <= time[-1]), "duration is not > times"
        assert (numpy.diff(time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"
        assert len(time) == len(self._value), "length of time and value must match"
        self._time = time
    time = property(get_time, set_time)

    def get_value(self):
        return self._value
    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
        assert len(self._time) == len(value), "length of time and value must match"
        self._value = value
    value = property(get_value, set_value)

    def get_duration(self): return self._duration
    def set_duration(self, duration):  # assume times are available, if not, this must be overridden
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (len(self._time) and duration <= self._time[-1]), "duration is not > times"
        self._duration = duration
    duration = property(get_duration, set_duration, doc="duration")

    def __eq__(self, other):
        if (self._fs == other._fs) and \
           (self._duration == other._duration) and \
           (len(self._time) == len(other._time)) and \
           (len(self._value) == len(other._value)) and \
           (self._time == other._time).all() and \
           numpy.allclose(numpy.round(self._value, 3), numpy.round(other._value, 3)):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._time)

    def __str__(self):
        return "%s, fs=%i, duration=%i, path=%s" % (list(zip(self._time, self._value)), self._fs, self._duration, self.path)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self.fs == other.fs, "sampling frequencies must match"
        time = numpy.concatenate((self.time, (other.time + self.duration).astype(other.time.dtype)))
        value = numpy.concatenate((self.value, other.value))
        duration = self.duration + other.duration
        return type(self)(time, value, self.fs, duration)

    #def __iadd__(self, other):
        #assert type(other) == type(self), "Cannot add Track objects of different types"
        #assert self.fs == other.fs, "sampling frequencies must match"
        #self._time = numpy.concatenate((self._time, (other._time + self._duration).astype(other.time.dtype)))
        #self._value = numpy.concatenate((self._value, other.value))
        #duration = self.duration + other.duration
        #return self

    def resample(self, fs):
        if fs != self._fs:
            factor = fs / self._fs
            time = numpy.round(factor * self._time).astype(TIME_TYPE)
            assert (numpy.diff(time) > 0).all(), "new fs causes times to fold onto themselves due to lack in precision"
            duration = int(numpy.ceil(factor * self._duration))  # need to use numpy.round for consistency - it's different from the built-in round
            return type(self)(time, self._value, fs, duration)
        else:
            return self

    def select(self, a, b):
        """(SHOULD!) return a copy"""
        #assert isinstance(a, TIME_TYPE)  # this doesn't seem necessary
        #assert isinstance(b, TIME_TYPE)
        assert b > a
        ai = self.time.searchsorted(a)
        bi = self.time.searchsorted(b)
        time = self._time[ai:bi] - a  # not possible to make this a view
        value = self._value[ai:bi]  # this is a view - should probably be a copy.
        return type(self)(time, value, self.fs, TIME_TYPE(b - a))

    default_suffix = '.tmv'

    @classmethod
    def read(cls, name, track_name=None, fs=300_000, duration=None, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        if name == os.path.splitext(name)[0]:
            if track_name:
                ext = '.' + track_name
            else:
                ext = cls.default_suffix
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()
        if ext in ('.pit', '.nrg', cls.default_suffix):
            self = cls.read_tmv(name, fs)
        elif ext == '.f0':
            self = cls.read_pitchtier(name, fs)[0]
        elif ext == '.pitchtier':
            self = cls.read_pitchtier(name, fs)
        else:
            self = None
            try:
                if self is None:
                    self = cls.read_f0(name, fs)[0]
            except:
                pass
            try:
                if self is None:
                    self = cls.read_pitchtier(name, fs)
            except:
                pass
            if self is None:
                raise ValueError("file '{}' has unknown format".format(name))
        if duration:
            self.set_duration(duration)
        self.path = name
        return self

    @classmethod
    def read_pitchtier(cls, name, fs=48000):
        import praat
        p = praat.PointTierFromFile(name)
        time = []
        value = []
        for point in p:
            time.append(point.mark)
            value.append(float(point.time))
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        value = numpy.array(value)
        d = numpy.diff(time)
        index = numpy.where(d > 0)[0]
        time = time[index]
        value = value[index]
        return TimeValue(time, value, fs, int(numpy.ceil(p.maxTime * fs)))

    @classmethod
    def read_f0(cls, name, frameRate=0.01, frameSize=0.0075, fs=48000):
        #return TimeValue(numpy.ndarray([1]).astype(TIME_TYPE), numpy.ndarray([1]), fs, 1)
        # 4 fields for each frame, pitch, probability of voicing, local root mean squared measurements, and the peak normalized cross-correlation value
        # frame arguments are in seconds
        f = open(name, 'r')
        lines = f.readlines()
        f.close()
        F = len(lines)
        time = numpy.round((numpy.arange(F) * frameRate + frameSize / 2) * fs).astype(TIME_TYPE)
        duration = numpy.round((F * frameRate + frameSize) * fs).astype(TIME_TYPE)
        #time =              numpy.arange(len(lines)) * frameRate + frameSize / 2
        value_f0 = numpy.zeros(F, numpy.float32)
        value_vox = numpy.zeros(F, numpy.float32)
        import re
        form = re.compile(r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$')
        for i, line in enumerate(lines):
            match = form.search(line)
            if not match:
                #logger.error('badly formatted f0 file: {}'.format("".join(lines)))
                raise Exception
                # return (TimeValue(numpy.array([0]), numpy.array([100]), fs, duration),
                #         Partition(numpy.array([0, duration]), numpy.array([0]), fs))
            f0, voicing, _energy, _xcorr = match.groups()
            value_f0[i] = float(f0)
            value_vox[i] = float(voicing)
        index = numpy.where(value_f0>0)[0]  # keep only nonzeros F0 values
        pit = TimeValue(time[index], value_f0[index], fs, duration)
        vox = Partition.from_TimeValue(TimeValue(time, value_vox, fs, duration))  # return a Partition
        return pit, vox

    f0read = read_f0

    @classmethod
    def read_tmv(cls, name, fs=300000):
        obj = numpy.loadtxt(name)
        time = obj[:,0]
        value = obj[:, 1]
        time= numpy.round(time * fs).astype(TIME_TYPE)
        duration = (time[-1] + 1).astype(TIME_TYPE)
        return TimeValue(time, value, fs, duration, path=name)

    @classmethod
    def read_frm(cls, name, fs=48000):
        f = open(name, 'r')
        lines = f.readlines()
        f.close()
        frameRate = 0.01
        frameSize = 0.049
        time = numpy.round((numpy.arange(len(lines)) * frameRate + frameSize / 2) * fs).astype(TIME_TYPE)
        duration = (time[-1] + 1).astype(TIME_TYPE)
        for i, line in enumerate(lines):
            fb = numpy.array(line.split()).astype(numpy.float)
            if i == 0:
                value = numpy.zeros((len(lines), len(fb)), numpy.float64)
            value[i,:] = fb
        return TimeValue(time, value, fs, duration, path=name)

    frmread = read_frm

    def write(self, name, track_name=None, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            if track_name:
                ext = '.' + track_name
            else:
                ext = self.default_suffix
            name += ext
        self.write_tmv(name)

    def write_f0(self, name, frameRate=0.01, frameSize=0.0075, fs=48000):
        raise NotImplementedError("TimeValue.write_f0: not yet implemented")

    def write_tmv(self, name):
        obj = numpy.c_[self._time / self._fs, self._value]
        numpy.savetxt(name, obj)
        # with open(name, 'w') as f:
        #     for tv in zip(self._time / self._fs, self._value):
        #         f.write('\t'.join(str(round(x, 16)) for x in tv) + '\n')

    def write_frm(self, name):
        with open(name, 'w') as f:
            if isinstance(self._value[0], Iterable):
                for fb in self._value:
                    f.write('\t'.join(str(round(f, 3)) for f in fb) + '\n')
            else:
                for fb in self._value:
                    f.write(str(round(fb, 3)) + '\n')

    def write_pitchtier(self, name):
        """Write Praat pitch tier; it does not work."""
        import praat
        p = praat.PointTier(name, min(self.get_time()), max(self.get_time()))
        for t, value in zip(self.get_time(), self.get_value()):
            p.addPoint(value)
        with open(name, 'w') as f:
            p.write(f)  # this praat function seems to be faulty

    @classmethod
    def from_Partition(cls, p):
        """convert a partition track into a time-value track"""
        assert isinstance(p, Partition)
        time = numpy.r_[p.time[:-1], p.time[-1] - 1, p.time[1:-1] - 1]
        time.sort()
        value = numpy.array([p.value[int(i)] for i in numpy.arange(0, len(p.value), 0.5)])  # double this to keep it constant
        return TimeValue(time, value, p.fs, p.duration)

    #def __getitem__(self, index):  # should I make the default just the value?
        #return (self._time[index], self._value[index])
        #return self._value[index] # ??

    #def __setitem__(self, index, value):
        #self._time[index] = value[0]
        #self._value[index] = value[1]

    def get_index(self, t):
        """return the index of the nearest available data"""
        if len(self._time) == 0:
            raise Exception
        if t <= 0:
            return 0
        if t >= self._time[-1]:
            return len(self._value) - 1
        # t is within bounds
        ri = self._time.searchsorted(t)
        li = ri - 1
        rt = self._time[ri]
        lt = self._time[li]
        if (t - lt) < (rt - t):
            return li
        else:
            return ri

    def _get_value(self, t, interpolation="nearest"):
        # check bounds
        if len(self._time) == 0:
            raise Exception
        if t < self._time[0]:
            # raise Exception('out of bounds left')
            # logger.warning('out of bounds left')
            v = self._value[0]  # better? yes!
        elif t > self._time[-1]:
            # raise Exception('out of bounds right')
            # logger.warning('out of bounds right')
            v = self._value[-1]  # better? yes!
        else:  # t is within bounds
            ri = self._time.searchsorted(t)  # a numpy function
            rt = self._time[ri]
            rv = self._value[ri]
            if rt == t:  # exact hit
                v = rv
            else:
                li = ri - 1
                lt = self._time[li]
                lv = self._value[li]
                if interpolation == "nearest":
                    if (t - lt) < (rt - t):
                        v = lv
                    else:
                        v = rv
                else:  # linear
                    a = float(t - lt) / float(rt - lt)
                    v = a * rv + (1.0 - a) * lv
        return v

    def get(self, T, interpolation="nearest"):  # start using interp_* methods, which are faster
        """return the values at times T"""
        if isinstance(T, numpy.ndarray):
            assert T.ndim == 1
            n = self._value.shape[1] if self._value.ndim > 1 else 1  # this can be done much better I'm sure
            V = numpy.empty((T.shape[0], n), dtype=self._value.dtype)
            for i, t in enumerate(T):
                V[i] = self._get_value(t, interpolation)  # TODO: speed this up by using a cached version of the interpolate function
            return V
        else:
            return self._get_value(T, interpolation)

    def interp_linear(self, T):
        assert numpy.all(numpy.diff(self.time) > 0)
        return numpy.interp(T, self.time, self.value)

    def interp_nearest(self, T):
        assert numpy.all(numpy.diff(self.time) > 0)
        index = numpy.interp(T, self.time, numpy.arange(T))
        return self.value[numpy.round(index)]  # TODO: test me!

    def interpolate(self, T, kind='linear'):
        """kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
        where 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of first, second or third order) or as an integer
        specifying the order of the spline interpolator to use.
        Default is 'linear'."""
        from scipy.interpolate import interp1d
        if len(self.time):
            f = interp1d(self.time, self.value, kind=kind, axis=0, copy=True, bounds_error=True, fill_value=numpy.nan)
            return f(T)
        else:
            logger.warning('interpolating without data')
            return numpy.ones(len(T)) * numpy.nan

    #def select_index(self, a, b):
        #"""return indeces such that a <= time < b"""
        #assert b > a
        #return range(self.time.searchsorted(a), self.time.searchsorted(b))

    #def select(self, a, b):
        #"""copy a section of the track, interval [)"""
        ##indexa = numpy.where((self.ti - a) >= 0)[0][0] # first inside
        ##indexb = numpy.where((self.ti - b) <= 0)[0][-1] # last inside
        #index = self.select_index(a, b)
        #if len(index) == 0:
            #if type(self._value) == numpy.ndarray:
                #value = numpy.array([])
            #else:
                #value = []
            #return type(self)(time=numpy.empty(0, dtype=TIME_TYPE), value=value, fs=self.fs, duration=b-a)
        ## on the line above, I first had [0], which cause all subsequent += to be in int, causing a bad bug. Perhaps this is the reason to choose all times as int
        #ai = index[0]
        #bi = index[-1]
        #time = self._time[ai:bi] #.copy() - don't think that's needed here ...
        #value = self._value[ai:bi] #.copy()
        ### now, to copy the track accurately, we have to go all the way to the edges, and capture the values there,
        ### in accordance to the underlying interpolation function
        #### limit selection
        ###if a < self._time[0]:
            ###a = self.ti[0]
        ###if b > self.ti[-1]:
            ###b = self.ti[-1]
        ####if a < self._time[index[0]]:  # prepend - not sure if this is needed
            ####time  = numpy.concatenate(([a], time))
            ####value = numpy.concatenate(([self.get(a, interpolation)], value))
        ### can't append in any way that make sense, I _think_ TODO: Check this
        ###if self.time[index[-1]] <= b:
            ###time = numpy.concatenate((time, [b]))
            ###value = numpy.concatenate((value, f(b).T))
        ###time -= time[0]
        #time -= a # TODO: Check this!
        #tv = type(self)(time=time, value=value, fs=self.fs, duration=b-a)
        #assert type(self._value) == type(tv.value)
        #return tv

    def draw_mpl(self, **kwargs):
        from matplotlib import pyplot
        if len(kwargs) == 0:
            kwargs = {'linestyle': "-"}
        if isinstance(self._value.flat[0], numpy.number):
            h = pyplot.plot(self._time / self._fs, self._value, **kwargs)
        else:
            h = pyplot.vlines(self._time / self._fs, numpy.zeros_like(self._time), numpy.ones_like(self._time), alpha=0.5)
            for i in range(len(self._time)):
                pyplot.text(self._time[i] / self._fs, 0.5, self._value[i], color='r', verticalalignment='baseline', horizontalalignment='center', alpha=0.5, rotation=-45)
        pyplot.axis(xmin=0, xmax=self.duration / self._fs)
        pyplot.xlabel('time (s)')
        pyplot.ylabel('value')
        return h

    def draw_pg(self, **kwargs):
        raise NotImplementedError

    draw = draw_mpl

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        if 0:
            from matplotlib import pylab
            pylab.plot(X, Y, 'rx-')
            for x, y in zip(self.time, time):
                pylab.plot([x, x, 0], [0, y, y])
            pylab.show()
        assert len(numpy.where(numpy.diff(time) <= 0)[0]) == 0  # no collapsed items
        self._time = time  # don't do duration check yet
        self.duration = Y[-1]  # changed this from _duration

    def interp(self, time):  # TODO: unify with get()
        if self._value.ndim == 1:
            return numpy.interp(time, self._time, self._value, self._value[0], self._value[-1])
        elif self._value.ndim == 2:
            value = numpy.empty((len(time), self._value.shape[1]), dtype=self._value.dtype)
            for j in range(self._value.shape[1]):
                value[:,j] = numpy.interp(time, self._time, self._value[:,j], self._value[0,j], self._value[-1,j])
            return value
        else:
            raise Exception


class Label(Track):
    """Like Partition, but label regions do NOT have to be contiguous"""
    default_suffix = '.lab'

    def check(self):
        """value[k] has beginning and ending times time[2*k] and time[2*k+1]"""
        assert isinstance(self._time, numpy.ndarray)
        assert self._time.ndim == 1
        assert self._time.dtype == TIME_TYPE
        # assert (numpy.diff(self._time.astype(numpy.float)) >= 0).all(), "times must be (non-strictly) monotonically increasing"
        if not (numpy.diff(self._time.astype(numpy.float)) >= 0).all():
            logger.warning('Label times must be (non-strictly) monotonically increasing')
        assert isinstance(self._value, numpy.ndarray)
        # assert self._value.ndim == 1 # TODO: can I remove this?
        assert isinstance(self._fs, int)
        assert self._fs > 0
        assert (numpy.diff(self._time[::2]) > 0).all(), "zero-duration labels are not permitted (but abutting labels are permitted)"
        assert len(self._time) == 2 * len(self._value), "length of time and value *2 must match" # this means an empty partition contains one time value at 0!!!
        assert isinstance(self._duration, TIME_TYPE) or isinstance(self._duration, int)
        assert not (len(self._time) and self._duration < self._time[-1]), "duration is not >= times"
        return True

    def __init__(self, time, value, fs, duration, path=None):
        super().__init__(path)
        self._time = time
        self._value = value
        self._fs = fs
        self._duration = duration
        self.type = "Label"
        assert self.check()
    
    def get_time(self):
        if __debug__:
            self.check()
        return self._time

    def set_time(self, time):
        self._time = time
        if __debug__:
            self.check()
    time = property(get_time, set_time)
    
    def get_value(self):
        if __debug__:
            self.check()
        return self._value

    def set_value(self, value):
        self._value = value
        if __debug__:
            self.check()
    value = property(get_value, set_value)
    
    def get_duration(self): return self._duration

    def set_duration(self, duration):
        assert isinstance(duration, (TIME_TYPE, int, numpy.int64))
        if len(self.time) > 0:
            assert duration > self.time[-1]
        self._duration = duration
    duration = property(get_duration, set_duration, doc="duration of track")
    
    def __str__(self):
        #s = [u"0"]
        #for i in range(len(self._value)):
        #    s.append(':%s:%i/%.3f' % (self._value[i], self._time[i + 1], self._time[i + 1] / self._fs))
        s = [u""]
        for i in range(len(self._value)):
            s.append('#%i: %i/%.3f: %s :%i/%.3f\n' % (i, self._time[2*i], self._time[2*i] / self._fs, self._value[i],
                                                    self._time[2*i + 1], self._time[2*i + 1] / self._fs))
        s = "".join(s)
        return "%sfs=%i\nduration=%i" % (s, self._fs, self._duration)

    def __add__(self, other):
        if self._fs != other._fs:
            raise Exception("sampling frequencies must match")
        time = numpy.hstack((self._time, self.duration + other._time))  # other._time[0] == 0
        value = numpy.hstack((self._value, other._value))
        duration = self.duration + other.duration
        return Label(time, value, self.fs, duration)

    def __len__(self):
        return len(self._value)

    def resample(self, fs):
        """resample to a certain fs"""
        if fs != self._fs:
            factor = fs / self._fs
            time = numpy.round(factor * self._time).astype(TIME_TYPE)
            time[-1] = numpy.ceil(factor * self._time[-1])
            assert (numpy.diff(time) > 0).all(), "new fs causes times to fold onto themselves due to lack in precision"
            duration = numpy.round(factor * self._duration).astype(TIME_TYPE)
            return type(self)(time, self._value, fs, duration)
        else:
            return self    

    def select(self, a, b):
        assert 0 <= a
        assert a < b
        assert b <= self.duration
        if __debug__:  # consistency check
            self.check()
        ai = self._time.searchsorted(a)
        bi = self._time.searchsorted(b)
        if ai % 2 == 1: # inside label
            ai -= 1
        if bi % 2 == 1: # inside label
            bi += 1
        value = self._value[(ai//2):(bi//2)]
        time = self.time[ai:bi]
        time = time - a
        duration = b - a
        if time[0] < 0:
            time[0] = 0
        if time[-1] > duration:
            time[-1] = duration
        return type(self)(time.astype(TIME_TYPE), value, self.fs, duration)

    @classmethod
    def read_lbl(cls, path, fs=300000, encoding='UTF8'):
        """load times, values from a .lbl label file"""
        with open(path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(path))
        lines += '\n' # make sure it's terminated
        time = []
        value = []
        # label_type = numpy.float64
        for i, line in enumerate(lines):
            if line != '\n':
                try:
                    t1, t2, label = line.split()
                except ValueError:
                    logger.warning('ignoring line "%s" in file %s at line %i' % (line, path, i + 1))
                    continue
                t1 = float(t1) #/ 1000  # this particular
                t2 = float(t2) #/ 1000  # file format
                if label[-1] == '\r':
                    label = label[:-1]
                dur = t2 - t1
                if dur > 0:
                    time.extend([t1, t2])
                    value.append(label)
                elif dur == 0:
                    logger.warning('zero-length label "%s" in file %s at line %i, ignoring' % (label, path, i + 1))
                else:
                    raise LabreadError("label file contains times that are not monotonically increasing")
        if len(time) == 0:
            raise(Exception('file is empty or all lines were ignored'))
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than X characters
        value = numpy.array(value)
        lab = Label(time, value, fs=fs, duration=TIME_TYPE(time[-1] + fs), path=path) # best guess at duration (+1 sec)
        return lab

    @classmethod
    def read(cls, *args, **kwargs):
        return cls.read_lbl(cls, *args, **kwargs)

    def draw_pg(self, y=.5, rotation=-75, fc=['b', 'g'], ec='k', span_alpha=0.05, text_alpha=0.3, ymin=0, ymax=1, color='k', values=[]):
        raise NotImplementedError


class Partition(Track):
    default_suffix = '.lab'

    def check(self):
        assert isinstance(self._time, numpy.ndarray)
        assert self._time.ndim == 1
        assert self._time.dtype == TIME_TYPE
        # assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing"
        if not (numpy.diff(self._time.astype(numpy.float)) > 0).all():
            logger.warning('Partition: times must be strictly monotonically increasing')
        assert isinstance(self._value, numpy.ndarray)
        # assert self._value.ndim == 1 # TODO: can I remove this?
        assert isinstance(self._fs, int)
        assert self._fs > 0
        # if len(self._time):
        assert self._time[0] == 0, "partition must begin at time 0"
        # assert (numpy.diff(self._time) > 0).all(), "zero-duration labels are not permitted"
        if not (numpy.diff(self._time) > 0).all():
            logger.warning('Partition: zero-duration labels are not permitted')
        assert len(self._time) == len(self._value) + 1, "length of time and value+1 must match" # this means an empty partition contains one time value at 0!!!
        # else:
        #    assert len(self._value) == 0
        return True

    def __init__(self, time, value, fs, path=None):
        super().__init__(path)
        self._time = time
        self._value = value
        self._fs = fs
        assert self.check()

    def get_time(self):
        # assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(), "times must be strictly monotonically increasing" # in case the user messed with .time[index] directly
        if not (numpy.diff(self._time.astype(numpy.float)) > 0).all():
            logger.warning('get_time times must be strictly monotonically increasing')
        return self._time

    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        if len(time):
            assert time[0] == 0, "partition must begin at time 0"
            # assert (numpy.diff(time) > 0).all(), "zero-duration labels are not permitted"
            # assert (numpy.diff(time) >= 0).all(), "zero-duration labels are not permitted"
            if not (numpy.diff(time) > 0).all():
                logger.warning("encorter zero-duration labels are not permitted")
            assert len(time) == len(self._value) + 1, "length of time and value+1 must match"
        else:
            assert len(self._value) == 0
        self._time = time
    time = property(get_time, set_time)

    def get_value(self):
        return self._value

    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
        # assert value.ndim == 1
        assert len(self._time) == len(self._value) + 1, "length of time and value must match"
        self._value = value
    value = property(get_value, set_value)

    def get_duration(self):
        if len(self._value):
            return self._time[-1]
        else:
            return 0  # or None, or raise Exception

    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        if len(self._value):
            assert duration > self._time[-2], "can't set duration to a smaller or equal value than the next-to-last boundary (this would result in losing the last value)"
            self._time[-1] = duration
        else:
            if duration != 0:
                raise Exception("cannot set duration of an empty Partition to anything but 0")
    duration = property(get_duration, set_duration, doc="duration of track in fs")

    def __eq__(self, other):
        try:
            if (self._fs == other._fs) and \
               (len(self._time) == len(other._time)) and \
               (len(self._value) == len(other._value)) and \
               (self._time == other._time).all() and \
               (self._value == other._value).all():
                return True
        except:
            pass
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._value)

    #def resample(self, fs):
        #if fs != self._fs:
            #factor = fs / self._fs
            #self._time = numpy.round(factor * self._time).astype(TIME_TYPE)
            #assert (numpy.diff(self._time) > 0).all(), "new fs causes times to fold onto themselves due to lack in precision"
            #self._fs = fs

    def resample(self, fs):
        """resample to a certain fs"""
        if fs != self._fs:
            #return type(self)(time = self._time.copy() * fs, value = self._value.copy(), fs=fs, duration=self._duration)
            factor = fs / self._fs
            time = numpy.round(factor * self._time).astype(TIME_TYPE)
            time[-1] = numpy.ceil(factor * self._time[-1])
            assert (numpy.diff(time) > 0).all(), "new fs causes times to fold onto themselves due to lack in precision"
            return type(self)(time, self._value, fs)
        else:
            return self

    #def select(self, a, b):
        #assert b > a
        #ai = self.time.searchsorted(a)
        #bi = self.time.searchsorted(b)
        #time = self._time[ai:bi] - a
        #value = self._value[ai:bi]
        #return type(self)(time, value, self.fs, b-a)

    def select(self, a, b):
        assert 0 <= a
        assert a < b
        assert b <= self._time[-1]
        if __debug__:  # consistency check
            self.check()
        ai = self._time.searchsorted(a)
        bi = self._time.searchsorted(b)
        if a < self._time[ai]:  # prepend
            ai -= 1
        if self._time[bi] < b:  # append
            bi += 1
        value = self._value[ai:bi]
        time = self.time[ai:bi+1].copy()  # otherwise we are going to modify the original below!
        #if a < self._time[ai]:  # prepend
        time[0] = a
        #if self._time[bi] < b:  # append
        time[-1] = b
        time = time - a  # must make a copy!
        return type(self)(time.astype(TIME_TYPE), value, self.fs)

    @classmethod
    def read(cls, name, fs=300_000, track_name=None, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing.
        It figures out which read function is appropriate based on extension.
        """
        if name == os.path.splitext(name)[0]:
            assert track_name, "neither track name nor extension specified"
            ext = '.' + track_name
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()[1:]
        if ext in ('lab', 'dem', 'rec'):
            return cls.read_lab(name, fs)
        elif ext == 'textgrid':
            return cls.read_textgrid(name, fs)
        else:
            try:
                return cls.read_lab(name, fs)
            except:
                pass
            try:
                return cls.read_textgrid(name, fs)
            except:
                pass
            raise ValueError("file '{}' has unknown format".format(name))

    @classmethod
    def read_textgrid(cls, name, fs=48000, encoding="UTF-8"):
        from pysig import praat
        t = praat.TextGridFromFile(name)
        time = [0]
        value = []
        for interval in t[0]:
            time.append(interval.maxTime)
            value.append(interval.mark.strip())
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # TODO: assert labels are not longer than 16 characters
        return Partition(time, numpy.array(value, dtype='U16'), fs=fs)  # up to 16 characters (labels could be words)

    @classmethod
    def read_lab(cls, name, fs=300_000, encoding='UTF8'):
        """load times, values from a label file"""
        with codecs.open(name, 'r', encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(name))
        time = []
        value = []
        #label_type = numpy.float64
        for i, line in enumerate(lines):
            try:
                t1, t2, label = line[:-1].split()
            except ValueError:
                logger.warning('ignoring line "%s" in file %s at line %i' % (line, name, i + 1))
                continue
            t1 = float(t1)
            t2 = float(t2)
            if label[-1] == '\r':
                label = label[:-1]
            #try:
            #    label = label_type(label)
            #except ValueError:
            #    if len(time) == 0:
            #        label_type = str
            #        label = label_type(label)
            #    else:
            #        raise
            if len(time) == 0:
                time.append(t1)
            else:
                if time[-1] != t1:
                    logger.warning('noncontiguous label "%s" in file %s at line %i, fixing' % (label, name, i + 1))
            dur = t2 - time[-1]
            if dur > 0:
                time.append(t2)
                value.append(label)
            elif dur == 0:
                logger.warning('zero-length label "%s" in file %s at line %i, ignoring' % (label, name, i+1))
            else:
                raise LabreadError("label file contains times that are not monotonically increasing")
        if len(time) == 0:
            raise(Exception('file is empty or all lines were ignored'))
        if time[0] != 0:
            logger.warning('moving first label boundary to zero')
            time[0] = 0
            # or insert a first label
            #time.insert(0, 0)
            #value.insert(0, default_label)
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than 8 characters
        #value = numpy.array(value, dtype='U16' if label_type is str else numpy.float64)
        value = numpy.array(value, dtype='U16')# if label_type is str else numpy.float64)
        return Partition(time, value, fs=fs, path=name)  # u1p to 16 characters (labels could be words)

    @classmethod
    def read_partition(cls, name, fs=48000, encoding='UTF8'):
        """load times, values from a label file"""
        with codecs.open(name, 'r', encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(name))
        time = [0]
        value = []
        label_type = numpy.float64
        for i, line in enumerate(lines):
            try:
                t, _t, label = line[:-1].split()
            except ValueError:
                logger.warning('ignoring line "%s" in file %s at line %i' % (line, name, i + 1))
                continue
            t = float(t)
            if label[-1] == '\r':
                label = label[:-1]
            try:
                label = label_type(label)
            except ValueError:
                if len(time) == 1:
                    label_type = unicode
                    label = label_type(label)
                else:
                    raise
            if len(time) == 1:
                dur = t
            else:
                dur = t - time[-1]

            time.append(t)
            if dur > 0:
                value.append(label)
            elif dur == 0:
                logger.warning('zero-length label "%s" in file %s at line %i, ignoring' % (label, name, i+1))
            else:
                raise LabreadError("label file contains times that are not monotonically increasing")
        if len(time) == 1:
            raise(Exception('file is empty or all lines were ignored'))
        if time[0] != 0:
            logger.warning('moving first label boundary to zero')
            time[0] = 0
            # or insert a first label
            #time.insert(0, 0)
            #value.insert(0, default_label)
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than 8 characters
        value = numpy.array(value, dtype='U16' if label_type is unicode else numpy.float64)
        return Partition(time, value, fs=fs)  # u1p to 16 characters (labels could be words)

    def write(self, name):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            assert name, "neither track name nor extension specified"
            name += self.default_suffix
        self.write_lab(name)

    def write_lab(self, file):
        if hasattr(file,'read'):
            f = file
        else:
            f = open(file, 'w')
        for i, v in enumerate(self._value):
            f.write("%f %f %s\n" % (self._time[i] / self.fs, self._time[i + 1] / self.fs, v))
        f.close()

    def write_partition(self, name):
        # written to generate a format exactly like
        # CMU-ARCTIC lab/*.lab files
        f = open(name, 'w')
        f.write("#\n")
        for i, v in enumerate(self._value):
            f.write("%f 125 %s\n" % (self.fs, self._time[i + 1] / self.fs, v))
        f.close()

    def write_textgrid(self, name):
        from pysig import praat
        it = praat.IntervalTier(name='1', minTime=0, maxTime=self.duration / self.fs)
        #time = numpy.array([    0, 10392, 20856, 33974, 35981, 37162, 63134, 82589], dtype=numpy.int32)
        #value = numpy.array([u'.pau', u'h', u'a\u028a', u'tc', u't', u's', u'.pau'], dtype='<U16')
        for i, v in enumerate(self._value):
            it.add(self._time[i] / self.fs, self._time[i+1] / self.fs, v)
        tg = praat.TextGrid(maxTime=self.duration / self.fs)
        tg.append(it)
        tg.write(name)
        
    @classmethod
    def from_TimeValue(cls, tv):
        """convert a time value track with repeating values into a partition track"""
        assert isinstance(tv, TimeValue)
        boundary = numpy.where(numpy.diff(tv.value))[0]
        time = numpy.empty(len(boundary), dtype=tv.time.dtype)
        value = numpy.empty(len(boundary) + 1, dtype=tv.value.dtype)
        duration = tv.duration
        for i in range(len(time)):
            index = boundary[i]
            time[i] = (tv.time[index] + tv.time[index + 1]) / 2
        time = numpy.concatenate(([0], time, [duration])).astype(TIME_TYPE)
        value[0] = tv.value[0]
        for i in range(len(boundary)):
            value[i + 1] = tv.value[boundary[i] + 1]
        par = Partition(time, value, tv.fs, path=tv.path)
        assert par.check()
        return par

    def __getitem__(self, index):
        return self._time[index], self._value[index], self._time[index + 1]

    def __setitem__(self, index, value):
        raise NotImplementedError  # this would be some work to support

    def __str__(self):
        s = [""]
        for i in range(len(self._value)):
            s.append('#%i: %i/%.3f: %s :%i/%.3f\n' % (i, self._time[i], self._time[i] / self._fs, self._value[i],
                                                    self._time[i + 1], self._time[i + 1] / self._fs))
        s = "".join(s)
        return "%s\nfs=%i\nduration=%i" % (s, self._fs, self.duration)

    def __add__(self, other):
        if self._fs != other._fs:
            raise Exception("sampling frequencies must match")
        time = numpy.hstack((self._time, self._time[-1] + other._time[1:]))  # other._time[0] == 0
        value = numpy.hstack((self._value, other._value))
        #duration = self.duration + other.duration
        return Partition(time, value, self.fs)

    def crossfade(self, partition, length):
        """append to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(partition), "Cannot add Track objects of different types"
        assert self.fs == partition.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert partition.duration >= length
        assert self.duration >= length
        # cut left, cut right, concatenate, could be sped up
        a = self.select(0, self.duration - length // 2)
        b = partition.select(length - length // 2, partition.duration)
        return a + b

    #def __iadd__(self, other):
        #if self._fs != other._fs:
            #raise Exception("sampling frequencies must match")
        #self._time = numpy.concatenate((self._time, other._time[1:] + self._time[-1]))  # other._time[0] must be 0
        #if isinstance(self._value, numpy.ndarray):
            #self._value = numpy.hstack((self._value, other._value))
        #elif isinstance(self._value, list):
            #self._value += other._value # list concatenation
        #else:
            #raise Exception
        ##self._duration += other._duration # TODO: test me!
        #return self

    def get(self, t):
        """returns current label at time t"""
        #return self.value[numpy.where((self.time - t) <= 0)[0][-1]] # last one that is <= 0
        return self._value[(self._time.searchsorted(t + 1) - 1).clip(0, len(self._value) - 1)]

    def append(self, time, value):
        """appends a value at the end, time is new endtime"""
        assert time > self._time[-1]
        self._time = numpy.hstack((self._time, time))
        self._value = numpy.hstack((self._value, value))  # so values must be numpy objects after all ?

    def insert(self, time: int, value):
        """modifies partition object to include value at time - other times are unchanged"""
        assert not (time == self._time).any(), "this boundary exists already"
        index = numpy.searchsorted(self._time, numpy.array([time]))[0]  # move _this_ index to the right
        self._time = numpy.hstack((self._time[:index], time, self._time[index:]))
        # so values must be numpy objects after all ?
        self._value = numpy.hstack((self._value[:index], value, self._value[index:]))

    def delete_merge_left(self, index):
        # TODO write unittests for me and my other half /
        # or replace by just CUT if it's possible to uniquely specify the latter
        """deletes a phoneme, leaves duration as is"""
        assert len(self._value) > 1
        assert 0 <= index < len(self._value)
        self._time = numpy.hstack((self._time[:index], self._time[index+1:]))
        self._value = numpy.hstack((self._value[:index], self._value[index+1:]))

    def delete_merge_right(self, index):
        """deletes a phoneme, leaves duration as is"""
        assert len(self._value) > 1
        assert 0 <= index < len(self._value)
        self._time = numpy.hstack((self._time[:index+1], self._time[index+2:]))
        self._value = numpy.hstack((self._value[:index], self._value[index+1:]))

    def merge_same(self):  # rename to uniq?
        """return a copy of myself, except identical values will be merged"""
        if len(self._value) <= 1:
            return Partition(self._time, self._value, self.fs)
        else:
            time = [self._time[0]]
            value = [self._value[0]]
            for i, v in enumerate(self._value[1:]):
                if v != value[-1]:  # new value
                    time.append(self._time[i+1])  # close last one, begin new one
                    value.append(v)
            time.append(self.duration)
            return Partition(numpy.array(time, dtype=TIME_TYPE), numpy.array(value), self.fs)

    #def select_index(self, a, b):
        #"""return indeces such that a <= time < b"""
        #return range(self._time.searchsorted(a), self._time.searchsorted(b))

    #def select(self, a, b):
        #"""copy a section, interval [)"""
        #index = self.select_index(a, b)
        #if len(index) == 0:
            #return type(self)(time=numpy.array([0, b-a], dtype=TIME_TYPE), value=numpy.array([self.get(a)]), fs=self.fs)
        #time = self._time[index] #.copy() - don't think that's needed here ...
        #value = self._value[index[0]:index[-1]] #.copy()
        ### limit selection
        ##if a < self._time[0]:
            ##a = self.ti[0]
        ##if b > self.ti[-1]:
            ##b = self.ti[-1]
        #if a < self._time[index[0]]:  # prepend
            #time  = numpy.concatenate(([a], time))
            #value = numpy.concatenate(([self.get(a)], value))
        #if self.time[index[-1]] <= b:
            #time = numpy.concatenate((time, [b]))
            #value = numpy.concatenate((value, [self.get(b)]))
        #time -= time[0]
        #return type(self)(time=time, value=value, fs=self.fs)

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        # assert len(numpy.where(numpy.diff(time) <= 0)[0]) == 0, 'some segment durations are non-positive' # Must allow this after all
        # self._time = time
        # may have to remove some collapsed items
        self._time = time
        while 1:
            index = numpy.where(numpy.diff(self._time) == 0)[0]
            if len(index):
                logger.warning('removing collapsed item #{}: "{}"'.format(index[0], self.value[index[0]]))
                self.delete_merge_right(index[0])
            else:
                break

    def draw_pg(self, y=.5, rotation=-75, fc=['b', 'g'], ec='k', span_alpha=0.05, text_alpha=0.3, ymin=0, ymax=1, color='k', values=[]):
        raise NotImplementedError


class Value(Track):
    """can store a singular value (e.g. comment string) of any type"""
    def __init__(self, value, fs, duration, path):
        super().__init__(path)
        assert isinstance(fs, int)
        assert fs > 0
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        self._value = value
        self._fs = fs
        self._duration = duration

    def __getitem__(self, k):
        return self.value[k]

    def __setitem__(self, k, v):
        self.value[k] = v

    def __eq__(self, other):
        if (self._fs == other._fs) and \
           (self._duration == other._duration) and \
           (type(other.value) is type(self.value)):
            return self.value == other.value
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return 1

    def __str__(self):
        return self._value

    def get_duration(self): return self._duration
    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        self._duration = duration
    duration = property(get_duration, set_duration, doc="duration of track")

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value
    value = property(get_value, set_value)

    def select(self, a, b):
        raise NotImplementedError

    def time_warp(self, _x, y):
        self.duration = y[-1]  # changed this from _duration

    @classmethod
    def identify_ext(cls, name, track_name=None):
        if name == os.path.splitext(name)[0]:
            if track_name is None:
                raise ValueError("neither track name nor extension specified")
            elif track_name == 'input':
                ext = '.txt'
            elif track_name == 'xml':
                ext = '.xml'
            elif track_name == 'search':
                ext = '.search'
            elif track_name == 'pronunciation':
                ext = '.pron'
            else:
                raise ValueError("unknown track name '{}'".format(track_name))
        else:
            ext = os.path.splitext(name)[1]
        return ext

    @classmethod
    def read(cls, name, fs, duration, track_name=None, *_args, **_kwargs):
        """Loads object from name, adding extension if missing."""
        ext = cls.identify_ext(name, track_name)
        name = os.path.splitext(name)[0] + ext
        if ext in ('.txt', '.xml', '.search'):
            with open(name, 'r') as f:
                value = f.read()
                if ext == '.search':
                    value = unicode(value, "UTF-8")
                self = Value(value, fs, duration)
        elif ext == '.pron':
            with open(name, 'r') as f:
                pron = [tuple(line.split('\t')) for line in f.read().split('\n')]
                self = Value(pron, fs, duration)
        else:
            raise ValueError("unknown extension '{}'".format(ext))
        return self

    def write(self, name, track_name=None, *_args, **_kwargs):
        """
        Saves object to name, adding extension if missing.
        """
        ext = self.identify_ext(name, track_name)
        name = os.path.splitext(name)[0] + ext
        if ext in ('.txt', '.xml', '.search'):
            with open(name, 'w') as f:
                f.write(self.get_value())
        elif ext == '.pron':
            with open(name, 'w') as f:
                f.write('\n'.join(word + '\t' + pron
                                  for word, pron in self.get_value()))
        else:
            raise ValueError("unknown extension '{}'".format(ext))


class MultiTrack(dict):
    """
    A dictionary containing time-synchronous tracks of equal duration and fs
    """
    def __init__(self, mapping={}):
        dict.__init__(self, mapping)
        if __debug__:  # long assert - TODO: do this on mapping, and then assign
            self.check()

    def check(self):
        if len(self) > 1:
            for i, (key, track) in enumerate(self.items()):
                if track.fs != self.fs:
                    raise AssertionError("all fs' must be equal, track #%i ('%s') does not match track #1" % (i, key))
                if track.duration != next(iter(self.values())).duration:
                    raise AssertionError("all durations must be equal, track #%i ('%s') does not match track #1" % (i, key))

    def get_fs(self):
        if len(self):
            return next(iter(self.values())).fs
        else:
            return 0  # or raise?

    def set_fs(self, fs):
        raise Exception("Cannot change fs, try resample()")
    fs = property(get_fs, set_fs, doc="sampling frequency")

    def get_duration(self):
        if len(self):
            if __debug__:  # long assert - TODO: do this on mapping, and then assign
                self.check()
            return next(iter(self.values())).duration
        else:
            return 0
    def set_duration(self, duration):
        raise Exception("The duration cannot be set, it is derived from its conents")
    duration = property(get_duration, set_duration, doc="duration, as defined by its content")

    def __eq__(self, other):
        # excluding wav from comparison as long as wav writing/reading is erroneous
        if (set(self.keys()) - {'wav'}) != (set(other.keys()) - {'wav'}):
            return False
        for k in self.iterkeys():
            if k != 'wav' and self[k] != other[k]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __setitem__(self, key, value):
        if len(self):
            if value.duration != self.duration:
                raise AssertionError("duration does not match")
            if value.fs != self.fs:
                raise AssertionError("fs does not match")
        dict.__setitem__(self, key, value)

    def __str__(self):
        s = ""
        for key, track in self.iteritems():
            s += "%s: %s\n" % (key, track)
        return s

    def __add__(self, other):
        if self is other:
            other = copy.deepcopy(other)
        obj = type(self)()
        for k in self:  #.iterkeys():
            obj[k] = self[k] + other[k]
        return obj

    def resample(self, fs):
        multiTrack = type(self)()
        for key, track in self.items():
            multiTrack[key] = track.resample(fs)
        return multiTrack

    def crossfade(self, other, length):
        """
        append multiTrack to self, using a crossfade of a specified length in samples
        """
        assert type(self) == type(other)
        assert self.keys() == other.keys()
        assert self.fs == other.fs
        assert isinstance(length, int)
        assert length > 0
        assert other.duration >= length
        assert self.duration >= length
        multiTrack = type(self)()
        for key, track in self.items():
            multiTrack[key] = self[key].crossfade(other[key], length)
        return multiTrack

    def select(self, a, b, keys=None):
        assert a >= 0
        assert a < b  # or a <= b?
        assert b <= self.duration
        """return a new multitrack object with all track views from time a to b"""
        if keys is None:
            keys = self.keys()
        multiTrack = type(self)()
        for key in keys:
            multiTrack[key] = self[key].select(a, b)
        return multiTrack

    # TODO: should this be deprecated in favor of / should this call - the more general time_warp function?
    def scale_duration(self, factor):
        if factor != 1:
            for t in self.itervalues():
                if isinstance(t, Partition):
                    t.time *= factor  # last time parameter IS duration, so no worries about duration
                elif isinstance(t, TimeValue) or isinstance(t, Event):
                    if factor > 1:  # make room for expanded times
                        t.duration = int(t.duration * factor)
                        t.time *= factor
                    else:
                        t.time *= factor
                        t.duration = int(t.duration * factor)
                else:
                    raise NotImplementedError  # wave?

    def time_warp(self, x, y):
        """in-place"""
        for track in iter(self.values()):
            track.time_warp(x, y)

    def draw_pg(self, keys=None):
        raise NotImplementedError

    def draw_speech_pg(self):
        raise NotImplementedError

    default_suffix = '.mtt'

    @classmethod
    def read(cls, name):
        """Loads info about stored tracks from name, adding extension if missing,
        and loads tracks by calling read(<name without extension>) for them.
        """
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += cls.default_suffix
        with open(name, 'rb') as mtt_file:
            track_infos = json.load(mtt_file)
        self = cls()
        for track_type_name, track_info_list in track_infos:
            track_type = globals()[track_type_name]
            track_info = dict(track_info_list)
            track = track_type.read(name_wo_ext, **track_info)
            self[track_info['track_name']] = track
        return self

    def write(self, name):
        """Saves info about stored tracks to name, adding extension if missing,
        and calls write(<name without extension>) for the contained tracks.
        Note!: not saving wav as long as wav writing/reading is erroneous
        """
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        track_infos = []  # list of dicts storing track info
        for track_name, track in sorted(self.items()):
            if track_name == 'wav':
                continue
            track_info = {'track_name': track_name,
                          'fs': int(track.get_fs()),
                          'duration': int(track.get_duration())}
            if type(track) == Value:
                track_info.update({'value_type': type(track.get_value()).__name__})
            track.write(name_wo_ext, **track_info)
            track_infos.append((type(track).__name__, sorted(track_info.items())))
        with open(name, 'wb') as mtt_file:
            json.dump(track_infos, mtt_file)


class HetMultiTrack(MultiTrack):  # may want to define common abstract class instead
    """
    A dictionary containing time-synchronous tracks of equal duration, but HETEROGENOUS fs
    """

    # this fs relates to the manner by which we time-index (possibly with float) into the multitrack object.
    # Use 1.0 for seconds.
    def __init__(self, mapping={}, fs=1.0):
        dict.__init__(self, mapping)
        if __debug__:  # long assert - TODO: do this on mapping, and then assign
            self.check()
        self._fs = fs

    def check(self):
        if len(self) > 1:
            duration = None
            for i, (key, track) in enumerate(self.items()):
                if duration is None:
                    duration = track.duration / track.fs
                if track.duration / track.fs != duration:
                    raise AssertionError("all durations must be equal, track #%i ('%s') does not match track #1" % (i, key))

    def get_fs(self):
        if len(self):
            return self._fs
        else:
            return 0  # or raise?

    def set_fs(self, fs):
        self._fs = fs
    fs = property(get_fs, set_fs, doc="sampling frequency of time-index")

    def select(self, a, b, keys=None):
        assert a >= 0
        assert a < b  # or a <= b?
        assert b <= self.duration
        """return a new object with all track views from time a to b"""
        if keys is None:
            keys = self.keys()
        obj = type(self)()
        for key in keys:
            trk = self[key]
            obj[key] = trk.select(a / self._fs * trk._fs, b / self._fs * trk._fs)  # untested
        return obj


class TestEvent(unittest.TestCase):
    def setUp(self):
        self.t = Event(numpy.array([3, 6], dtype=TIME_TYPE), 1, 10)
        self.u = Event(numpy.array([0, 2], dtype=TIME_TYPE), 1, 3)

    def test_init(self):
        t = Event(numpy.array([], dtype=TIME_TYPE), 1, 0)  # empty
        t = Event(numpy.array([], dtype=TIME_TYPE), 1, 10)  # empty
        self.assertRaises(AssertionError, Event, numpy.array([6, 3], dtype=TIME_TYPE), 1, 10)  # bad times
        self.assertRaises(Exception, Event, numpy.array([3, 6], dtype=TIME_TYPE), 1, 5)  # duration too short

    def test_duration(self):
        self.t.duration = 8  # ok
        self.assertRaises(Exception, self.t.set_duration, 5)  # duration too short

    def test_eq(self):
        self.assertTrue(self.t == self.t)
        self.assertFalse(self.t == self.u)

    def test_add(self):
        v = self.t + self.u
        self.assertTrue(v == Event(numpy.array([3, 6, 10, 12], dtype=TIME_TYPE), 1, 13))
        self.t += self.u
        self.assertTrue(self.t == Event(numpy.array([3, 6, 10, 12], dtype=TIME_TYPE), 1, 13))

    def test_select(self):
        t = self.t.select(2,7)
        self.assertTrue(t == Event(numpy.array([1, 4], dtype=TIME_TYPE), 1, 5))
        t = self.t.select(3,7)
        self.assertTrue(t == Event(numpy.array([0, 3], dtype=TIME_TYPE), 1, 4))
        t = self.t.select(3,6)
        self.assertTrue(t == Event(numpy.array([0], dtype=TIME_TYPE), 1, 3))
        t = self.t.select(2,6)
        self.assertTrue(t == Event(numpy.array([1], dtype=TIME_TYPE), 1, 4))

    def test_pml(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix='test_pml_')
        filename = tmp.name
        tmp.close()
        self.t.pmlwrite(filename)
        s = Event.pmlread(filename)
        os.unlink(filename)
        # duration CANNOT be encoded in the file (or can it?)
        s.duration = int(numpy.round(self.t.duration * s.fs / self.t.fs))
        s = s.resample(self.t.fs)
        self.assertTrue(numpy.allclose(s.time, self.t.time))


class TestWave(unittest.TestCase):
    def setUp(self):
        self.w = Wave(numpy.arange(0, 16000), 16000)
        self.v = Wave(numpy.arange(100, 200), 16000)

    def test_init(self):
        t = Wave(numpy.array([], dtype=TIME_TYPE), 1)  # empty

    def test_eq(self):
        self.assertTrue(self.w == self.w)
        self.assertFalse(self.w == self.v)

    def test_add(self):
        t = self.w + self.v
        self.assertTrue(t.duration == 16100)
        self.assertTrue(t.value[16050] == 150)
        self.w += self.v
        self.assertTrue(self.w.duration == 16100)
        self.assertTrue(self.w.value[16050] == 150)

    def test_select(self):
        w = self.w.select(10,20)
        self.assertTrue(w == Wave(numpy.arange(10, 20, dtype=TIME_TYPE), 16000))


class TestTimeValue(unittest.TestCase):
    def setUp(self):
        self.t1 = TimeValue((numpy.linspace(1, 9, 3)).astype(TIME_TYPE), numpy.array([1, 4, 2]), 1, 10)
        self.t2 = TimeValue((numpy.linspace(2, 8, 4)).astype(TIME_TYPE), numpy.array([1, 4, 8, 2]), 1, 10)
        self.s1 = TimeValue(numpy.array([0, 1, 2], dtype=TIME_TYPE), numpy.array(['.pau', 'aI', '.pau'], dtype='S8'), 1, 3)
        self.s2 = TimeValue(numpy.array([0, 1, 2], dtype=TIME_TYPE), numpy.array(['.pau', 'oU', '.pau'], dtype='S8'), 1, 3)
        desc = numpy.dtype({"names": ['string', 'int'], "formats": ['S30', numpy.uint8]})  # record arrays
        self.r1 = TimeValue(numpy.array([0, 1], dtype=TIME_TYPE), numpy.array([('abc', 3), ('def', 4)], dtype=desc), 1, 2)
        self.f1 = TimeValue(numpy.array([0, 1], dtype=TIME_TYPE), numpy.array([numpy.arange(3), numpy.arange(4)], dtype=numpy.ndarray), 1, 2)

    def test_init(self):
        t = TimeValue(numpy.empty(0, dtype=TIME_TYPE), numpy.empty(0), 1, 0)  # empty
        t = TimeValue(numpy.empty(0, dtype=TIME_TYPE), numpy.empty(0), 1, 10)  # empty
        self.assertRaises(AssertionError, TimeValue, numpy.array([6, 3], dtype=TIME_TYPE), numpy.array([3, 6]), 1, 10)  # bad times
        self.assertRaises(Exception, TimeValue, numpy.array([3, 6], dtype=TIME_TYPE), numpy.array([3, 6]), 1, 5)  # duration too short

    def test_duration(self):
        self.t1.duration = 11  # ok
        self.assertRaises(Exception, self.t1.set_duration, 5)  # duration too short
        self.s1.duration = 3  # ok
        self.r1.duration = 3  # ok

    def test_eq(self):
        self.assertTrue(self.t1 == self.t1)
        self.assertFalse(self.t1 == self.t2)

    def test_add(self):
        t = self.t1 + self.t2
        self.assertTrue(t.duration == 20)
        self.assertTrue(t.time[5] == 16)
        self.assertTrue(t.value[5] == 8)
        self.t1 += self.t2
        t = self.t1
        self.assertTrue(t.duration == 20)
        self.assertTrue(t.time[5] == 16)
        self.assertTrue(t.value[5] == 8)
        s = self.s1 + self.s2
        r = self.r1 + self.r1

    def test_select(self):
        t = self.t1.select(1, 5)
        self.assertTrue(t == TimeValue(numpy.array([0], dtype=TIME_TYPE), numpy.array([1]), 1, 4))
        t = self.t1.select(1, 6)
        self.assertTrue(t == TimeValue(numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5))
        t = self.t1.select(1, 6)
        self.assertTrue(t == TimeValue(numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5))
        t = self.t1.select(2, 5)
        self.assertTrue(t == TimeValue(numpy.array([], dtype=TIME_TYPE), numpy.array([]), 1, 3))


class TestPartition(unittest.TestCase):
    def setUp(self):
        self.p1 = Partition(numpy.array([0, 5, 6, 10], dtype=TIME_TYPE), numpy.array(["start", "middle", "end"]), 1)
        self.p2 = Partition(numpy.array([0, 5, 10], dtype=TIME_TYPE), numpy.array(["start2", "end2"]), 1)

    def NOtest_init(self):  # should one allow an empty partition?
        p = Partition(numpy.array([0, 1], dtype=TIME_TYPE), numpy.zeros(1, dtype=TIME_TYPE), 1)
        #p = Partition(numpy.empty(0, dtype=TIME_TYPE), numpy.empty(0, dtype=TIME_TYPE), 1)

    def test_insert(self):
        p = Partition(numpy.array([0, 5, 6, 10], dtype=TIME_TYPE), numpy.array(["start", "middle", "end"]), 1)
        #p.insert(0, "even earlier")
        #p.insert(1, "really the end")
        p.insert(7, "still the middle")
        self.assertTrue((p.time == numpy.array([0, 5, 6, 7, 10])).all())

    def test_duration(self):
        self.p1.duration = 12  # ok
        self.p1.duration = 8  # ok
        self.assertRaises(Exception, self.p1.set_duration, 6)  # duration too short

    def test_eq(self):
        self.assertTrue(self.p1 == self.p1)
        self.assertFalse(self.p1 == self.p2)

    def test_add(self):
        p = self.p1 + self.p2
        self.assertTrue(p.duration == 20)
        self.assertTrue(p.time[4] == 15)
        self.assertTrue(p.value[4] == "end2")
        self.p1 += self.p2
        p = self.p1
        self.assertTrue(p.duration == 20)
        self.assertTrue(p.time[4] == 15)
        self.assertTrue(p.value[4] == "end2")

    def test_select(self):
        p = self.p1.select(5, 6)
        self.assertTrue(p == Partition(numpy.array([0, 1], dtype=TIME_TYPE), numpy.array(['middle']), 1))
        p = self.p1.select(5, 7)
        self.assertTrue(p == Partition(numpy.array([0, 1, 2], dtype=TIME_TYPE), numpy.array(['middle', 'end']), 1))
        p = self.p1.select(4, 6)
        self.assertTrue(p == Partition(numpy.array([0, 1, 2], dtype=TIME_TYPE), numpy.array(['start', 'middle']), 1))
        p = self.p1.select(4, 7)
        self.assertTrue(p == Partition(numpy.array([0, 1, 2, 3], dtype=TIME_TYPE), numpy.array(['start', 'middle', 'end']), 1))

    def test_from_TimeValue(self):
        tv = TimeValue(numpy.arange(9, dtype=TIME_TYPE) * 10 + 10, numpy.array([0.0, 1.0, 1.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0]), 1, 100)
        p = Partition.from_TimeValue(tv)
        self.assertTrue((p.time == numpy.array([0, 15, 35, 65, 100])).all())
        self.assertTrue((p.value == numpy.array([0.0, 1.0, 4.0, 8.0])).all())

    def test_merge_same(self):
        p = Partition(numpy.array([0, 3, 6, 10], dtype=TIME_TYPE), numpy.array(["1", "1", "2"]), 1)
        p = p.merge_same()
        self.assertTrue(p.value[1] == "2")


class TestMultiTrack(unittest.TestCase):
    def setUp(self):
        self.e = Event(numpy.array([3, 6], dtype=TIME_TYPE), 1, 10)
        self.w = Wave(numpy.arange(0, 10, dtype=numpy.int16), 1)
        self.t = TimeValue((numpy.linspace(1, 9, 3)).astype(TIME_TYPE), numpy.array([1, 4, 2]), 1, 10)
        self.p = Partition(numpy.array([0, 5, 6, 10], dtype=TIME_TYPE), numpy.array(["start", "middle", "end"]), 1)
        self.m = MultiTrack({"e": self.e, "w": self.w, "t": self.t, "p": self.p})

    def test_str(self):
        str(self.m)

    #def test_add(self):
        #answer = self.multiTrack1 + self.multiTrack2
        #self.assertTrue(self.mResult == answer)
        #answer = copy.copy(self.multiTrack1)
        #answer += self.multiTrack2
        #self.assertTrue(answer == self.mResult)

    def test_resample(self):
        m = self.m.resample(2)
        self.assertTrue(m["e"].duration == m["w"].duration == m["t"].duration == m["p"].duration == 20)
        self.assertTrue(m["e"].time[0] == 6)

    #def test_select1(self):
        #ws = self.wave.select(0, self.wave.duration)
        #self.assertTrue(ws == self.wave)

    #def test_select2(self):
        #tv = TimeValue(numpy.array([1, 5, 9]), numpy.array([1., 4., 2.]), 1, duration=10)
        #tv1 = tv.select(0, 2, interpolation="linear")
        #tv2 = tv.select(2, 10, interpolation="linear")
        #tvs = tv1 + tv2
        #v1 = tv.get(numpy.linspace(0., 10., 11), interpolation="linear").transpose()
        #v2 = tvs.get(numpy.linspace(0., 10., 11), interpolation="linear").transpose()
        #print v1
        #print v2
        #self.assertAlmostEqual( numpy.sum(numpy.abs(v1 - v2)), 0)

    #def test_select3(self):
        #pt = Partition(fs=1, time=numpy.array([0., 6, 12, 18]), value=['.pau','h', 'E'], duration=18)
        #pt1 = pt.select(4.5, 12.5)
        ##print pt1


class TestCrossfade(unittest.TestCase):
    def test_wave(self):
        wav1 = Wave(numpy.array([ 1,  1,  1,  1,  1]), 1)
        wav2 = Wave(numpy.array([10, 10, 10, 10, 10]), 1)
        length = 3
        wav = wav1.crossfade(wav2, length)
        self.assertEqual(wav1.duration + wav2.duration - length, wav.duration)
        self.assertTrue(numpy.allclose(wav.value, numpy.array([1, 1, 3, 5, 7, 10, 10])))

    def test_event(self):
        evt1 = Event(numpy.array([1, 5, 9], dtype=TIME_TYPE), 1, 10)
        evt2 = Event(numpy.array([2, 5, 9], dtype=TIME_TYPE), 1, 10)
        length = 2
        evt = evt1.crossfade(evt2, length)
        self.assertEqual(evt1.duration + evt2.duration - length, evt.duration)
        self.assertTrue(numpy.allclose(evt.time, numpy.array([1, 5, 10, 13, 17])))

    def test_partition(self):
        prt1 = Partition(numpy.array([0, 8, 10], dtype=TIME_TYPE), numpy.array(['1', '2']), 1)
        prt2 = Partition(numpy.array([0, 2, 10], dtype=TIME_TYPE), numpy.array(['3', '4']), 1)
        length = 4
        prt = prt1.crossfade(prt2, length)
        self.assertEqual(prt1.duration + prt2.duration - length, prt.duration)
        self.assertTrue(numpy.allclose(prt.time, numpy.array([0, 8, 16])))
        self.assertTrue((prt.value == numpy.array(['1', '4'])).all())

    def test_timeValue(self):
        pass  # TODO: implement me!


def example_wave_audio():
    print('playing files')
    Wave.wav_read('test-mono.wav').play()
    Wave.wav_read('test-stereo.wav').play()
    print('recording and then playing 3 seconds (mono)')
    Wave.record(1, 22050, 3).play()
    print('recording and then playing 3 seconds (stereo)')
    Wave.record(2, 22050, 3).play()



if __name__ == "__main__":
    #unittest.main()
    #viewer()
    #example_wave_draw()
    #trk = Track.read('../../../dat/test-mwm.wav')
    #trk = Track.read('../../../dat/test-mwm.lab')
    #print(trk)
    print([t.__name__ for t in get_track_classes()])
