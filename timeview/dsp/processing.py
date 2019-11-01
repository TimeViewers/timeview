import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable

import numpy as np
from scipy import signal


from . import tracking
from . import dsp
from . import viterbi

Tracks = Union[tracking.Wave, tracking.TimeValue, tracking.Partition]
# type alias


class InvalidDataError(Exception):
    pass


class InvalidParameterError(Exception):
    pass


class ProcessError(Exception):
    pass


class DefaultProgressTracker(object):
    def update(self, value: int):
        print(f"{value}%", end='...', flush=True)


class Processor(metaclass=ABCMeta):
    name = "Processor"
    acquire: Dict[str, object] = {}  # TODO: how to merge with data?

    def __init__(self):
        self.data: Dict[str, Tracks] = {}
        self.parameters: Dict[str, Tracks] = {}  # default parameters
        self.progressTracker = None

    def set_data(self, data: Dict[str, Tracks]) -> None:
        for key in self.acquire.keys():
            if type(data[key]) != self.acquire[key]:  # check type
                raise InvalidDataError
        self.data = data

    def get_parameters(self) -> Dict[str, str]:
        # default parameters can be modified here based on the data
        return {k: str(v) for k, v in self.parameters.items()}

    def set_parameters(self, parameters: Dict[str, str]) -> None:
        if __debug__:
            for name, value in parameters.items():
                logging.debug(f'Received parameter {name} of value {value}')
        try:
            for key in parameters.keys():
                if type(self.parameters[key]) == np.ndarray:
                    self.parameters[key] = np.fromstring(parameters[key]
                                                         .rstrip(')]')
                                                         .lstrip('[('),
                                                         sep=' ')
                else:
                    self.parameters[key] =\
                        type(self.parameters[key])(parameters[key])
        except Exception as e:
            raise InvalidParameterError(e)
        # additional parameter checking can be performed here

    def process(self, progressTracker=None):  # -> Tuple[Tracks]:
        """
        :param progressTracker: call process_tracker.updateProgress(int)
        with an integer between 0-100 indicating how far process is
        :return:
        """
        if progressTracker is None:
            self.progressTracker = DefaultProgressTracker()
        else:
            self.progressTracker = progressTracker

    def del_data(self):
        self.data = {}


def get_processor_classes() -> Dict[str, Callable[..., Processor]]:
    def all_subclasses(c):
        return c.__subclasses__() + [a for b in c.__subclasses__()
                                     for a in all_subclasses(b)]

    return {obj.name: obj for obj in all_subclasses(Processor)}


#######################################################################################################################

#
# class ConverterToFloat64(Processor):
#     name = 'Conversion to Float64'
#     acquire = {'wave': tracking.Wave}
#
#     def process(self, **kwargs) -> Tuple[tracking.Wave]:
#         Processor.process(self, **kwargs)
#         wav = self.data['wave']
#         wav = wav.convert_dtype(np.float64)
#         wav.path = wav.path.with_name(wav.path.stem + '-float64').with_suffix(
#             tracking.Wave.default_suffix)
#         return wav,
#
#
# class ConverterToInt16(Processor):
#     name = 'Conversion to Int16'
#     acquire = {'wave': tracking.Wave}
#
#     def process(self, **kwargs) -> Tuple[tracking.Wave]:
#         Processor.process(self, **kwargs)
#         wav = self.data['wave']
#         wav = wav.convert_dtype(np.int16)
#         wav.path = wav.path.with_name(wav.path.stem + '-int16').with_suffix(
#             tracking.Wave.default_suffix)
#         return wav,
#

class Filter(Processor):
    name = 'Linear Filter'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        # default is pre-emphasis
        self.parameters = {'B': np.array([1., -.95]),
                           'A': np.array([1.])}

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        wav = self.data['wave']
        x = wav.value
        self.progressTracker.update(10)
        y = signal.lfilter(self.parameters['B'],
                           self.parameters['A'], x).astype(x.dtype)
        self.progressTracker.update(90)
        new_track = tracking.Wave(y,
                                  fs=wav.fs,
                                  path=wav.path
                                          .with_name(wav.path.stem + '-filtered')
                                          .with_suffix(tracking.Wave
                                                               .default_suffix)),
        return new_track


class ZeroPhaseFilter(Filter):
    name = 'Zero-phase Linear Filter'
    acquire = {'wave': tracking.Wave}

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        wav = self.data['wave']
        x = wav.value
        self.progressTracker.update(10)
        y = signal.filtfilt(self.parameters['B'],
                            self.parameters['A'], x).astype(x.dtype)
        self.progressTracker.update(90)
        return tracking.Wave(y,
                             fs=wav.fs,
                             path=wav.path
                                     .with_name(wav.path.stem +
                                                '-0phasefiltered')
                                     .with_suffix(wav.path.suffix)),


class EnergyEstimator(Processor):
    name = 'RMS-Energy (dB)'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {'frame_size': 0.020,  # in seconds
                           'frame_rate': 0.010}  # in seconds

    def process(self, **kwargs) -> Tuple[tracking.TimeValue]:
        Processor.process(self, **kwargs)
        wav = self.data['wave']
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        frame = dsp.frame(wav,
                          self.parameters['frame_size'],
                          self.parameters['frame_rate'])
        self.progressTracker.update(70)
        frame.value *= signal.hann(frame.value.shape[1])
        value = 20 * np.log10(np.mean(frame.value ** 2.0, axis=1) ** 0.5)
        self.progressTracker.update(90)
        nrg = tracking.TimeValue(frame.time,
                                 value,
                                 wav.fs,
                                 wav.duration,
                                 path=wav.path
                                         .with_name(wav.path.stem + '-energy')
                                         .with_suffix(tracking.TimeValue
                                                              .default_suffix))
        nrg.min = value.min()
        nrg.max = value.max()
        nrg.unit = 'dB'
        return nrg,


class SpectralDiscontinuityEstimator(Processor):
    name = 'Spectral Discontinuity Estimator'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {'frame_size': 0.005,  # seconds, determines freq res.
                           'NFFT': 256,
                           'normalized': 1,
                           'delta_order':1}

    def process(self, **kwargs) -> Tuple[tracking.TimeValue]:
        Processor.process(self, **kwargs)
        # wav = self.data['wave']
        wav: tracking.Wave = self.data['wave']
        self.progressTracker.update(10)
        ftr, time, frequency = dsp.spectrogram(wav,
                                               self.parameters['frame_size'],
                                               self.parameters['frame_size'],  # frame_rate = frame_size
                                               NFFT=self.parameters['NFFT'],
                                               normalized=self.parameters['normalized'])
        if self.parameters['normalized']:
            ftr = ftr - np.mean(ftr, axis=1).reshape(-1, 1)

        time = (time[:-1] + time[1:]) // 2
        assert self.parameters['delta_order'] > 0
        dynamic_win = np.arange(-self.parameters['delta_order'], self.parameters['delta_order'] + 1)

        win_width = self.parameters['delta_order']
        win_length = 2 * win_width + 1
        den = 0
        for s in range(1, win_width+1):
            den += s**2
        den *= 2
        dynamic_win = dynamic_win / den

        N, D = ftr.shape
        print(N)
        temp_array = np.zeros((N + 2 * win_width, D))
        delta_array = np.zeros((N, D))
        self.progressTracker.update(90)
        temp_array[win_width:N+win_width] = ftr
        for w in range(win_width):
            temp_array[w, :] = ftr[0, :]
            temp_array[N+win_width+w,:] = ftr[-1,:]

        for i in range(N):
            for w in range(win_length):
                delta_array[i, :] += temp_array[i+w,:] * dynamic_win[w]
        value = np.mean(np.diff(delta_array, axis=0) ** 2, axis=1) ** 0.5
        dis = tracking.TimeValue(time, value, wav.fs, wav.duration, path=wav.path.with_name(wav.path.stem + '-discont')
                                 .with_suffix(tracking.TimeValue
                                              .default_suffix))
        dis.min = 0
        dis.max = value.max()
        dis.unit = 'dB'
        dis.label = 'spectral discontinuity'
        self.progressTracker.update(100)
        return dis,



class NoiseReducer(Processor):
    name = 'Noise Reducer'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {'silence_percentage': 10,
                           'frame_rate': 0.01}  # in seconds

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        inp = self.data['wave']
        inp = inp.convert_dtype(np.float64)
        self.progressTracker.update(20)
        out = dsp.spectral_subtract(inp, self.parameters['frame_rate'], self.parameters['silence_percentage'])  # TODO: pull this up into here
        self.progressTracker.update(90)
        out.path = inp.path.with_name(inp.path.stem + '-denoised').with_suffix(
                                      tracking.Wave.default_suffix)
        return out,


class ActivityDetector(Processor):
    name = 'Activity Detector'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {'threshold': -30.0,
                           'smooth': 1.0,
                           'frame_size': 0.020,  # in seconds
                           'frame_rate': 0.01}

    def process(self, **kwargs) -> Tuple[tracking.Partition, tracking.TimeValue]:
        Processor.process(self, **kwargs)
        wav = self.data['wave']
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        M, time, frequency = dsp.spectrogram(wav,
                                             self.parameters['frame_size'],
                                             self.parameters['frame_rate'])
        self.progressTracker.update(20)
        # Emax = np.atleast_2d(np.max(M, axis=1)).T
        Emax = 20 * np.log10(np.mean((10 ** (M / 10)), axis=1) ** 0.5)
        P = np.empty((len(Emax), 2))
        P[:, 0] = 1 / (1 + np.exp(Emax - self.parameters['threshold']))
        P[:, 1] = 1 - P[:, 0]  # complement
        self.progressTracker.update(30)
        seq, _ = viterbi.search_smooth(P, self.parameters['smooth'])
        self.progressTracker.update(90)
        tmv = tracking.TimeValue(time, seq, wav.fs, wav.duration,
                                 wav.path.with_name(wav.path.stem + '-act')
                                 .with_suffix(
                                     tracking.TimeValue.default_suffix))
        par = tracking.Partition.from_TimeValue(tmv)
        par.value = np.char.mod('%d', par.value)
        emax = tracking.TimeValue(time, Emax, wav.fs, wav.duration,
                                  wav.path.with_name(wav.path.stem + '-emax')
                                  .with_suffix(
                                      tracking.TimeValue.default_suffix))
        emax.min = Emax.min()
        emax.max = Emax.max()
        emax.unit = 'dB'
        emax.label = 'maximum frequency magnitude'
        return par, emax


class F0Analyzer(Processor):
    name = 'F0 Analysis'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.t0_min = 0
        self.t0_max = 0
        self.parameters = {'smooth': 0.01,
                           'f0_min': 51,  # in Hertz
                           'f0_max': 300,  # in Hertz
                           'frame_size': 0.040,  # in seconds
                           'frame_rate': 0.010,  # in seconds
                           'dop threshold': 0.7,
                           'energy threshold': 0.1}

    def set_parameters(self, parameter: Dict[str, str]):
        super().set_parameters(parameter)
        assert self.parameters['f0_min'] < self.parameters['f0_max'],\
            'f0_min must be < f0_max'
        assert self.parameters['frame_size'] >\
            (2 / self.parameters['f0_min']), 'frame_size must be > 2 / f0_min'

    def process(self, **kwargs) -> Tuple[tracking.TimeValue,
                               tracking.TimeValue,
                               tracking.Partition]:
        Processor.process(self, **kwargs)
        wav = self.data['wave']
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        R, time, frequency = dsp.correlogram(wav,
                                             self.parameters['frame_size'],
                                             self.parameters['frame_rate'])

        self.progressTracker.update(30)
        t0_min = int(round(wav.fs / self.parameters['f0_max']))
        t0_max = int(round(wav.fs / self.parameters['f0_min']))
        index = np.arange(t0_min, t0_max + 1, dtype=np.int)
        E = R[:, 0]  # energy
        R = R[:, index]  # only look at valid candidates
        # normalize
        R -= R.min()
        R /= R.max()
        # find best sequence
        seq, _ = viterbi.search_smooth(R, self.parameters['smooth'])
        self.progressTracker.update(80)
        # if 0:
        #     from matplotlib import pyplot as plt
        #     plt.imshow(R.T, aspect='auto', origin='lower', cmap=plt.cm.pink)
        #     plt.plot(seq)
        #     plt.show()
        # F0 track
        f0 = wav.fs / (t0_min + seq)
        # degree of periodicity
        dop = R[np.arange(R.shape[0]), seq]
        # voicing
        v = ((dop > self.parameters['dop threshold']) &
             (E > self.parameters['energy threshold'])
             #  (seq > 0) & (seq < len(index) - 1)
             ).astype(np.int)
        v = signal.medfilt(v, 5)  # TODO: replace by a 2-state HMM
        f0[v == 0] = np.nan
        # prepare tracks
        f0 = tracking.TimeValue(time, f0, wav.fs, wav.duration,
                                wav.path
                                   .with_name(wav.path.stem + '-f0')
                                   .with_suffix(tracking.TimeValue
                                                        .default_suffix))
        f0.min = self.parameters['f0_min']
        f0.max = self.parameters['f0_max']
        f0.unit = 'Hz'
        f0.label = 'F0'
        dop = tracking.TimeValue(time, dop, wav.fs, wav.duration,
                                 wav.path
                                 .with_name(wav.path.stem + '-dop')
                                 .with_suffix(
                                     tracking.TimeValue.default_suffix))
        dop.min = 0
        dop.max = 1
        dop.label = 'degree of periodicity'
        vox = tracking.TimeValue(time, v, wav.fs, wav.duration,
                                 wav.path
                                 .with_name(wav.path.stem + '-vox')
                                 .with_suffix(
                                     tracking.TimeValue.default_suffix))
        vox = tracking.Partition.from_TimeValue(vox)
        vox.label = 'voicing'
        return f0, dop, vox


class Differentiator(Processor):
    name = "Differentiator"
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {}

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        wav: tracking.Wave = self.data['wave']
        trk = tracking.Wave(wav.value.copy(), wav.fs, wav.duration, path=wav.path.with_name(wav.path.stem + '-diff'))
        trk.value = np.gradient(trk.value)
        trk.max = wav.max
        trk.min = wav.min
        trk.unit = wav.unit
        trk.label = wav.label
        return trk,


class PeakTracker(Processor):
    name = 'Peak Tracker'
    acquire = {'wave': tracking.Wave}

    def __init__(self):
        super().__init__()
        self.parameters = {'smooth': 1.,
                           'freq_min': 100,
                           'freq_max': 1000,
                           'frame_size': 0.02,  # seconds, determines freq res.
                           'frame_rate': 0.01,
                           'NFFT': 512}

    def get_parameters(self):
        if 'wave' in self.data:
            self.parameters['freq_max'] = self.data['wave'].fs / 2
        return super().get_parameters()

    def set_parameters(self, parameter: Dict[str, str]):
        super().set_parameters(parameter)
        if not self.parameters['freq_min'] < self.parameters['freq_max']:
            raise InvalidParameterError('freq_min must be < freq_max')

    def process(self, **kwargs) -> Tuple[tracking.TimeValue]:
        Processor.process(self, **kwargs)
        # wav = self.data['wave']
        wav: tracking.Wave = self.data['wave']
        self.progressTracker.update(10)
        ftr, time, frequency = dsp.spectrogram(wav,
                                               self.parameters['frame_size'],
                                               self.parameters['frame_rate'],
                                               NFFT=self.parameters['NFFT'])
        self.progressTracker.update(50)
        a = frequency.searchsorted(self.parameters['freq_min'])
        b = frequency.searchsorted(self.parameters['freq_max'])
        # import time as timer
        # print('searching')
        # tic = timer.time()
        seq, _ = viterbi.search_smooth(ftr[:, a:b], self.parameters['smooth'])
        self.progressTracker.update(90)
        # toc = timer.time()
        # print(f'done, took: {toc-tic}')
        trk = tracking.TimeValue(time, frequency[a + seq], wav.fs, wav.duration,
                                 wav.path
                                    .with_name(wav.path.stem + '-peak')
                                    .with_suffix(
                                        tracking.TimeValue.default_suffix))
        trk.min = 0
        trk.max = wav.fs / 2
        trk.unit = 'Hz'
        trk.label = 'frequency'
        return trk,


class PeakTrackerActiveOnly(PeakTracker):
    name = 'Peak Tracker (active regions only)'
    acquire = {'wave': tracking.Wave, 'active': tracking.Partition}

    def process(self, **kwargs) -> Tuple[tracking.TimeValue]:
        peak = super().process(**kwargs)[0]
        active = self.data['active']
        for i in range(len(active.time) - 1):
            if active.value[i] in ['0', 0]:
                a = np.searchsorted(peak.time / peak.fs, active.time[i] / active.fs)
                b = np.searchsorted(peak.time / peak.fs, active.time[i+1] / active.fs)
                peak.value[a:b] = np.nan
        return peak,


# from rpy2 import robjects
# from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
# class ExampleR(Processor):
#     name = 'Example R plug-in'
#     acquire = {'activity partition': tracking.Partition,
#                'peak track': tracking.TimeValue}
#
#     def __init__(self):
#         super().__init__()
#         self.parameters = {'some': 0}
#
#     def process(self, **kwargs) -> Tuple[tracking.Partition]:
#         Processor.process(self, **kwargs)
#         par: tracking.Partition = self.data['activity partition']
#         tmv: tracking.TimeValue = self.data['peak track']
#         fs = max(par.fs, tmv.fs)
#         par = par.resample(fs)
#         tmv = tmv.resample(fs)
#         self.progressTracker.update(10)
#         with open(Path(__file__).resolve().parent / 'example.R') as f:
#             script = f.read()
#         self.progressTracker.update(11)
#         pack = SignatureTranslatedAnonymousPackage(script, 'pack')
#         self.progressTracker.update(12)
#         pack.setup()
#         self.progressTracker.update(13)
#         value =\
#             pack.predict_rodent_class(robjects.FloatVector(par.time / par.fs),
#                                       robjects.FloatVector(par.value),
#                                       robjects.FloatVector(tmv.time / tmv.fs),
#                                       robjects.FloatVector(tmv.value),
#                                       par.fs)
#         self.progressTracker.update(90)
#         par = tracking.Partition(par.time, np.array(value), fs=fs,
#                                  path=par.path
#                                  .with_name(par.path.stem + '-class')
#                                  .with_suffix(
#                                      tracking.Partition.default_suffix))
#         return par,