#!/usr/bin/env python3

"""
Example of using the TimeView python API to configure the GUI

Activate the conda timeview environment before running this
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from timeview.api import Track, Wave, TimeView


# example setup

# read from disk
wav = Track.read(Path(__file__).with_name('speech-mwm.wav'))
lab = Track.read(Path(__file__).with_name('speech-mwm.lab'))

# create ourselves
fs = 16000
x = np.zeros(2 * fs, dtype=np.float64)
x[1 * fs] = 1
syn = Wave(x, fs)

app = TimeView()
app.add_view(wav, 0, y_min=-10_000, y_max=10_000)
app.add_view(lab, 0)
app.add_view(wav, 1, renderer_name='Spectrogram')  # linked
app.add_view(lab, 1)  # linked
app.add_view(syn, 2)
app.add_view(syn, 2, renderer_name='Spectrogram', y_max=4000)  # linked

app.start()
