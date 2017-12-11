#!/usr/bin/env python3

"""
Example of using the TimeView python API to process files without the GUI

Activate the conda timeview environment before running this
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from timeview.api import Track, processing


if 0:  # speech
    wav_name = Path(__file__).with_name('speech-mwm.wav')
    wav = Track.read(wav_name)
    par_name = Path(__file__).with_name('speech-mwm.lab')
    par = Track.read(par_name)

    processor = processing.Filter()
    processor.set_data({'wave': wav})
    results, = processor.process()

    print(results)


# rodent
par_name = Path(__file__).with_name('rodent-E1023.lab')
par = Track.read(par_name)
tmv_name = Path(__file__).with_name('rodent-E1023.tmv')
tmv = Track.read(tmv_name)

processor = processing.RodentCallClassifier()
processor.set_data({'activity partition': par, 'peak track': tmv})
results, = processor.process()

print(results)
