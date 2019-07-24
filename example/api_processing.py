#!/usr/bin/env python3

"""
Example of using the TimeView python API to process files without the GUI

Activate the conda timeview environment before running this
"""

# TODO: this should be moved to signalworks, and plugins should be defined there

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from timeview.api import Track, processing


wav_name = Path(__file__).with_name('speech-mwm.wav')
wav = Track.read(wav_name)
par_name = Path(__file__).with_name('speech-mwm.lab')
par = Track.read(par_name)

processor = processing.Filter()
processor.set_data({'wave': wav})
results, = processor.process()

print(results)
