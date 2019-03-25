TimeView
=

![screenshot](docs/source/TimeView.png)

Overview
-
Timeview is a cross-platform desktop application for viewing and editing
Waveforms, Time-Value data, and Segmentation data. 
These data can easily be analyzed or manipulated using a library of built-in processors;
for example, a linear filter can operate on a waveform, or an activity detector can create a segmentation from a waveform.
Processors can be easily customized or created created from scratch.

This is a very early preview, and is not suitable for general usage yet.


Features
-
* *Cross-platform*, verified to run on macOS, Linux, and Windows
* Flexible arrangement of any number of *panels*, 
which contain any number of superimposed *views* (e.g. waveforms, spectrograms, feature trajectories, segmentations)
* Views can easily be *moved* between panels
* Views can be *linked* so that modifications in one panel are reflected in other panels
* *Customizable Rendering* of views (e.g. frame_size for spectrogram)
* *On-the-fly Spectrogram* rendering automatically adjusts frame-rate and FFT-size to calculate information for each available pixel without interpolation
* *Editable segmentation* (insertion, deletion, modification of boundaries; modification of labels)
* Basic *processing plug-ins* are provided (e.g. activity detection, F0-analysis)
* Processing plug-ins are easily *customizable* or *extendable* using python (bridging to R via `rpy2` is also possible, an example is provided)
* API allows accessing processing plugins for *batch file processing* or *preconfiguring the GUI* (examples are provided)
* *EDF-file-format* support
* A *dataset-manager* allows grouping of files into datasets, for quick access to often-used files
* *Command Line Interface* support, for easy chaining with other tools

An introductory video is available at: https://vimeo.com/245480108


Installation
-
From an empty python 3.6+ python environment run

```
$ pip install git+https://github.com/lxkain/timeview
$ timeview
$ timeview -h
```

Development Environment
-
In your 3.6+ python environment run

```
$ git clone https://github.com/lxkain/timeview.git timeview
$ cd timeview/timeview
$ python __main__.py
```

Help
-
After the application has started, select "Help" from the Menu, and then "TimeView Help" to learn more.
