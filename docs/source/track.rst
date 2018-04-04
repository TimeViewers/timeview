Tracks
======

TimeView supports three different kind of :index:`track` objects:

1. Wave (e.g. an audio waveform, sampled at a *uniform* interval)
2. Time-Value data (e.g. a contour, sampled at *non-uniform* intervals)
3. Partition (also known as Segmentation, e.g. a phonetic segmentation)

At the moment TimeView supports loading of Waveforms in .wav format, Partitions in .lab format, and EDF files in .edf format.

A user can load a track from disk, and see additional information about the current track by selecting the appropriate entries from the menu.

A user can also create a new Partition track.

A view is a particular way of displaying a track's information, according to a *rendering* strategy. One or more views can be shown in a panel.
