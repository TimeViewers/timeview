.. TimeView documentation master file, created by
   sphinx-quickstart on Wed Sep 20 15:45:33 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TimeView Documentation
======================

Welcome to TimeView, a cross-platform (Windows, MacOS, Linux) desktop application for viewing and editing
time-based signals such as waveform, time-value, and segmentation data.
It is possible to "undock" this help window from the main application window, by clicking on the "two windows" icon on the top right.
Now you can move and resize the help window.

After starting the application, you are presented with a single empty panel.

First Steps using Speech Data
-----------------------------

Open the supplied speech waveform by selecting Track/Open from the menu.
Select "speech-mwm.wav" in the supplied "example" directory.
The waveform should display using the default time-domain based Waveform rendering.
You can now navigate time using either
(1) using the mouse, by left-click dragging in the view area to *move*, or by right-click dragging or using the wheel to *zoom*,
(2) the Navigation menu, or
(3) the associated arrow-based keyboard shortcuts.

Change the Rendering to "Spectrogram", and navigate around.
You can change the color of the spectrogram by clicking on the "color" table cell.
You can change the frame *size* (and other parameters) by choosing Track/Options from the menu;
this sets configuration parameters for the current view.
The frame *rate* is automatically updated dependent on the number of available pixels, to always give you the highest time-resolution possible.
You can also increase or decrease the panel vertical height from the Panel menu or the associated shortcuts.
In the case of rendering the spectrogram, this may increase the FFT size to always give you the highest frequency-resolution possible.

Sometimes it is useful to look at the same data at *different* time locations.
To do this, uncheck the Panel/Synchronize checkbox in the menu.
Link the current view by right-clicking on the file name and then selecting "Link Track / Link to New Panel".
Notice that the time axes of the two panels are now independent.
Check the synchronize checkbox again, and notice that the currently selected panel sets the time region of the other panels.
Now close the second panel by selecting "Panel / Remove Panel" from the menu, or using the associated shortcut.

Focus in on a spectrogram frequence range of interest by calling up the options and setting "y_max" to "8000".

Let's find regions of activity by running the "Activity Detector" processor from the Processing Menu.
A dialogue box will appear. Accept the defaults and click "Process".
Two more tracks will be added, the activity segmentation (a Partition-type track) with "0" and "1" labels,
and the maximum energy per frame (a Time-Value-type track),
the treshold on which the activation detection is based.
To see the units of energy in dB, make sure the TimeValue track is currently selected,
as the y-axis is always labeled with the currently selected view.

You may want to see the spectrogram and the waveform at the same time.
To do this, right-click on the speech.wav filename and select "Link to New Panel" and set the rendering to "Waveform".
Now link the "speech-mwm-act.lab" file to the second panel as well.
Finally, hide the energy view by unchecking the "Show" checkbox.
This leaves just the spectrogram and the segmentation visible.

Let's say that you are not completely satisfied with the segmentation, and would like to adjust its boundaries.
Change one (or both) of the partition rendering to "editable", and drag the boundaries around.
Notice how the two segmentations remain synchronized.
To remove a boundary, double-click it.
To add a new boundary, double-click between existing boundaries, and you can start typing the new label for this segment.
You can click any existing segment label to edit it.
TAB and shift-TAB cycles through the segment labels forward and backward.

You can move views from panel to panel anytime by right-clicking on the filename and selecting "Move View".


First Steps using Rodent Vocalization Data
------------------------------------------

Open the supplied rodent audio waveform by selecting Track/Open from the menu.
Select "rodent-E1023.wav" in the supplied "example" directory.
The waveform should display using the default time-domain based rendering.
Change the Rendering to "Spectrogram" and change

, and navigate around, by using either
(1) using the mouse, by left-click dragging in the view area to *move*, or by right-click dragging or using the wheel to *zoom*,
(2) the Navigation menu, or
(3) the associated arrow-based keyboard shortcuts.

 Zoom in on the region around 40 seconds.
Change the frame_size of the spectrogram by clicking on the sliders icon to configure the parameters.
Set "frame_size"=0.001.

. Let's remove additive noise by running the "Noise Reducer" Processor from the Processing Menu.
. A dialogue box will appear. Accept the defaults and click "Process" - this may take a while.
. After the output appears, change the output rendering to "Spectrogram".
. Remove the original by selecting it and clicking on the "delete" pushbutton.

Let's find regions of activity by running the "Activity Detector" processor from the Processing Menu.
A dialogue box will appear.
Change the defaults to reflect "threshold" = -24.5, "smooth" = 0.5, "frame_size"=0.001, "frame_rate"=0.001, and click "Process".
Two more tracks will be added, the activity segmentation (a Partition-type track), and the maximum energy per frame (a Time-Value-type track), on which the activation detection is based.
Hide the energy track, and change the rendering for the segmentation to "editable".
Now you can change boundaries and labels as in the speech example.

Finally, let's run the "Peak Tracker (active regions only)" Processor.
Specify "freq_min" = 40000, "freq_max" = 120000, "frame_size" = 0.001, "frame_rate" = 0.001, "smooth"=0.1,and "NFFT" = 512.
After a while, you will see a Time-Value object appear that tracks the spectral peaks.

Table of Contents
-----------------

(also always available on the left)

.. toctree::
   :maxdepth: 2

   track.rst
   panel.rst
   view.rst
   process.rst
   dataset.rst



..
   Indices and tables
   ==================
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
