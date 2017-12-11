Views
======

A :index:`view` is comprised of a track, a specific method to display the track's information in the form of a *renderer*, and other information (e.g. color).
The view area on the left of the panel shows one or more overlapping rendering outputs, while the view table on the right side of the panel shows the currently active views.

Each view may offer several different types of rendering. A view can be also be shown or hidden, and its colors and y-axis range can be changed.
All these actions can be accomplished by clicking on the associated table cells.

To add a new view and thus also a new track, click the "+" button. This is identical to choosing File/Open.
To remove a view, select the view to be removed, and click the "-" button.


Navigation
----------
Moving around in time is easy. A user can move forward/right, backward/left, and zoom in and out.
It is also possible to go to the very beginning or end of the track.
Finally, the zoom level can be set to "1:1", which means that the resolution of the screen is equal to the resolution of the underlying signals,
and the zoom level can be set to "Fit", which means that the zoom level will be set such that all tracks are visible.

It is also possible to drag the mouse cursor left and right to move forward or backward in time, or to use the mouse wheel to change zoom levels.
These actions are also available via the menu, and finally also via convenient keyboard shortcuts using the arrow keys.


Rendering
---------

The following Renderers are available:

* Waveform: Time-Domain representation
  This renderer features supersampling for fast display of even very large waveforms.

* Waveform: Frequency-Domain representation / Spectrogram
  The frame rate of the spectrogram is set automatically such that additional detail becomes available when zooming in.

* Time-Value: Time-Domain representation

* Partition: read-only
  Read-only rendering is ideal for viewing-only

* Partition: editable
  This renderer allows editing of label boundaries and values.
  To *move* boundaries, simply drag the vertical line after it has turned red.
  To *remove* a boundary, double-click the vertical line after it has turned red.
  To *add* a new boundary, double-click in-between existing lines, but be sure none of them have turned red. If necessary, zoom in more.
  To *edit* a label value, double-click on the text-box. The TAB key, and shift-TAB allow for quick back and forth selection of values.

It is relatively easy to write additional renderers for custom visualizations.
