# LOFAR single station imaging
This repository contains code for imaging LOFAR data with direct Fourier transforms. This is a straightforward way to image single station data, or data from a limited number of baselines. Both sky images and near-field (ground) images are supported.

For imaging larger LOFAR data sets (in Measurement Sets), see the LOFAR imaging cookbook.

Much of the code in this repository was originally written by Vanessa Moss, based on the tutorial by Michiel Brentjens (github.com/brentjens).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lofar-astron/lofarimaging/master?filepath=lofarimaging.ipynb)

### Installation
Download or clone this repository. To install requirements, run

```
>>> pip install -r requirements.txt
```

If you want to use calibration tables, download them from ASTRON, e.g. through

```
>>> svn co https://svn.astron.nl/Station/trunk/CalTables
```

### To start the notebook
Open the notebook in a Jupyter notebook (or Jupyterlab) instance. You can start such an instance with 
`>> jupyter notebook`

### To run the entire notebook
`Kernel > Restart & Run All`

### Notes
This code ships with an example XST-dataset (`20170720_095816_mode_3_xst_sb297.dat`), obtained using station DE603LBA, specifically collecting data in subband 297 (58 MHz). It should run on any other XST-dataset, provided the station name is changed accordingly. By default, the code visualises timestep #0 of the 30 timesteps of integration, but this can also be modified to any other time slice.

### Release: Version 1.5

 * Overlay the ground plot on a satellite image (both in a static PNG and as a Leaflet overlay in the notebook)
 * Annotate CygA, CasA and Sun on the sky plot
 * Support HBA imaging using one tile per element
 * Use station rotations from lofarantpos
 * Rotate the antennas, not the image
 * Show longitude and latitute of maximum pixel
 * Speed up the code somewhat by using numba and numexpr
 * Move some code from the notebook to a python file

### Release: Version 1.4
Update to automatically parse some information from data file names, based on the wrapper script written by Mattia Mancini for recording station data (e.g. https://svn.astron.nl/viewvc/LOFAR/trunk/LCU/StationTest/rspctlprobe.py). Some formatting removed for compatibility with different operating systems. To obtain the LOFAR antenna database needed for antenna positions, please install: `pip install lofarantpos`.

### Release: Version 1.3
Notebook has been updated to include station calibration for LBA (HBA to come later). This requires the "caltables" folder to be in the same directory as the notebook, but that can be changed as long as the location is changed also in the function. The calibration tables themselves are not included in this repository due to their size, but you can access them [here](http://astron.nl/~moss/caltables.zip) (~65 MB download): , or download them from their [repository](https://svn.astron.nl/Station/trunk/CalTables).
